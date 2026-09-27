import math
from typing import Literal, Callable
import torch
from transformers import PreTrainedTokenizerBase
import torch.nn as nn
from .tokenize_prompt_and_output import tokenize_prompt_and_output
from .get_response_log_probs import get_response_log_probs
from .compute_rollout_rewards import compute_rollout_rewards
from .compute_group_normalized_rewards_grpo import compute_group_normalized_rewards
from .compute_policy_gradient_loss_on_policy import compute_policy_gradient_loss
from .aggregate_loss_across_microbatch_sequence import aggregate_loss_across_microbatch

def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    数据流是 ``rewards [B] -> advantages [B] -> token tensors [B, T]``；
    每个 microbatch 只做一次 forward/backward，所有 microbatch 完成后再更新优化器。

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if importance_reweighting_method != "none" and old_log_probs is None:
        raise ValueError(
            f"old_log_probs must be provided for {importance_reweighting_method}"
        )
    if importance_reweighting_method in {"grpo", "gspo"} and cliprange is None:
        raise ValueError(
            f"cliprange must be provided for {importance_reweighting_method}"
        )
    if len(repeated_prompts) != len(rollout_responses) or len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("prompts, responses, and ground truths must have equal lengths")
    if len(rollout_responses) % gradient_accumulation_steps != 0:
        raise ValueError("rollout batch must divide evenly across microbatches")
    if old_log_probs is not None and old_log_probs.shape[0] != len(rollout_responses):
        raise ValueError("old_log_probs must have one row per rollout response")

    device = next(model.parameters()).device
    # 先保存原始 microbatch 大小；删除零 advantage 样本后用它减少累积步数。
    original_batch_size = len(rollout_responses)
    original_microbatch_size = original_batch_size // gradient_accumulation_steps

    # reward 在 tokenization 前计算，这样零 advantage 的回答可以直接跳过模型 forward。
    raw_rewards, reward_metadata = compute_rollout_rewards(  # raw_rewards: [B]。
        reward_fn,
        rollout_responses,
        repeated_ground_truths
    )


    advantages, advantage_metadata = compute_group_normalized_rewards(  # advantages: [B]。
        raw_rewards,
        group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer
    )

    # 零 advantage 的 response 对 policy loss 没有贡献；保留其余样本的原始顺序。
    keep_mask = advantages != 0  # [B] bool，标出保留的回答。
    num_kept = int(keep_mask.sum())  # Python 整数标量。
    keep_indices = keep_mask.nonzero(as_tuple=False).flatten().tolist()  # [B_kept] 的 Python 索引列表。

    repeated_prompts = [
        repeated_prompts[i] for i in keep_indices
    ]

    rollout_responses = [
        rollout_responses[i] for i in keep_indices
    ]

    repeated_ground_truths = [
        repeated_ground_truths[i] for i in keep_indices
    ]

    advantages = advantages[keep_mask]  # [B_kept]。
    if old_log_probs is not None:
        old_log_probs = old_log_probs[keep_mask.to(old_log_probs.device)]

    if num_kept == 0:
        # 整个 batch 没有梯度信号时，跳过 forward/backward 和 optimizer.step。
        optimizer.zero_grad()
        return torch.zeros((), device=device), {  # loss 是零维标量张量 []。
            **reward_metadata,
            **advantage_metadata,
        }

    # 只为有效样本构造 [B, T] 的 input、label 和 response mask。
    tokenized = tokenize_prompt_and_output(
        repeated_prompts,
        rollout_responses,
        tokenizer
    )

    input_ids = tokenized["input_ids"].to(device)  # [B_kept, T]。
    labels = tokenized["labels"].to(device)  # [B_kept, T]。
    response_mask = tokenized["response_mask"].to(device)  # [B_kept, T] bool。
    advantages = advantages.to(device)  # [B_kept]。
    if old_log_probs is not None:
        old_log_probs = old_log_probs.to(device)
    metadata: dict[str, torch.Tensor | float] = {
        **reward_metadata,
        **advantage_metadata,
    }

    batch_size = input_ids.shape[0]
    # 删除样本后尽量保持原 microbatch 大小，从而降低显存和重复 forward 次数。
    gradient_accumulation_steps = max(
        1,
        math.ceil(batch_size / original_microbatch_size),
    )
    microbatch_size = min(original_microbatch_size, batch_size)
    total_loss = torch.zeros((), device=device)  # 零维标量张量 []，累计各 microbatch loss。
    clip_fraction_values: list[torch.Tensor] = []
    for i in range(0, batch_size, microbatch_size):
        end = i+microbatch_size
        inputs_microbatch = input_ids[i:end]  # [b, T]，b 是当前 microbatch 大小。
        labels_microbatch = labels[i:end]  # [b, T]。
        advantages_microbatch = advantages[i:end]  # [b]。

        # teacher forcing 对已采样 response 重新计算可反向传播的 log-probability。
        log_probs = get_response_log_probs(
            model,
            inputs_microbatch,
            labels_microbatch
        )["log_probs"]  # [b, T]。

        old_log_probs_microbatch = (
            None if old_log_probs is None else old_log_probs[i:end]
        )
        if old_log_probs_microbatch is not None:
            if old_log_probs_microbatch.shape[0] != log_probs.shape[0]:
                raise ValueError(
                    "old_log_probs and current log_probs must have the same batch size"
                )
            if old_log_probs_microbatch.shape[1] < log_probs.shape[1]:
                raise ValueError("old_log_probs is shorter than current log_probs")
            # Filtering can reduce the batch's padded length; crop old values
            # to the same right-padded token width as the current forward pass.
            old_log_probs_microbatch = old_log_probs_microbatch[:, :log_probs.shape[1]]

        response_mask_microbatch = response_mask[i:end]

        # response-level advantage 会广播到每个有效 response token。
        loss_per_token, loss_metadata = compute_policy_gradient_loss(
            advantages_microbatch,
            log_probs,
            importance_reweighting_method,
            old_log_probs_microbatch,
            cliprange,
            response_mask_microbatch
        )  # loss_per_token: [b, T]。

        micro_loss = aggregate_loss_across_microbatch(  # 零维标量张量 []。
            loss_per_token,
            response_mask[i:end],  # [b, T]。
            loss_normalization,
            normalization_constant
        )
        # sequence 聚合得到的是 microbatch 平均值，需要在累积步之间再平均；
        # constant 聚合已经使用固定的全局 normalization_constant，不能重复除法。
        if loss_normalization == "sequence":
            micro_loss = micro_loss / gradient_accumulation_steps

        micro_loss.backward()
        total_loss += micro_loss.detach()
        metadata.update(loss_metadata)
        if "clip-fraction" in loss_metadata:
            clip_fraction_values.append(loss_metadata["clip-fraction"])

    # 所有 microbatch 的梯度累积完成后再裁剪并执行一次 optimizer 更新。
    grads_norm = None
    if max_grad_norm is not None:
        grads_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    if grads_norm is not None:
        metadata["grad_norm"] = grads_norm.detach()  # 零维标量张量 []。
    if clip_fraction_values:
        metadata["clip-fraction"] = torch.stack(clip_fraction_values).mean().detach()
    return total_loss,metadata



