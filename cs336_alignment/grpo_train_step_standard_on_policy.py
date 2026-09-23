from typing import Literal,Callable
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
    if baseline != "mean":
        raise NotImplementedError("only baseline='mean' is supported")
    if advantage_normalizer != "std":
        raise NotImplementedError("only advantage_normalizer='std' is supported")
    if importance_reweighting_method != "none":
        raise NotImplementedError("only on-policy GRPO is supported")
    if old_log_probs is not None or cliprange is not None:
        raise NotImplementedError("old_log_probs/cliprange are unsupported on-policy")
    if loss_normalization != "sequence":
        raise NotImplementedError("only sequence loss normalization is supported")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if len(repeated_prompts) != len(rollout_responses) or len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("prompts, responses, and ground truths must have equal lengths")
    if len(rollout_responses) % gradient_accumulation_steps != 0:
        raise ValueError("rollout batch must divide evenly across microbatches")
    
    device = next(model.parameters()).device
    
    tokenized = tokenize_prompt_and_output(
        repeated_prompts, 
        rollout_responses, 
        tokenizer
    )

    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn, 
        rollout_responses, 
        repeated_ground_truths
    )

    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards, 
        group_size, 
        baseline=baseline, 
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer
    )

    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device)
    advantages = advantages.to(device)
    metadata: dict[str, torch.Tensor | float] = {
        **reward_metadata,
        **advantage_metadata,
    }

    batch_size = input_ids.shape[0]
    microbatch_size = batch_size // gradient_accumulation_steps
    total_loss = torch.zeros((), device=device)
    for i in range(0, batch_size, microbatch_size):
        end = i+microbatch_size
        inputs_microbatch = input_ids[i:end]
        labels_microbatch = labels[i:end]
        advantages_microbatch = advantages[i:end]

        log_probs = get_response_log_probs(
            model,
            inputs_microbatch,
            labels_microbatch
        )["log_probs"]

        loss_per_token, loss_metadata = compute_policy_gradient_loss(
            advantages_microbatch,
            log_probs,
            importance_reweighting_method
        )

        micro_loss = aggregate_loss_across_microbatch(
            loss_per_token,
            response_mask[i:end],
            loss_normalization,
            normalization_constant
        ) / gradient_accumulation_steps

        micro_loss.backward()
        total_loss += micro_loss.detach()
        metadata.update(loss_metadata)

    grads_norm = None
    if max_grad_norm is not None:
        grads_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    if grads_norm is not None:
        metadata["grad_norm"] = grads_norm.detach()
    return total_loss,metadata
        




