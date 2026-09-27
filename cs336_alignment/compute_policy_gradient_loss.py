from typing import Literal

import torch


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the per-token policy-gradient surrogate loss.

    ``raw_rewards_or_advantages`` has one scalar per response, while
    ``policy_log_probs`` has shape ``[B, T]``. GSPO additionally needs
    ``response_mask`` because it first computes one ratio per response.
    """
    if importance_reweighting_method not in {"none", "noclip", "grpo", "gspo"}:
        raise ValueError(
            f"Unknown importance_reweighting_method: {importance_reweighting_method}"
        )
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape [batch_size, sequence_length]")

    batch_size, sequence_length = policy_log_probs.shape
    advantages = raw_rewards_or_advantages.reshape(-1, 1)
    if advantages.shape[0] != batch_size:
        raise ValueError("There must be one reward/advantage per response")

    if importance_reweighting_method == "none":
        # The outer aggregation applies response_mask.
        return -advantages * policy_log_probs, {}

    if old_log_probs is None:
        raise ValueError(
            f"old_log_probs must be provided for {importance_reweighting_method}"
        )
    if old_log_probs.shape != policy_log_probs.shape:
        raise ValueError("old_log_probs and policy_log_probs must have the same shape")
    if importance_reweighting_method in {"grpo", "gspo"} and cliprange is None:
        raise ValueError(
            f"cliprange must be provided for {importance_reweighting_method}"
        )

    # old_log_probs is fixed; only current policy_log_probs remains in graph.
    log_ratio = policy_log_probs - old_log_probs.detach()

    if importance_reweighting_method == "noclip":
        ratio = torch.exp(log_ratio)
        return -advantages * ratio, {}

    if importance_reweighting_method == "grpo":
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio,
            min=1.0 - cliprange,
            max=1.0 + cliprange,
        )
        surrogate = torch.minimum(
            advantages * ratio,
            advantages * clipped_ratio,
        )
        clipped_positions = (ratio < 1.0 - cliprange) | (
            ratio > 1.0 + cliprange
        )
        if response_mask is not None:
            valid = response_mask.bool()
            if valid.shape != policy_log_probs.shape:
                raise ValueError("response_mask and policy_log_probs must have the same shape")
            denominator = valid.sum().clamp_min(1)
            clip_fraction = (clipped_positions & valid).sum() / denominator
        else:
            clip_fraction = clipped_positions.float().mean()
        return -surrogate, {"clip-fraction": clip_fraction}

    if response_mask is None:
        raise ValueError("response_mask must be provided for gspo")
    response_mask = response_mask.bool()
    if response_mask.shape != policy_log_probs.shape:
        raise ValueError("response_mask and policy_log_probs must have the same shape")

    # GSPO uses the geometric-mean ratio over valid response tokens.
    masked_log_ratio = log_ratio.masked_fill(~response_mask, 0.0)
    response_lengths = response_mask.sum(dim=-1).clamp_min(1).to(log_ratio.dtype)
    sequence_log_ratio = masked_log_ratio.sum(dim=-1) / response_lengths
    sequence_ratio = sequence_log_ratio.exp().unsqueeze(-1)
    clipped_sequence_ratio = torch.clamp(
        sequence_ratio,
        min=1.0 - cliprange,
        max=1.0 + cliprange,
    )
    sequence_surrogate = torch.minimum(
        advantages * sequence_ratio,
        advantages * clipped_sequence_ratio,
    )
    # Broadcast the sequence loss to satisfy the common [B, T] interface.
    per_token_loss = -sequence_surrogate.expand(batch_size, sequence_length)
    clipped_sequences = (
        (sequence_ratio < 1.0 - cliprange)
        | (sequence_ratio > 1.0 + cliprange)
    )
    return per_token_loss, {"clip-fraction": clipped_sequences.float().mean()}
