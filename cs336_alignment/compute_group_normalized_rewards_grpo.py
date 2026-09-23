from typing import Literal

import torch


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    if baseline != "mean":
        raise NotImplementedError(f"Unsupported baseline: {baseline}")
    if advantage_normalizer != "std":
        raise NotImplementedError(
            f"Unsupported advantage normalizer: {advantage_normalizer}"
        )
    if raw_rewards.ndim != 1:
        raise ValueError("raw_rewards must be a 1D tensor")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if raw_rewards.numel() % group_size != 0:
        raise ValueError("raw_rewards length must be divisible by group_size")

    groups = raw_rewards.reshape(-1, group_size)
    group_means = groups.mean(dim=1, keepdim=True)
    centered_rewards = groups - group_means
    group_stds = groups.std(dim=1, keepdim=True, unbiased=True)
    advantages = (centered_rewards / (group_stds + advantage_eps)).reshape_as(
        raw_rewards
    )

    metadata = {
        "mean_reward": float(raw_rewards.mean().item()),
        "mean_advantage": float(advantages.mean().item()),
    }
    return advantages, metadata
