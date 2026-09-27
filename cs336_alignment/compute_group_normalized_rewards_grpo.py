from typing import Literal

import torch


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """按每个 prompt 的回答组计算 response-level advantages。

    记 P 为 prompt 数、G 为 ``group_size``、B=P*G 为 rollout 数。
    ``raw_rewards`` 和返回的 ``advantages`` 形状都是 ``[B]``；reshape 后每一行
    对应一个 prompt 的 G 个回答。baseline 和 normalizer 的组合分别
    实现 GRPO、Dr. GRPO、RFT 和 MaxRL 的 reward 权重。
    """

    if baseline not in {"mean", "none"}:
        raise NotImplementedError(f"Unsupported baseline: {baseline}")
    if advantage_normalizer not in {"std", "none","mean"}:
        raise NotImplementedError(
            f"Unsupported advantage normalizer: {advantage_normalizer}"
        )

    if raw_rewards.ndim != 1:
        raise ValueError("raw_rewards must be a 1D tensor")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if raw_rewards.numel() % group_size != 0:
        raise ValueError("raw_rewards length must be divisible by group_size")

    # 每一行是同一个 prompt 的 group，后面的均值和标准差只在行内计算。
    groups = raw_rewards.reshape(-1, group_size)  # [P, G]，每行对应同一个 prompt。

    if baseline == "mean":
        # 组均值是 response-level baseline；它会让同组相对更好的回答获得正权重。
        group_means = groups.mean(dim=1, keepdim=True)  # [P, 1]。
        baseline_adjusted_rewards = groups - group_means  # [P, G]。
    else:
        baseline_adjusted_rewards = groups  # [P, G]。

    if advantage_normalizer == "std":
        # Use the sample standard deviation required by the assignment.
        group_stds = groups.std(dim=1, keepdim=True, unbiased=True)  # [P, 1]。
        advantages = baseline_adjusted_rewards / (group_stds + advantage_eps)  # [P, G]。
    elif advantage_normalizer == "none":
        advantages = baseline_adjusted_rewards  # [P, G]。
    elif advantage_normalizer == "mean":
        # MaxRL 使用组均值作分母，强调相对于当前平均 reward 的改进幅度。
        group_means = groups.mean(dim=-1, keepdim=True)  # [P, 1]。
        advantages = baseline_adjusted_rewards / (group_means + advantage_eps)  # [P, G]。

    # 恢复与 raw_rewards 相同的一维布局，便于和 rollout 列表逐条对齐。
    advantages = advantages.reshape_as(raw_rewards)  # [B]，恢复 rollout 顺序。

    metadata = {
        "mean_reward": float(raw_rewards.mean().item()),
        "mean_advantage": float(advantages.mean().item()),
    }
    return advantages, metadata
