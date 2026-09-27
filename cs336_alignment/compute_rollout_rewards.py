from typing import Callable

import torch


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """调用 verifier，为每条 rollout 生成 reward，并返回可记录的统计量。"""
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            "rollout_responses and repeated_ground_truths must have the same length"
        )

    # 每个 response 和对应 ground truth 必须保持一一对应；两份 Python 列表长度均为 B。
    scores = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(
            rollout_responses, repeated_ground_truths, strict=True
        )
    ]

    # 训练使用的主 reward 是一维张量 [batch_size]；其他分数只用于日志。
    raw_rewards = torch.tensor(  # [B]，每条 rollout 一个标量 reward。
        [score["reward"] for score in scores], dtype=torch.float32
    )
    # 保留原始 rollout batch 的统计量，即使后续会删除零 advantage 样本。
    # metadata 中是 batch 级 Python 标量，不是向量张量。
    metadata = {
        "mean_reward": float(raw_rewards.mean().item()),
        "mean_format_reward": float(
            sum(score["format_reward"] for score in scores) / len(scores)
        ),
    }
    return raw_rewards, metadata
