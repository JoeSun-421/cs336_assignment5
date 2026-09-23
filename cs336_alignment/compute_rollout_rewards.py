from typing import Callable

import torch


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            "rollout_responses and repeated_ground_truths must have the same length"
        )

    scores = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(
            rollout_responses, repeated_ground_truths, strict=True
        )
    ]

    raw_rewards = torch.tensor(
        [score["reward"] for score in scores], dtype=torch.float32
    )
    metadata = {
        "mean_reward": float(raw_rewards.mean().item()),
        "mean_format_reward": float(
            sum(score["format_reward"] for score in scores) / len(scores)
        ),
    }
    return raw_rewards, metadata
