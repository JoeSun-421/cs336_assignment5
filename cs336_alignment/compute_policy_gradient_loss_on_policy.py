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
    B = raw_rewards_or_advantages.shape[0]
    advantages = raw_rewards_or_advantages.reshape(-1,1)
    if importance_reweighting_method == "none":
        per_token_loss = -advantages * policy_log_probs
    elif importance_reweighting_method == "noclip":
        assert old_log_probs is not None, "old_log_probs must be provided for importance reweighting"
        raise ValueError("noclip importance reweighting is not implemented yet")
    elif importance_reweighting_method == "grpo":
        assert old_log_probs is not None, "old_log_probs must be provided for importance reweighting"
        assert cliprange is not None, "cliprange must be provided for grpo"
        raise ValueError("grpo importance reweighting is not implemented yet")
    elif importance_reweighting_method == "gspo":
        assert old_log_probs is not None, "old_log_probs must be provided for importance reweighting"
        assert cliprange is not None, "cliprange must be provided for gspo"
        raise ValueError("gspo importance reweighting is not implemented yet")  
    else:
        raise ValueError(f"Unknown importance_reweighting_method: {importance_reweighting_method}") 
    clip_fraction = torch.tensor(0.0)
    return per_token_loss, {"clip-fraction": clip_fraction}