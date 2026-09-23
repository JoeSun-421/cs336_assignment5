import torch
from typing import Literal


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor, 
    mask: torch.Tensor, 
    loss_normalization: Literal["sequence", "constant"] = "sequence", 
    normalization_constant: int | None = None
) -> torch.Tensor:
    # Here mask=True means that the token participates in the loss.
    mask = mask.bool()
    masked_loss = per_token_policy_gradient_loss.masked_fill(~mask, 0.0)
    if loss_normalization == "sequence":
        valid_tokens = mask.sum(dim=-1).clamp_min(1)
        loss = (masked_loss.sum(dim=-1) / valid_tokens).mean()
    elif loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError("normalization_constant cannot be None when loss_normalization is 'constant'")
        loss = torch.sum(masked_loss) / normalization_constant
    else:
        raise ValueError(f"Unknown loss_normalization: {loss_normalization}")

    return loss
