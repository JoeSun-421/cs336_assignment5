import torch
from typing import Literal


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor, 
    mask: torch.Tensor, 
    loss_normalization: Literal["sequence", "constant"] = "sequence", 
    normalization_constant: int | None = None
) -> torch.Tensor:
    """per_token_policy_gradient_loss 和 mask: [B, T]；返回可反向传播的标量 []。

    ``sequence`` 先对每条回答的有效 token 求平均，再对回答求平均；
    ``constant`` 则把所有有效 token 的 loss 求和后除以固定常数。
    """
    # Here mask=True means that the token participates in the loss.
    mask = mask.bool()
    masked_loss = per_token_policy_gradient_loss.masked_fill(~mask, 0.0)  # [B, T]。
    if loss_normalization == "sequence":
        # 每条回答的长度不同，先用自己的有效 token 数做归一化。
        valid_tokens = mask.sum(dim=-1).clamp_min(1)  # [B]，每条回答有效 token 数。
        loss = (masked_loss.sum(dim=-1) / valid_tokens).mean()  # 先得到 [B]，再平均为标量 []。
    elif loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError("normalization_constant cannot be None when loss_normalization is 'constant'")
        # 该常数通常已经代表整个训练 batch 的固定归一化尺度。
        loss = torch.sum(masked_loss) / normalization_constant  # 标量张量 []。
    else:
        raise ValueError(f"Unknown loss_normalization: {loss_normalization}")

    return loss
