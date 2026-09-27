import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel

model_id_or_dir = "/home/joe_sun/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B"
device = torch.device("cuda")

def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    """加载训练策略和 tokenizer；训练设备由调用方传入。"""
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device=='cpu' else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer

def compute_entropy(logits):
    """logits: [B, T, V]；返回按词表维 V 求和后的 entropy: [B, T]。"""
    probs = F.softmax(logits,dim=-1)  # [B, T, V]，沿词表维归一化。
    return -torch.sum(probs*torch.log(probs),dim=-1)  # [B, T]，每个位置一个 entropy。

def get_response_log_probs(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False
)-> dict[str,torch.Tensor]:
    """input_ids、labels: [B, T]；返回 log_probs（及可选 entropy）: [B, T]。"""
    # logits 形状为 [B, T, V]；每个位置只取 labels 指定 token 的 log-probability。
    outputs = model(input_ids,return_dict = True)
    logits = outputs.logits
    probs = F.log_softmax(logits,dim = -1)  # [B, T, V]，每个位置对词表取 log-softmax。
    labels = labels.unsqueeze(-1)  # [B, T, 1]，gather 所需的索引维。

    # gather 后得到 [B, T]，与 response_mask 对齐。
    log_probs = torch.gather(probs,dim= -1,index= labels).squeeze(-1)  # [B, T]。

    if return_token_entropy:
        # entropy 使用未归一化 logits 计算，便于观察策略的探索程度。
        token_entropy = compute_entropy(logits)  # [B, T]，逐 token 熵。
        return {
            "log_probs": log_probs,
            "token_entropy": token_entropy
        }
    else:
        return {
            "log_probs": log_probs
        }   


