import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel

model_id_or_dir = "/home/joe_sun/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B"
device = torch.device("cuda")

def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device=='cpu' else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer

def compute_entropy(logits):
    probs = F.softmax(logits,dim=-1)
    return -torch.sum(probs*torch.log(probs),dim=-1)

def get_response_log_probs(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False
)-> dict[str,torch.Tensor]:
    outputs = model(input_ids,return_dict = True)
    logits = outputs.logits
    probs = F.log_softmax(logits,dim = -1)
    labels = labels.unsqueeze(-1)

    log_probs = torch.gather(probs,dim= -1,index= labels).squeeze(-1)

    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        return {
            "log_probs": log_probs,
            "token_entropy": token_entropy
        }
    else:
        return {
            "log_probs": log_probs
        }   




