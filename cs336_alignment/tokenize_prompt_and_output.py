import torch
def tokenize_prompt_and_output(
        prompt_strs: list[str],
        output_strs: list[str],
        tokenizer,
) -> dict[str,torch.Tensor]:
    """拼接 prompt 和 response，并生成 teacher-forcing 所需的张量。

    记 B 为 batch 中回答数，L 为 padding 后完整序列长度，则返回的
    ``input_ids``、``labels`` 和 ``response_mask`` 形状都是 ``[B, L-1]``。
    mask 只让 response token 参与 policy-gradient loss。
    """
    # 先分别编码，记录 prompt 边界，后面才能屏蔽 prompt token 的 loss。
    full_prompt_and_output_ids = []  # Python 序列列表；第 i 条长度为 L_i。
    prompt_lens_list = []  # [B]，每条 prompt 的 token 数（Python 整数列表）。
    output_lens_list = []  # [B]，每条 prompt + response 的总 token 数。
    padding_len = 0 
    batch_size = len(prompt_strs)
    for i in range(batch_size):
        prompt_ids = tokenizer.encode(prompt_strs[i])  # 第 i 条 prompt: [P_i]。
        output_ids = tokenizer.encode(output_strs[i])  # 第 i 条 response: [R_i]。
        full_prompt_and_output_ids.append(prompt_ids + output_ids)  # 第 i 条完整序列: [L_i]。

        prompt_ids_len = len(prompt_ids)
        output_ids_len = len(output_ids)

        prompt_lens_list.append(prompt_ids_len)
        output_lens_list.append(output_ids_len + prompt_ids_len)

        padding_len = max(padding_len,len(full_prompt_and_output_ids[i]))

    # 把不同长度的拼接序列右侧 padding 到同一长度。
    pad_id = tokenizer.pad_token_id
    full = torch.full((batch_size,padding_len), pad_id, dtype=torch.long)  # [B, L]。
    for i,ids in enumerate(full_prompt_and_output_ids):
        full[i,:len(ids)] = torch.tensor(ids,dtype=torch.long)  # ids 张量形状为 [L_i]。
    full_prompt_and_output_ids = full  # [B, L]，每行右侧 padding。

    # 原始 mask 对 prompt、response、padding 分别标记为 False、True、False。
    mask = torch.zeros((batch_size,padding_len), dtype= torch.bool)  # [B, L]。
    for i in range(batch_size):
        mask[i,:prompt_lens_list[i]] = False
        mask[i,prompt_lens_list[i]:output_lens_list[i]] = True
        mask[i,output_lens_list[i]:] = False

    # 切片后 input_ids、labels、response_mask 都是 [B, L-1]；
    # labels 比 input_ids 左移一位，mask 与 labels 的目标 token 位置对齐。
    return {
             "input_ids": full_prompt_and_output_ids[:,:-1],
             "labels": full_prompt_and_output_ids[:,1:],
             "response_mask": mask[:,1:]
        }





 
    
