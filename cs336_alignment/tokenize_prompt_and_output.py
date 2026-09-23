import torch
def tokenize_prompt_and_output(
        prompt_strs: list[str],
        output_strs: list[str],
        tokenizer,
) -> dict[str,torch.Tensor]:
    
    full_prompt_and_output_ids = []
    prompt_lens_list = []
    output_lens_list = []
    padding_len = 0 
    batch_size = len(prompt_strs)
    for i in range(batch_size):
        prompt_ids = tokenizer.encode(prompt_strs[i])
        output_ids = tokenizer.encode(output_strs[i])
        full_prompt_and_output_ids.append(prompt_ids + output_ids)

        prompt_ids_len = len(prompt_ids)
        output_ids_len = len(output_ids)

        prompt_lens_list.append(prompt_ids_len)
        output_lens_list.append(output_ids_len + prompt_ids_len)

        padding_len = max(padding_len,len(full_prompt_and_output_ids[i]))

    pad_id = tokenizer.pad_token_id
    full = torch.full((batch_size,padding_len), pad_id, dtype=torch.long)
    for i,ids in enumerate(full_prompt_and_output_ids):
        full[i,:len(ids)] = torch.tensor(ids,dtype=torch.long)
    full_prompt_and_output_ids = full

    mask = torch.zeros((batch_size,padding_len), dtype= torch.bool)
    for i in range(batch_size):
        mask[i,:prompt_lens_list[i]] = False
        mask[i,prompt_lens_list[i]:output_lens_list[i]] = True
        mask[i,output_lens_list[i]:] = False

    return {
             "input_ids": full_prompt_and_output_ids[:,:-1],
             "labels": full_prompt_and_output_ids[:,1:],
             "response_mask": mask[:,1:]
        }





 
    

