from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from icl.utils.other import dict_to
from transformers import LogitsProcessorList
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.generation import GenerationMixin
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import os
from icl.analysis.manager_for_train import AttentionerManagerBase

class LMForwardAPI(nn.Module):
    def __init__(self, model_name, tokenizer, label_dict: Dict[int, str], device='cuda'):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.pad_token_id = tokenizer.eos_token_id
        self.model.eos_token_id = tokenizer.eos_token_id
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device

    def generate(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            max_length: int,
            output_attentions: Optional[bool] = True, 
            do_sample: Optional[bool] = False,
            attentionermanger: Optional[AttentionerManagerBase]=None,
        ) -> torch.LongTensor:
        if attentionermanger is not None:
            attentionermanger.register_input_ids(input_ids)
            
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=do_sample,  # greedy decoding
            output_attentions=output_attentions if attentionermanger is None else False,  
            return_dict_in_generate=True,  
        )
        
        generated_text = self.tokenizer.decode(outputs.sequences[0][input_ids.shape[-1]:], skip_special_tokens=True)
        if 'deepseek-r1' in self.model_name.lower():
            if '</think>' not in generated_text:
                generated_text=''
            else:
                generated_text = generated_text.split('</think>')[-1].strip()

        if outputs.attentions is not None:
            attentions = [attn.detach().cpu() for attn in outputs.attentions[0]]
        else:
            attentions=None
        return generated_text, attentions
    
