import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import scipy.stats as stats

class AttentionAdapterBase:
    def __init__(self, *args, **kwargs):
        # super().__init__()
        self.use_flag = True
        self.support_idxs=None

    def register_input_ids(self, input_ids: torch.Tensor):
        self.input_ids = input_ids

class AttentionAdapter(AttentionAdapterBase):
    def __init__(self, psg_num, n_head, in_features, out_features, device) -> None:
        super().__init__()
        self.psg_num = psg_num
        self.n_head = n_head
        self.device=device
        self.in_features=in_features
        self.out_features=out_features

        self.psg_pos=None   #lst(psg_num+1)
        self.que_pos=None   #lst(2)
        self.ans_pos=None   #lst(1)

        self.top_k=10
    
    def DSASAttn(self, attn_weights):
        if attn_weights.shape[-2]==1:
            return attn_weights
        seqlen=attn_weights.shape[-2]-1
        psg_pos = self.psg_pos
        que_pos = self.que_pos
        psg_idxs=[range(psg_pos[i], psg_pos[i+1]) for i in range(len(psg_pos)-1)]   
        que_idxs=[range(que_pos[i], que_pos[i+1]) for i in range(len(que_pos)-1)]   
        ans_idx=-1
        temp_weights=attn_weights.squeeze(0) if attn_weights.shape[0]==1 else attn_weights
        temp_weights=temp_weights.sum(0)    #seq,seq
        psg_que_ans=torch.zeros(len(psg_idxs)).to(self.device)
        for i in range(len(psg_idxs)):
            merge_matrix=torch.cat([temp_weights[que_idxs[0]][:,psg_idxs[i]], temp_weights[ans_idx,psg_idxs[i]].expand(len(que_idxs[0]),-1)], dim=0) 
            select_values,_=torch.topk(merge_matrix.sum(0), self.top_k)
            psg_que_ans[i]=select_values.sum(0)
        
        # Stage 1
        # norm+sigmoid
        avg_val = torch.mean(psg_que_ans)
        std_val = torch.std(psg_que_ans)
        norm_val = (psg_que_ans - avg_val) / (std_val + 1e-8) 
        norm_val = torch.sigmoid(norm_val)*0.5+0.5
        # compute gama
        _, indices = torch.sort(norm_val, descending=True)
        gama = torch.ones(len(psg_idxs), device=self.device)
        all_tokens = torch.arange(seqlen, device=self.device, dtype=torch.float32)  
        for i in range(len(psg_idxs) // 2):
            idx = indices[i]
            tokens = torch.tensor(psg_idxs[idx], device=self.device)
            mean = torch.mean(all_tokens)
            sigma = torch.std(all_tokens)
            max_tokens = torch.max(tokens).float()
            min_tokens = torch.min(tokens).float()
            # CDF
            z_max = (max_tokens - mean) / (sigma * torch.sqrt(torch.tensor(2.0)))
            z_min = (min_tokens - mean) / (sigma * torch.sqrt(torch.tensor(2.0)))
            cdf_diff = 0.5 * (torch.erf(z_max) - torch.erf(z_min))
            delta = max_tokens - min_tokens
            gama[i] = sigma * cdf_diff / delta
        # compute gi
        gi = torch.ones_like(gama, device=self.device)
        for i in range(len(psg_idxs) // 2):
            idx = indices[i]
            gi[idx] = ((len(psg_idxs)//2 + 1) / (i + 1)) ** gama[idx]
        # compute wi+norm
        alpha = 1
        beta = 0.7
        wi = norm_val * torch.pow(gi, alpha)
        min_wi = torch.min(wi)
        max_wi = torch.max(wi)
        wi_norm = (wi - min_wi) / (max_wi - min_wi + 1e-8)
        wi_norm = (1 - beta) * wi_norm + beta
        #update attn_weights
        for i in range(len(psg_idxs)):
            if wi_norm[i]<0.5*(beta+1):
                attn_weights[:,:,ans_idx,psg_idxs[i]] *= wi_norm[i]
                que_mask = torch.zeros_like(attn_weights, dtype=torch.bool).to(self.device)
                que_mask[:,:,que_idxs[0]][:,:,:,psg_idxs[i]] = True
                attn_weights[que_mask] *= wi_norm[i]


        # Stage 2
        wi_mean = wi_norm.mean()
        p_key = torch.where(wi_norm > wi_mean)[0].tolist()
        p_irr = torch.where(wi_norm <= wi_mean)[0].tolist()
        p_key = sorted(p_key)
        p_irr = sorted(p_irr)
        for i in p_key:
            irr_j = [j for j in p_irr if j < i]
            # modify key->irr
            for j in irr_j:
                mask=torch.zeros_like(attn_weights, dtype=torch.bool).to(self.device)
                mask[:,:,psg_idxs[i]][:,:,:,psg_idxs[j]] = True
                attn_weights[mask] *= wi_norm[j]
        for i in p_irr:
            key_j = [j for j in p_key if j < i]
            # modify irr->key
            for j in key_j:
                mask=torch.zeros_like(attn_weights, dtype=torch.bool).to(self.device)
                mask[:,:,psg_idxs[i]][:,:,:,psg_idxs[j]] = True
                attn_weights[mask] *= wi_norm[i]

        return attn_weights
    