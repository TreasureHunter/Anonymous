import warnings
import torch


class Predictor:
    def __init__(self, label_id_dict, pad_token_id, 
                 task_name, tokenizer, layer, psg_num, sample_id_pnum_map=None) -> None:
        self.label_id_dict = label_id_dict
        self.pad_token_id = pad_token_id
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.layer = layer
        self.max_psg_num=psg_num
        self.psg_num=psg_num
        self.prefix_idxs=label_id_dict
        
        self.sample_id_pnum_map=sample_id_pnum_map


    def get_pos(self, inputs, id=None):
        if isinstance(inputs,list):
            inputs=torch.tensor(inputs).view(1,-1)
        label_id_dict = self.label_id_dict
        final_pos = [inputs.shape[-1]-1]
        if self.sample_id_pnum_map is not None and id is not None:
            self.psg_num=self.sample_id_pnum_map[id]

        def get_match_indices(input_ids, tokens):
            tokens = torch.tensor(tokens, device=input_ids.device)
            match_len = len(tokens)
            match_indices = []
            for sample in input_ids:
                windows = sample.unfold(0, size=match_len, step=1)
                match_ind = (windows == tokens).all(dim=1).nonzero(as_tuple=True)[0].tolist()
                match_indices.append(match_ind)
            return match_indices
            
        que_matches=get_match_indices(inputs,label_id_dict[0])[-1]
        psg_matches=[]
        for i in range(1,1+self.psg_num):
            temp_matches=get_match_indices(inputs,label_id_dict[i])
            if temp_matches==[[]] or len(temp_matches[0])!=1:
                return None,None,None
            psg_matches.append(temp_matches[0][-1])
        temp_matches=get_match_indices(inputs,label_id_dict[1+self.max_psg_num])
        psg_matches.append(temp_matches[0][-1])
        que_end=get_match_indices(inputs,label_id_dict[2+self.max_psg_num])[0][-1]
        que_matches.append(que_end)
        if len(psg_matches)!=self.psg_num+1:
            return None,None,None
            
        return psg_matches, que_matches, final_pos
