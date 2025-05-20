from transformers import PreTrainedModel
from icl.util_classes.predictor_classes import Predictor
from icl.analysis.adapter_for_train import AttentionAdapter
from icl.model_attentions.custom_llama_attention import CustomLlamaAttention
from icl.model_attentions.custom_qwen_attention import CustomQwen2Attention
from icl.model_attentions.custom_mistral_attention import CustomMistralAttention

AttentionType={
    'llama-3.1-8b-instruct': CustomLlamaAttention,
    'llama-3.2-3b-instruct': CustomLlamaAttention,
    'qwen2.5-7b-instruct': CustomQwen2Attention, 
    'qwen2.5-14b-instruct': CustomQwen2Attention, 
    'qwen2.5-32b-instruct': CustomQwen2Attention, 
    'mistral-7b-instruct':CustomMistralAttention,
}

class AttentionerManagerBase:
    def __init__(self, model: PreTrainedModel, predictor: Predictor,model_name, psg_num, device, n_head, num_layer, top_k=10):
        self.psg_num = psg_num
        self.model_name=model_name
        self.n_head = n_head
        self.device = device
        self.num_layer=num_layer
        self.model = model
        self.attention_adapters = self.register_attentioner_to_model()
        self.predictor = predictor
        self.top_k=top_k

    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, input_ids):
        self._input_ids = input_ids
        psg_pos, que_pos, ans_pos = self.predictor.get_pos(input_ids.cpu())
        for attention_adapter in self.attention_adapters:
            if attention_adapter is not None:
                attention_adapter.register_input_ids(input_ids)
                attention_adapter.psg_pos = psg_pos
                attention_adapter.que_pos = que_pos
                attention_adapter.ans_pos = ans_pos
                attention_adapter.top_k=self.top_k

    def register_input_ids(self, input_ids):
        self.input_ids = input_ids

    def register_attentioner_to_model(self):
        raise NotImplementedError




class AttentionerManager(AttentionerManagerBase):
    def __init__(self, model: PreTrainedModel, model_name, psg_num, predictor: Predictor, device, n_head, num_layer):
        super().__init__(model, predictor, model_name, psg_num, device, n_head, num_layer)
        
    def register_attentioner_to_model(self):
        attention_adapters = []
        for i, layer in enumerate(self.model.model.layers):
            if i<self.num_layer//2:
                attention_adapters.append(None)
                continue
            attention_adapter = AttentionAdapter(psg_num=self.psg_num, n_head=self.n_head, \
                                                 in_features=layer.self_attn.v_proj.in_features,\
                                                 out_features=layer.self_attn.v_proj.out_features, \
                                                 device=self.device)
            original_attn = layer.self_attn
            custom_attn = AttentionType[self.model_name](original_attn.config, attention_adapter, original_attn.layer_idx)
            custom_attn.load_state_dict(original_attn.state_dict(), strict=False)
            layer.self_attn = custom_attn
            attention_adapters.append(attention_adapter)
        return attention_adapters
    
    def register_support_idxs(self, support_idxs):
        for attention_adapter in self.attention_adapters:
            if attention_adapter is not None:
                attention_adapter.support_idxs = support_idxs
