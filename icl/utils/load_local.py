import os.path
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model_layer_num(model = None, model_name = None):
    num_layer = None
    if model is not None:
        if hasattr(model.config, 'num_hidden_layers'):
            num_layer = model.config.num_hidden_layers
        elif hasattr(model.config, 'n_layers'):
            num_layer = model.config.n_layers
        elif hasattr(model.config, 'n_layer'):
            num_layer = model.config.n_layer
        else:
            pass
    elif model_name is not None:
        pass
    if num_layer is None:
        raise ValueError(f'cannot get num_layer from model: {model} or model_name: {model_name}')
    return num_layer
