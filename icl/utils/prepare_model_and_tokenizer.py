import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM

from ..util_classes.arg_classes import DeepArgs


def load_model_and_tokenizer(args: DeepArgs):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.pad_token_id = tokenizer.eos_token_id
    model.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer


def get_label_id_dict_for_args(args: DeepArgs, tokenizer):
    if args.model_name in ['mistral-7b-instruct']:
        label_id_dict=args.label_dict
        for k in label_id_dict.keys():
            if k==0:
                label_id_dict[k]=tokenizer.encode(label_id_dict[k], add_special_tokens=False)[2:]
            else:
                label_id_dict[k]=tokenizer.encode('\n'+label_id_dict[k], add_special_tokens=False)[3:]
    else:
        label_id_dict = {k: tokenizer.encode(v, add_special_tokens=False) for k, v in
                          args.label_dict.items()}
    return label_id_dict
