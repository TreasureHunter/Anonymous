'''
task_name: choices:['hotpotqa','2wikimqa','musique']
model_name: choices:['llama-3.1-8b-instruct','llama-3.2-3b-instruct','qwen2.5-7b-instruct','mistral-7b-instruct','deepseek-r1-8b']
'''
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
import torch
import json
from torch.utils.data import Subset


from icl.lm_apis.lm_api_base import LMForwardAPI
from icl.utils.data_wrapper import tokenize_dataset
from icl.utils.load_huggingface_dataset import load_huggingface_dataset
from icl.utils.prepare_model_and_tokenizer import get_label_id_dict_for_args
from transformers import Trainer, TrainingArguments, PreTrainedModel, AutoModelForCausalLM, \
    AutoTokenizer
from icl.utils.load_local import convert_path_old, get_model_layer_num
from icl.util_classes.arg_classes import DSASArgs
from icl.util_classes.predictor_classes import Predictor
from transformers import HfArgumentParser
from icl.utils.other import dict_to
from icl.utils.random_utils import set_seed

from icl.analysis.manager_for_train import AttentionerManager
from evaluation import update_answer

hf_parser = HfArgumentParser((DSASArgs,))
args: DSASArgs = hf_parser.parse_args_into_dataclasses()[0]
dataset = load_huggingface_dataset(args.task_name,preprocess=True)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

def custom_collate(batch):
    return {
        'input_ids': torch.stack([torch.tensor(x['input_ids']) for x in batch]),
        'attention_mask': torch.stack([torch.tensor(x['attention_mask']) for x in batch]),
        'question': [x['question'] for x in batch],
        'answer': [x['answer'] for x in batch],
        '_id': [x['_id'] for x in batch] if args.task_name in ['2wikimqa','hotpotqa'] else [x['id'] for x in batch],
    }

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token
args.label_id_dict = get_label_id_dict_for_args(args, tokenizer)

model = LMForwardAPI(model_name=args.model_path, tokenizer=tokenizer,
                     device='cuda',
                     label_dict=args.label_dict)
num_layer = get_model_layer_num(model=model.model, model_name=args.model_path)
predictor = Predictor(label_id_dict=args.label_id_dict, pad_token_id=tokenizer.pad_token_id,
                      task_name=args.task_name, tokenizer=tokenizer, layer=num_layer, psg_num=args.psg_num)

set_seed(args.seeds[0])
wraped_devset = dataset['dev'].shuffle(args.seeds[0])
wraped_devset=tokenize_dataset(wraped_devset, tokenizer=tokenizer, model_name=args.model_name)

valid_indices = [i for i in range(len(wraped_devset)) 
                if predictor.get_pos(wraped_devset[i]['input_ids'])[0] is not None]
wraped_devset = Subset(wraped_devset, valid_indices)  
print(f'Sample size: {len(wraped_devset)}')
if os.path.exists(f'{args.save_file_name}.jsonl'):
    exclude_ids = []
    with open(f'{args.save_file_name}.jsonl', 'r') as f:
        for line in f:        
            data = json.loads(line)
            exclude_ids.append(data['_id'])
    valid_indices = []
    original_dataset = wraped_devset.dataset 
    for idx in wraped_devset.indices:
        sample = original_dataset[idx]
        if args.task_name not in ['musique']:
            if sample['_id'] not in exclude_ids:
                valid_indices.append(idx)
        else:
            if sample['id'] not in exclude_ids:
                valid_indices.append(idx)
    if valid_indices==[]:
        print('Finished!')
        exit(0)
    wraped_devset = Subset(original_dataset, valid_indices)

attentionermanger = AttentionerManager(model.model, model_name=args.model_name, psg_num=args.psg_num,
                        predictor=predictor, device=model.device, n_head = args.n_head, num_layer=num_layer)

training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                  per_device_eval_batch_size=1,
                                  per_device_train_batch_size=1)
trainer = Trainer(model=model, args=training_args, train_dataset=wraped_devset, data_collator=custom_collate)
train_dataloader = trainer.get_train_dataloader()
model.model = model.model.bfloat16()

os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
with open(f'{args.save_file_name}.jsonl', 'w') as f:
    for idx, data in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
        data = dict_to(data, model.device)
        output_text, _ = model.generate(data['input_ids'], data['attention_mask'], \
                                                       max_length=data['input_ids'].shape[-1]+args.max_new_tokens, attentionermanger=attentionermanger)
        print(output_text)
        
        metrics={'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
        em, prec, _=update_answer(metrics, output_text, data['answer'][0])
        if em:
            saveLabel='Good'
        elif prec==0:
            saveLabel='Bad'
        else:
            saveLabel='Other'
        sample_dict={
            '_id':data['_id'][0],
            'question':data['question'][0],
            'answer':data['answer'][0],
            'responses':output_text,
            'leng':data['input_ids'].shape[-1],
            'label':saveLabel
        }
        f.write(json.dumps(sample_dict, ensure_ascii=False)+'\n')
        torch.cuda.empty_cache() 
