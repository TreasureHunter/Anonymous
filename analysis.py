'''
task_name: choices:[hotpotqa,2wikimqa,musique]
model_name: choices:[Llama-3.1-8B-Instruct,Qwen2.5-7B-Instruct]
'''
import pickle
import os
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
from icl.util_classes.arg_classes import AttrArgs
from icl.util_classes.predictor_classes import Predictor
from transformers import HfArgumentParser
from icl.utils.other import dict_to
from icl.utils.random_utils import set_seed

from evaluation import update_answer
import os

hf_parser = HfArgumentParser((AttrArgs,))
args: AttrArgs = hf_parser.parse_args_into_dataclasses()[0]
dataset = load_huggingface_dataset(args.task_name,preprocess=True)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

def get_proportion(saliency, psg_pos, que_pos, ans_pos):
    psg_idxs=[range(psg_pos[i], psg_pos[i+1]) for i in range(len(psg_pos)-1)]   # psg_num
    que_idxs=[range(que_pos[i], que_pos[i+1]) for i in range(len(que_pos)-1)]   # 1 ele
    ans_idx=ans_pos[0]

    assert len(saliency.shape) == 3 or (len(saliency.shape) == 4 and saliency.shape[0] == 1)
    if len(saliency.shape) == 4:
        saliency = saliency.squeeze(0)
    saliency=saliency.sum(0)    #[seq,seq]
    saliency = saliency.float().numpy()
    np.fill_diagonal(saliency, 0)   

    proportion1={}
    proportion2={}
    top_k=10
    for i in range(len(psg_idxs)):
        # psg,ans
        proportion1[i]=np.sort(saliency[ans_idx,psg_idxs[i]])[-top_k:].sum()  #psg.i->ans
    for i in range(len(psg_idxs)):
        # psg,que
        proportion2[i] = np.sort(np.sum(saliency[que_idxs[0],][:,psg_idxs[i]], axis=0))[-top_k:].sum()
    proportion3 = np.sort(saliency[ans_idx,que_idxs[0]])[-top_k:].sum()
    proportion4 = saliency.sum() - sum(proportion1.values()) - sum(proportion2.values())-proportion3

    N = int(ans_idx)
    sum1=top_k
    sum2=len(que_idxs[0])*sum1
    sum4 = (N + 1) * N / 2 - args.psg_num*sum1 - args.psg_num*sum2 - top_k
    proportion1 = list(proportion1[k] / sum1 for k in proportion1.keys())
    proportion2 = list(proportion2[k] / sum2 for k in proportion2.keys())
    proportion3 = proportion3 / sum1
    proportion4 = proportion4 / sum4

    proportion1.extend(proportion2)
    proportion1.append(proportion3)
    proportion1.append(proportion4)
    proportions = np.array(proportion1) 
    return proportions

def custom_collate(batch):
    return {
        'input_ids': torch.stack([torch.tensor(x['input_ids']) for x in batch]),
        'attention_mask': torch.stack([torch.tensor(x['attention_mask']) for x in batch]),
        'question': [x['question'] for x in batch],
        'answer': [x['answer'] for x in batch],
        '_id': [x['_id'] for x in batch] if args.task_name in ['2wikimqa','hotpotqa'] else [x['id'] for x in batch],
        'support_idxs': [x['support_idxs'] for x in batch],
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
wraped_trainset = dataset['train'].shuffle(args.seeds[0]).select(range(3*args.sample_size))
wraped_trainset=tokenize_dataset(wraped_trainset, tokenizer=tokenizer, model_name=args.model_name)
threshold_length=2500 if args.task_name in ['musique'] else 1700
wraped_trainset = [data for data in wraped_trainset if len(data['input_ids']) <= threshold_length][:args.sample_size]

valid_indices = [i for i in range(len(wraped_trainset)) 
                if predictor.get_pos(wraped_trainset[i]['input_ids'])[0] is not None]
wraped_trainset = Subset(wraped_trainset, valid_indices) 
print(f'Training sample size: {len(wraped_trainset)}')

training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                  per_device_eval_batch_size=1,
                                  per_device_train_batch_size=1)
trainer = Trainer(model=model, args=training_args, train_dataset=wraped_trainset, data_collator=custom_collate)
train_dataloader = trainer.get_train_dataloader()
model.model = model.model.bfloat16()

os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
pros_list = []
with open(f'{args.save_file_name}.jsonl', 'w') as f:
    for _, data in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
        data = dict_to(data, model.device)
        psg_pos, que_pos, ans_pos = predictor.get_pos(data['input_ids'].cpu())
        if psg_pos is None or len(psg_pos)!=args.psg_num+1 or len(que_pos)!=2 or len(ans_pos)!=1:
            torch.cuda.empty_cache()  
            continue
        output_text, attention_matrix = model.generate(data['input_ids'], data['attention_mask'], max_length=data['input_ids'].shape[-1]+args.max_new_tokens)
        print(output_text)
        attention_matrix = [attn.detach().cpu() for attn in attention_matrix]
        pros = []
        for i in range(len(model.model.model.layers)):
            pro=get_proportion(attention_matrix[i], psg_pos, que_pos, ans_pos)
            pros.append(pro)
        pros = np.array(pros)   
        pros = pros.T
        pros_list.append(pros)
        
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
            'support_idxs':data['support_idxs'][0],
            'leng':data['input_ids'].shape[-1],
            'label':saveLabel
        }
        f.write(json.dumps(sample_dict, ensure_ascii=False)+'\n')

        torch.cuda.empty_cache()  
        

pros_list = np.array(pros_list) 

with open(args.save_file_name, 'wb') as f:
    pickle.dump(pros_list, f)

