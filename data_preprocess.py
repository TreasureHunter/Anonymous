'''
task_name: choices[hotpotqa,2wikimqa,musique]
'''
from icl.util_classes.arg_classes import DeepArgs
from transformers.hf_argparser import HfArgumentParser
from icl.utils.load_huggingface_dataset import load_huggingface_dataset
import ast
from datasets import Dataset
import json
import argparse
INSTRUCTION="Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
parser=argparse.ArgumentParser()
parser.add_argument('--data_path',default='data')
parser.add_argument('--task',choices=['hotpotqa','2wikimqa','musique'],default='musique')
args=parser.parse_args()
dataset = load_huggingface_dataset(args.task_name)

def prepare_analysis_dataset(task_name,input_dataset,save_path,split):
    input_dataset=[{k:str(v) for k,v in sample.items()} for sample in input_dataset]
    input_dataset = {key: [item[key] for item in input_dataset] for key in input_dataset[0].keys()}
    input_dataset=Dataset.from_dict(input_dataset)

    def wrap_sentence(examples):
        if task_name in ['hotpotqa','2wikimqa']:
            standard_context=''
            lst_context=ast.literal_eval(examples['context'])
            for i,psg in enumerate(lst_context):
                title=psg[0]
                sentences=psg[1]
                psg_str=f'Passage {i+1}: '+title+'\n'+' '.join(sentences)+'\n'
                standard_context+=psg_str
            examples['sentence']=INSTRUCTION.format(context=standard_context,input=examples['question'])
            
            if split=='test':
                return examples
            
            all_titles=[p[0] for p in lst_context]
            supporting_facts=ast.literal_eval(examples['supporting_facts'])
            support_titles=set([lst[0] for lst in supporting_facts])
            examples['support_idxs']=[]
            for title in support_titles:
                try:
                    index = all_titles.index(title)
                except ValueError:
                    index = -1 
                examples['support_idxs'].append(index)
            return examples
        elif task_name in ['musique']:
            standard_context=''
            lst_context=ast.literal_eval(examples['paragraphs'])
            sup_idxs=[]
            for i,psg_dict in enumerate(lst_context):
                title=psg_dict['title']
                text=psg_dict['paragraph_text']
                psg_str=f'Passage {i+1}: '+title+'\n'+text+'\n'
                standard_context+=psg_str
                if split=='train':
                    if psg_dict['is_supporting']:
                        sup_idxs.append(psg_dict['idx'])
            if split=='train':
                assert bool(examples['answerable'])
                examples['support_idxs']=sup_idxs
            examples['sentence']=INSTRUCTION.format(context=standard_context,input=examples['question'])
            return examples
    input_dataset=input_dataset.map(wrap_sentence,batched=False)

    samples = []
    for sample in input_dataset:
        samples.append(sample)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    return input_dataset

train_save_path=f'{args.data_path}/preprocess/{args.task_name}_train.json'
dev_save_path=f'{args.data_path}/preprocess/{args.task_name}_dev.json'
test_save_path=f'{args.data_path}/preprocess/{args.task_name}_test.json'
wraped_trainset = prepare_analysis_dataset(args.task_name,dataset['train'],train_save_path,'train')
wraped_devset = prepare_analysis_dataset(args.task_name,dataset['dev'],dev_save_path,'train')
wraped_testset = prepare_analysis_dataset(args.task_name,dataset['test'],test_save_path,'test')
print(wraped_trainset)
print(wraped_devset)
print(wraped_testset)
