import re
import json
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--data_path',default='data/longtext')
parser.add_argument('--task',choices=['hotpotqa','2wikimqa','musique'],default='musique')
args=parser.parse_args()
INSTRUCTION="Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"

data=[]
with open(f'{args.data_path}/{args.task}.jsonl','r') as f:
    for line in f:
        data.append(json.loads(line))

pattern = r'(Passage \d{1,2}:)\n'
save_lst=[]
for sample in data:
    matches = re.findall(pattern, sample['context'])
    psg_num=len(matches)
    modified_context = re.sub(pattern, r'\1 ', sample['context'])
    input_sen=INSTRUCTION.format(context=modified_context,input=sample['input'])
    save_dict={
        '_id': sample['_id'],
        'psg_num': psg_num,
        'question': sample['input'],
        'answer': sample['answers'][0],
        'sentence': input_sen,
    }
    save_lst.append(save_dict)
print(len(save_lst))
with open(f'data/preprocess/lb_{args.task}.json','w') as f:
    json.dump(save_lst,f,indent=2)