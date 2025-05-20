import json
import os
from evaluation import update_answer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='results/data')
parser.add_argument('--model_name', type=str, default='qwen2.5-32b-instruct')
parser.add_argument('--is_baseline', type=bool, default=True)
parser.add_argument('--seed', type=int, default=42)
args=parser.parse_args()
if args.is_baseline:
    dirpath=os.path.join(args.data_path,'baseline',args.model_name)
else:
    dirpath=os.path.join(args.data_path,'DSAS',args.model_name)
fnames=[f for f in os.listdir(dirpath) if f.endswith('jsonl')]

save_res={}
for fname in fnames:
    fpath=os.path.join(dirpath,fname)
    temp_res=[]
    with open(fpath, 'r') as f:
        for line in f:
            temp_res.append(json.loads(line))
    metrics={'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    for i in range(len(temp_res)):
        ans=temp_res[i]['answer']
        resp=temp_res[i]['responses']
        update_answer(metrics, resp, ans)
    for k, v in metrics.items():
        metrics[k]=v/len(temp_res)

    save_res[fname.split('_')[0]]=metrics
print(save_res)
save_path=os.path.join(dirpath,'result.json')
with open(save_path, 'w') as f:
    json.dump(save_res, f, indent=4)