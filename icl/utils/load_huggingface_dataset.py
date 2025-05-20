import os.path

from datasets import load_dataset
ROOT_FOLEDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_from_local(task_name, splits):
    dataset_path = os.path.join(ROOT_FOLEDER, 'datasets', task_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset_path: {dataset_path}")
    dataset = load_dataset(dataset_path)
    dataset = [dataset[split] for split in splits]
    return dataset


def load_huggingface_dataset(task_name,preprocess=False):
    if preprocess:
        if 'long' in task_name:
            temp_name=task_name.split('-')[-1]
            fpath=f'data/preprocess/lb_{temp_name}'
            dataset=load_dataset("json",data_files={'dev':f'{fpath}.json'})
            return dataset
        else:
            fpath=f'data/preprocess/{task_name}'
            dataset_dict = load_dataset(
                "json",
                data_files={
                    "train": f'{fpath}_train.json',
                    "dev": f'{fpath}_dev.json',
                    "test": f'{fpath}_test.json'
                }
            )
            return dataset_dict
    else:
        fpath=f'data/{task_name}'
        import json
        if task_name in ['hotpotqa','2wikimqa']:
            #json format
            with open(f'{fpath}_train.json', 'r', encoding='utf-8') as f:
                data_train=json.load(f)
            with open(f'{fpath}_dev.json', 'r', encoding='utf-8') as f:
                data_dev=json.load(f)
            with open(f'{fpath}_test.json', 'r', encoding='utf-8') as f:
                data_test=json.load(f)
            return {'train':data_train,'dev':data_dev,'test':data_test}
        elif task_name in ['musique']:
            #jsonl format
            data_train=[]
            data_dev=[]
            data_test=[]
            with open(f'{fpath}_train.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    data_train.append(json.loads(line)) 
            with open(f'{fpath}_dev.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    data_dev.append(json.loads(line)) 
            with open(f'{fpath}_test.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    data_test.append(json.loads(line))
            return {'train':data_train,'dev':data_dev,'test':data_test}
