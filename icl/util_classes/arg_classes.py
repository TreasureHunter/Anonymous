import os
import pickle
import warnings
from dataclasses import field, dataclass
from typing import List, Optional
from ..project_constants import FOLDER_ROOT
import json

@dataclass
class DeepArgs:
    task_name: str = "long-hotpotqa"
    model_name: str = "llama-3.1-8b-instruct"
    seeds: List[int] = field(default_factory=lambda: [42])
    sample_size: int = 1000
    batch_size: int = 1
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'data', 'deep', f'{os.path.basename(model_name)}')
    gpu: int=2

    def __post_init__(self):
        self.model_name = self.model_name.lower()
        assert self.model_name in ['llama-3.1-8b-instruct','llama-3.2-3b-instruct',\
                                   'qwen2.5-7b-instruct','mistral-7b-instruct','deepseek-r1-8b','qwen2.5-14b-instruct','llama-2-13b','qwen2.5-32b-instruct']
        assert self.task_name in ['hotpotqa','2wikimqa','musique', 'long-hotpotqa', 'long-2wikimqa', 'long-musique']
        with open('config/model2path.json','r') as f:
            self.model_path = json.load(f)[self.model_name]
        with open('config/dataset2num.json') as f:
            self.psg_num = json.load(f)[self.task_name]

        if self.model_name in ['deepseek-r1-8b']:
            self.max_new_tokens=500
        else:
            self.max_new_tokens=50

        self.actual_sample_size = self.sample_size

        label_dict={0: '.\n\nQuestion'}
        for i in range(1, 1+self.psg_num):
            label_dict[i] = f'Paragraph {i}:'
        label_dict[1+self.psg_num]='Answer the question based'
        label_dict[2+self.psg_num]='Answer:'
        self.label_dict = label_dict

        self.file_name = f"{self.task_name}_{self.seeds[0]}"
        self.save_file_name = os.path.join(self.save_folder, self.file_name)
        
    def load_result(self):
        with open(self.save_file_name, 'rb') as f:
            return pickle.load(f)

@dataclass
class AttrArgs(DeepArgs):
    def __post_init__(self):
        super().__post_init__()
        self.save_folder=os.path.join(FOLDER_ROOT, 'results', 'data', 'attr', f'{os.path.basename(self.model_name)}')
        self.save_file_name = os.path.join(self.save_folder, self.file_name)

@dataclass
class DSASArgs(DeepArgs):
    n_head: int = 32
    def __post_init__(self):
        super().__post_init__()
        self.save_folder=os.path.join(FOLDER_ROOT, 'results', 'data', 'DSAS', f'{os.path.basename(self.model_name)}')
        self.save_file_name = os.path.join(self.save_folder, self.file_name)

class BaselineArgs(DeepArgs):
    def __post_init__(self):
        super().__post_init__()
        self.save_folder=os.path.join(FOLDER_ROOT, 'results', 'data', 'baseline', f'{os.path.basename(self.model_name)}')
        self.save_file_name = os.path.join(self.save_folder, self.file_name)
