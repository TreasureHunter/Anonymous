import datasets

def tokenize_dataset(dataset, tokenizer, model_name):
    def tokenize_function(examples):
        examples['sentence']=tokenizer.apply_chat_template([{"role": "user", "content": examples['sentence']}], tokenize=False, add_generation_prompt=True)
        return tokenizer(
            examples["sentence"],
            padding=False, 
            return_tensors=None  
        )
    tokenized_datasets = dataset.map(tokenize_function, batched=False)
    return tokenized_datasets


def remove_str_columns(dataset):
    remove_keys = {k for k, v in dataset.features.items() if v.dtype == 'string'}
    dataset = dataset.remove_columns(list(remove_keys))
    return dataset
