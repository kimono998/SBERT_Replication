from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from datasets import Dataset, DatasetDict
import pandas as pd
SEED = 42
BATCH_SIZE = 16


def get_datasets(mode = 'nli', test_id = None, seed = None, link = None, test_only = False):
    if seed:
        torch.manual_seed(seed)
    # Load and tokenize SNLI dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if mode == 'nli':
        snli_dataset = load_dataset("stanfordnlp/snli")
        tokenized_snli = prepare_dataset(snli_dataset, tokenizer, mode='snli')
        # Load and tokenize MultiNLI dataset
        multi_nli_dataset = load_dataset("nyu-mll/multi_nli")
        tokenized_multi_nli = prepare_dataset(multi_nli_dataset, tokenizer, mode='multi-nli')
        # Concatenate datasets
        combined_train = ConcatDataset([tokenized_snli['train'], tokenized_multi_nli['train']])
        combined_validation = ConcatDataset([tokenized_snli['validation'], tokenized_multi_nli['validation_matched']])
        combined_test = ConcatDataset([tokenized_snli['test'], tokenized_multi_nli['validation_mismatched']])
        # Create DataLoaders
        train_dataloader = DataLoader(combined_train, shuffle=True, batch_size=BATCH_SIZE)
        validation_dataloader = DataLoader(combined_validation, batch_size=BATCH_SIZE)
        test_dataloader = DataLoader(combined_test, batch_size=BATCH_SIZE)

    if mode == 'sts' or mode == 'sts_b':
        sts_dataset = load_dataset(link if link else "mteb/stsbenchmark-sts")
        tokenized_sts = prepare_dataset(sts_dataset, tokenizer, mode = mode)
        if test_only:
            test_dataloader = DataLoader(tokenized_sts['test'], shuffle=True, batch_size=BATCH_SIZE)
            return test_dataloader

        train_dataloader = DataLoader(tokenized_sts['train'], shuffle=True, batch_size=BATCH_SIZE)
        validation_dataloader = DataLoader(tokenized_sts['validation'], batch_size=BATCH_SIZE)
        test_dataloader = DataLoader(tokenized_sts['test'], batch_size=BATCH_SIZE)
    if mode == 'afs':
        afs_dataset = Dataset.from_csv('data/afs_data.csv')
        tokenized_afs = prepare_dataset(afs_dataset, tokenizer, 'afs')
        train_data, val_data = split_afs(tokenized_afs, test_id)

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
        validation_dataloader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE)
        return train_dataloader, validation_dataloader

    return train_dataloader, validation_dataloader

def split_afs(dataset, test_id):
    train_data = dataset.filter(lambda x: x['dataset_id'] != test_id)
    val_data = dataset.filter(lambda x: x['dataset_id'] == test_id)

    return train_data, val_data
def prepare_dataset(dataset, tokenizer, mode='snli'):
    # Tokenize dataset
    tokenized_datasets = dataset.map(lambda x: tokenize_ds(tokenizer, x, True if mode == 'sts' or mode =='afs' or mode == 'sts_b' else False), batched=True)
    # Remove unnecessary columns
    if mode =='snli':
        columns = ['premise', 'hypothesis']
    if mode =='multi-nli':
        columns = [
            'promptID', 'pairID', 'premise_binary_parse', 'premise_parse',
            'hypothesis_binary_parse', 'hypothesis_parse', 'genre','premise','hypothesis'
        ]
    if mode == 'sts_b':
        columns = ['split', 'genre', 'dataset', 'year', 'sid']
    if mode == 'sts':
        columns = ['split']
    if mode != 'afs':
        tokenized_datasets = tokenized_datasets.remove_columns(columns)
        # Rename column
    tokenized_datasets = tokenized_datasets.rename_column("score" if mode == 'sts' or mode == 'afs' else 'label' , "labels")
    # Set format
    tokenized_datasets.set_format("torch")

    return tokenized_datasets


def tokenize_ds(tokenizer, examples, sts = False):
    s1_target = 'sentence1' if sts else 'premise'
    s2_target = 'sentence2' if sts else 'hypothesis'

    sentence_1 = tokenizer(examples[s1_target], padding='max_length', truncation=True, max_length=512)
    sentence_2 = tokenizer(examples[s2_target], padding='max_length', truncation=True, max_length=512)

    return {
        'input_ids_sentence_1': sentence_1['input_ids'],
        'attention_mask_sentence_1': sentence_1['attention_mask'],
        'input_ids_sentence_2': sentence_2['input_ids'],
        'attention_mask_sentence_2': sentence_2['attention_mask'],
        'label': examples['label'] if not sts else examples['score']
    }


def prepare_model_weights(path):
    checkpoint = torch.load(path, map_location='cpu')

    # should be adapted, this one works with classifier objective
    keys_to_keep = [key for key in checkpoint.keys() if key not in ['weight_matrix.weight', 'weight_matrix.bias']]
    # Create a new state dictionary with only the keys to keep
    filtered_state_dict = {key: checkpoint[key] for key in keys_to_keep}
    new_state_dict = {k.replace('base_model.', '', 1): v for k, v in filtered_state_dict.items()}

    return new_state_dict
    # executed like this base_model.load_state_dict(new_state_dict)


