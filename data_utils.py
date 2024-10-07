from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from datasets import Dataset, DatasetDict


# datasets to consider
# NLI
# MultiNLI
# STS B
# all the other STS benchmarks

BATCH_SIZE = 16

def get_nli_data():
    torch.manual_seed(189)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", padding_side='right')
    datasets = ['stanfordnlp/snli', 'nyu-mll/multi_nli']

    tokenized = []
    for dataset in datasets:
        tokenized_data = preprocess_nli_dataset(dataset, tokenizer)
        tokenized.append(tokenized_data)
        print(tokenized_data)
    
    train_combined = ConcatDataset([tokenized[0]['train'], tokenized[1]['train']])
    # val_combined = ConcatDataset([tokenized[0]['validation'], tokenized[1]['validation_matched']])
    train_dataloader = DataLoader(train_combined, shuffle=True, batch_size=BATCH_SIZE)
    # validation_dataloader = DataLoader(val_combined, shuffle=True, batch_size=BATCH_SIZE)

    return train_dataloader

def get_sts_data(dataset, seed):
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", padding_side='right')

    tokenized_data = preprocess_sts(dataset, tokenizer)
    if dataset == 'mteb/stsbenchmark-sts':
        train_dataloader = DataLoader(tokenized_data['train'], shuffle=True, batch_size=BATCH_SIZE)
        test_dataloader = DataLoader(tokenized_data['test'], shuffle=True, batch_size=BATCH_SIZE)
        return train_dataloader, test_dataloader

    else:
        test_dataloader = DataLoader(tokenized_data['test'], shuffle=True, batch_size=BATCH_SIZE)
        return None, test_dataloader

def preprocess_nli_dataset(dataset: str, tokenizer):
    data = load_dataset(dataset)
    tokenized_data = data.map(
        lambda x: tokenize_ds(tokenizer, x, False),
        batched=True)
    
    # Handle each split separately
    filtered_data = {}
    for split in tokenized_data:
        # Filter out samples where the label is -1
        filtered_split = tokenized_data[split].filter(lambda x: x['labels'] != -1)
        
        # Print number of removed samples
        num_samples_before = len(tokenized_data[split])
        num_samples_after = len(filtered_split)
        num_removed = num_samples_before - num_samples_after
        print(f"Number of data points removed from {split}: {num_removed}")        
        filtered_data[split] = filtered_split
        
    filtered_data = DatasetDict(filtered_data)
    if dataset == 'nyu-mll/multi_nli':
        extra_col = [
            'promptID', 'pairID', 'premise_binary_parse', 'premise_parse',
            'hypothesis_binary_parse', 'hypothesis_parse', 'genre', 'premise', 'hypothesis','label'
        ]

    else:
        extra_col = ['premise', 'hypothesis', 'label']

    filtered_data = filtered_data.remove_columns(extra_col)
    # tokenized_data = tokenized_data.rename_columns("label", "labels")

    filtered_data.set_format("torch")

    return filtered_data

# def _sts_data(**kwargs):
#
#
#     if kwargs.benchmark:
#         dataset = 'mteb/stsbenchmark-sts'
#     else:
#         datasets = {'sts_12': 'mteb/sts12-sts',
#                     'sts_13': 'mteb/sts13-sts',
#                     'sts_14': 'mteb/sts14-sts',
#                     'sts_15': 'mteb/sts15-sts',
#                     'sts_16': 'mteb/sts16-sts',
#                     'sts_b': 'mteb/stsbenchmark-sts'
#                     }


def preprocess_sts(dataset: str, tokenizer):
    data = load_dataset(dataset)

    # Extract scores to compute normalization parameters
    if dataset == 'mteb/stsbenchmark-sts':
        scores = [item['score'] for item in data['train']]
    else:
        scores = [item['score'] for item in data['test']]

    min_score = min(scores)
    max_score = max(scores)
    print(max_score)
    print(min_score)

    def normalize_score(score):
        # normalize between -1 and 1
        return 2 * ((score - min_score) / (max_score - min_score)) - 1

    def tokenize_and_normalize(examples):
        s1_target = 'sentence1'
        s2_target = 'sentence2'
        sentence_1 = tokenizer(examples[s1_target], padding='max_length')
        sentence_2 = tokenizer(examples[s2_target], padding='max_length')
        normalized_scores = [normalize_score(score) for score in examples['score']]
        return {
            'input_ids_sentence_1': sentence_1['input_ids'],
            'attention_mask_sentence_1': sentence_1['attention_mask'],
            'input_ids_sentence_2': sentence_2['input_ids'],
            'attention_mask_sentence_2': sentence_2['attention_mask'],
            'labels': normalized_scores
        }

    if dataset == 'mteb/stsbenchmark-sts':
        extra_col = ['split', 'genre', 'dataset', 'year', 'sid', 'score']
    else:
        extra_col = ['split', 'score']


    tokenized_data = data.map(
        lambda x: tokenize_and_normalize(x),
        batched=True
    )

    tokenized_data = tokenized_data.remove_columns(extra_col)
    tokenized_data.set_format("torch")

    return tokenized_data


def tokenize_ds(tokenizer, examples, sts = False):
    s1_target = 'sentence1' if sts else 'premise'
    s2_target = 'sentence2' if sts else 'hypothesis'

    sentence_1 = tokenizer(examples[s1_target], padding='max_length')
    sentence_2 = tokenizer(examples[s2_target], padding='max_length')

    return {
        'input_ids_sentence_1': sentence_1['input_ids'],
        'attention_mask_sentence_1': sentence_1['attention_mask'],
        'input_ids_sentence_2': sentence_2['input_ids'],
        'attention_mask_sentence_2': sentence_2['attention_mask'],
        'labels': examples['label'] if not sts else examples['score']
    }




