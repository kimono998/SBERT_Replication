from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from scipy.stats import spearmanr, rankdata, pearsonr
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import os
import json

RES_PATH = 'data/results.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def calculate_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro')


def calculate_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')


def calculate_rho(y_true, y_pred):
    # convert to ranks, then get rho
    # model_ranks = rankdata(y_pred)
    # gold_ranks = rankdata(y_true)

    rho, p_value = spearmanr(y_pred, y_true)

    return rho


def calculate_pearson_r(y_true, y_pred):
    pearson_correlation, p_value = pearsonr(y_pred, y_true)
    return pearson_correlation


def write_to_json(data):
    if os.path.exists(RES_PATH):
        # Read the existing data
        with open(RES_PATH, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        # If file doesn't exist, initialize with empty dictionary
        existing_data = {}

    # Merge the existing data with the new data
    updated_data = merge_dicts(existing_data, data)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(RES_PATH), exist_ok=True)

    # Write the updated data back to the file
    with open(RES_PATH, 'w') as json_file:
        json.dump(updated_data, json_file, indent=4)

    print(f"Data written to '{RES_PATH}'")


def merge_dicts(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            merge_dicts(d1[k], v)
        else:
            d1[k] = v
    return d1


def get_metrics(y_true, y_pred, rank=True):
    if rank:
        test_rho = calculate_rho(y_true, y_pred)
        test_pearson = calculate_pearson_r(y_true, y_pred)

        return {"spearman_rho": test_rho,
                "pearson_r": test_pearson
                }
    else:
        test_accuracy = calculate_accuracy(y_true, y_pred)
        test_f1 = calculate_f1(y_true, y_pred)
        precision = calculate_precision(y_true, y_pred)
        recall = calculate_recall(y_true, y_pred)

        return {'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'test_precision': precision,
                'test_recall': recall
                }


def test(model, test_dataloader, model_name='SBERT', dataset_name='STSb', rank=True):
    model = model.to(device)
    model.eval()
    test_preds = []
    test_labels = []

    writer = SummaryWriter(log_dir=os.path.join('runs', model_name))

    # Initialize tqdm for test dataloader
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Testing', leave=False):
            input_1 = batch['input_ids_sentence_1'].to(device)
            input_2 = batch['input_ids_sentence_2'].to(device)
            attention_1 = batch['attention_mask_sentence_1'].to(device)
            attention_2 = batch['attention_mask_sentence_2'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_1, attention_1, input_2, attention_2)
            if rank:
                test_preds.extend(outputs.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
            else:
                _, predicted = torch.max(outputs, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
    metrics = get_metrics(test_labels, test_preds, rank=rank)
    # Log metrics to TensorBoard
    for key, value in metrics.items():
        writer.add_scalar(f'{model_name}/{key}', value)
    writer.close()

    # Write metrics to a JSON file
    write_to_json({model_name: {dataset_name: metrics}})




