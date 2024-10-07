import os

from jupyter_server.auth import passwd
from transformers import BertModel
from model import *
from train import training
from data_utils import *
from eval import test

STS_DATASETS = {'sts_12': 'mteb/sts12-sts',
            'sts_13': 'mteb/sts13-sts',
            'sts_14': 'mteb/sts14-sts',
            'sts_15': 'mteb/sts15-sts',
            'sts_16': 'mteb/sts16-sts',
            'sts_b': 'mteb/stsbenchmark-sts'
            }

SEEDS = [0,1,2,3,4,5,6,7,8,9]

# use
# train base bert on nli.
def train_nli(name):
    train_dataloader = get_nli_data()
    print('dataloader_done')
    _, test_dataloader = get_sts_data('mteb/stsbenchmark-sts', 42)
    base_model = BertModel.from_pretrained("bert-base-uncased")
    mod = SBert_Classification_Objective(base_model)
    trained_model = training(model = mod,
                             train_dataloader = train_dataloader,
                             val_dataloader = test_dataloader,
                             sts_eval=True,
                             name = f'{name}'
                             )
    test(model=trained_model, test_dataloader=test_dataloader, model_name = f'{name}', dataset_name='STSb')


# train base bert or nli bert on stsb dataset.
def stsb_w_seeds(base_model = None, name = 'BERT_NO_NLI_STSb_SEED_'):

    for seed in SEEDS:
        if not base_model:
            base_model = BertModel.from_pretrained("bert-base-uncased")

        mod = SBert_Regression_Objective(base_model=base_model)

        train_dataloader, test_dataloader = get_sts_data('mteb/stsbenchmark-sts', seed)

        mod = training(model = mod,
                         train_dataloader=train_dataloader,
                         val_dataloader = test_dataloader,
                         mode = 'sim',
                         name = f'{name}{seed}',
                         num_epochs=4)
        test(model=mod, test_dataloader=test_dataloader, model_name = f'{name}{seed}', dataset_name = 'STSb')


# evaluate with multiple seeds on all sts datasets
def eval_all_sts(base_model = None, name = 'None'):


    for dataset in STS_DATASETS.keys():
        for seed in SEEDS:
            if not base_model:
                base_model = BertModel.from_pretrained("bert-base-uncased")
            mod = SBert_Regression_Objective(base_model=base_model)

            _, test_dataloader = get_sts_data(STS_DATASETS[dataset], seed)

            try:
                test(model=mod, test_dataloader=test_dataloader, model_name=f'{name}_{seed}', dataset_name={dataset})
            except:
                print(f'Error occured, could not evaluate on {dataset} for seed {seed}')

def  main():

    # train NLI bert
    train_nli('SBERT_NLI_V0')
    # with 10 seeds
    # train no nli sbert on stsb
    stsb_w_seeds(name='BERT_NO_NLI_STSb_V0_SEED_')
    # with 10 seeds
    # train nli sbert on stsb

    # with 10 seeds
    # evaluate noNli_sbert_noStsb, nli_sbert_stsb, nli_sbert_no_stsb, and noNli_sbert_stsb on all stsb datasets

    # do sent eval

# # import datasets
# def main():
#     # stsb_w_seeds()
#     # train_stsb_only()
#     # nli done
#     # train_nli()
#     base_model = BertModel.from_pretrained("bert-base-uncased")
#     base_model.load_state_dict(torch.load('checkpoints/0_base_model_stsb_nli_trained_best_score.pth'))
#     model = SBert_Regression_Objective(base_model)
#
#     _, test_dataloader = get_sts_data('mteb/stsbenchmark-sts', 21)
#     # base_model = BertModel.from_pretrained("bert-base-uncased")
#     # base_model.load_state_dict(torch.load('checkpoints/0_base_model_best_score.pth'))
#     # trained_model = SBert_Regression_Objective(base_model)
#     test(model=model, test_dataloader=test_dataloader, model_name = 'stsb_nli_best_model', dataset_name = 'STSb')
#     # train_nli_stsb(model, 21)
#
#
#     # Train NLI -> Done
#     # Train STSb ONLY -> TBD (10 seeds)
#     # Train STSb w NLI -> TBD (10 seeds)





if __name__ == "__main__":
    main()
