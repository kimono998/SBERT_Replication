import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import json

# bert works with 512 x 768 embeddings
# 512 is context limit (num input tokens)
# 768 is dimension of embeddings
# pool into 1x768

MODULE_SAVE_PATH = 'checkpoints'
class SBert_Classification_Objective(nn.Module):
    def __init__(self, base_model, hidden_size=768, num_classes=3, sts_eval = False):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size # 768 default bert
        self.num_classes = num_classes # 3 for NLI
        self.classifier = nn.Linear(self.hidden_size*3, num_classes)
        self.sts_eval = sts_eval

    # https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py
    # https://stackoverflow.com/questions/65083581/how-to-compute-mean-max-of-huggingface-transformers-bert-token-embeddings-with-a
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() #we differentiate between padding tokens and actual attentioned tokens.
        sum_embeddings = torch.sum(token_embeddings*input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        output_vector = sum_embeddings / sum_mask
        # print(output_vector.shape) # 1x768 expected
        return output_vector


    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        # Encode both inputs
        output1 = self.base_model(input_ids=input_ids1, attention_mask=attention_mask1).last_hidden_state
        output2 = self.base_model(input_ids=input_ids2, attention_mask=attention_mask2).last_hidden_state

        # Mean pooling
        mean_pooled_output1 = self.mean_pooling(output1, attention_mask1)
        mean_pooled_output2 = self.mean_pooling(output2, attention_mask2)

        if self.sts_eval:
            similarity_score = F.cosine_similarity(mean_pooled_output1,
                                                   mean_pooled_output2)  # sim score between the outputs
            similarity_score = torch.clamp(similarity_score, min=-1.0, max=1.0)

            return similarity_score

        else:
            # Calculate absolute difference and concatenate
            abs_diff = torch.abs(mean_pooled_output1 - mean_pooled_output2) # (a,b,|a-b|)
            combined_output = torch.cat((mean_pooled_output1, mean_pooled_output2, abs_diff), 1)
            # print(combined_output.shape)
            logits = self.classifier(combined_output) # output  raw logits that go into the loss fn

            return logits

    def print_name(self):
        print(self.__class__.__name__)

    def mode_sts_eval(self):
        self.sts_eval = True

    def mode_normal(self):
        self.sts_eval = False


    def save(self, suff = None):
        """
        Saves all elements for this seq. sentence embedder into different sub-folders
        """

        modules = self._modules  # Assuming _modules is a dict of modules
        os.makedirs(MODULE_SAVE_PATH, exist_ok=True)

        for idx, (name, module) in enumerate(modules.items()):
            module_name = f'{idx}_{name}'
            if suff:
                module_name = f'{module_name}_{suff}'
            module_path = os.path.join(MODULE_SAVE_PATH, f'{module_name}.pth')
            torch.save(module.state_dict(), module_path)

        metadata = [{'idx': idx, 'name': name, 'path': os.path.basename(module_path)} for idx, (name, module) in
                    enumerate(modules.items())]

        json_name = f'{self.__class__.__name__}_{suff}_modules.json' if suff\
                    else f'{self.__class__.__name__}_modules.json'

        with open(os.path.join(MODULE_SAVE_PATH, json_name), 'w') as fOut:
            json.dump(metadata, fOut, indent=2)



class SBert_Regression_Objective(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() #we differentiate between padding tokens and actual attentioned tokens.
        sum_embeddings = torch.sum(token_embeddings*input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        output_vector = sum_embeddings / sum_mask
        # print(output_vector.shape)
        return output_vector


    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):

        output1 = self.base_model(input_ids1, attention_mask=attention_mask1).last_hidden_state
        output2 = self.base_model(input_ids2, attention_mask=attention_mask2).last_hidden_state

        # Mean pooling
        mean_pooled_output1 = self.mean_pooling(output1, attention_mask1)
        mean_pooled_output2 = self.mean_pooling(output2, attention_mask2)

        # Compute cosine similarity
        similarity_score = F.cosine_similarity(mean_pooled_output1, mean_pooled_output2) # sim score between the outputs

        return similarity_score

    def print_name(self):
        print(self.__class__.__name__)

    def save(self, suff = None):
        """
        Saves all elements for this seq. sentence embedder into different sub-folders
        """

        modules = self._modules  # Assuming _modules is a dict of modules
        os.makedirs(MODULE_SAVE_PATH, exist_ok=True)

        for idx, (name, module) in enumerate(modules.items()):
            module_name = f'{idx}_{name}'
            if suff:
                module_name = f'{module_name}_{suff}'
            module_path = os.path.join(MODULE_SAVE_PATH, f'{module_name}.pth')
            torch.save(module.state_dict(), module_path)

        metadata = [{'idx': idx, 'name': name, 'path': os.path.basename(module_path)} for idx, (name, module) in
                    enumerate(modules.items())]

        json_name = f'{self.__class__.__name__}_{suff}_modules.json' if suff\
                    else f'{self.__class__.__name__}_modules.json'

        with open(os.path.join(MODULE_SAVE_PATH, json_name), 'w') as fOut:
            json.dump(metadata, fOut, indent=2)


