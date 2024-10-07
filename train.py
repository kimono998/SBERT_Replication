from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
# from torch.optim import AdamW
from transformers import AdamW
import torch.nn.functional as F
import torch
import datetime
import json
import transformers
from tqdm import tqdm
from typing import Dict

NO_DECAY = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# need a better evaluator - in the original script, they train on NLI but evaluate on STS. We need to save best STS eval
# model. Performance on NLI is meaningless.

def training(model: nn.Module,
             train_dataloader: torch.utils.data.DataLoader,
             val_dataloader: torch.utils.data.DataLoader,
             num_epochs: int = 1,
             optimizer_params: Dict = {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
             weight_decay: float = 0.01,
             max_grad_norm: float = 1,
             mode: str = 'cls',
             seed: bool = None,
             testid: bool = None,
             name = 'nli',
             sts_eval = False,
             ):

    eval_freq = max(1, int(0.1 * len(train_dataloader)))  # Ensure eval_freq is at least 1

    model = model.to(device)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    call_name = '{}_{}_Obj_{}'.format(name, mode, timestamp) if not seed else '{}_{}_Obj_{}_seed_{}'.format(name, mode, timestamp, seed)

    if testid:
        call_name += f'_tid_{testid}'

    writer = SummaryWriter(f'runs/{call_name}')

    loss_fn = nn.CrossEntropyLoss() if mode == 'cls' else nn.MSELoss()

    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in NO_DECAY)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in NO_DECAY)], 'weight_decay': 0.0}
    ]

    total_steps: int = len(train_dataloader) * num_epochs
    warmup_steps: float = int(0.1 * total_steps)


    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_params)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    training_steps: int = 0
    best_val_loss = 99999

    # initial model evaluation
    val_loss = validation_step(val_dataloader, model, loss_fn, sts_eval)
    writer.add_scalar('Loss/Val', val_loss, training_steps)
    print(f'Initial Val Loss: {val_loss}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss

    for epoch in range(num_epochs):

        # start training
        running_loss: float = 0.0
        train_progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                              desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

        for step, batch in train_progress:
            loss: float = training_step(model,
                                        batch,
                                        optimizer,
                                        scheduler,
                                        loss_fn,
                                        max_grad_norm
                                        )

            running_loss += loss
            training_steps += 1
            avg_loss = running_loss/(step+1)

            # log stuff
            writer.add_scalar('Loss/Train', avg_loss, training_steps)
            print(f'Batch Step/Avg.Loss: {training_steps}/{avg_loss}')
            if (step+1) % eval_freq == 0:
                val_loss = validation_step(val_dataloader, model, loss_fn, sts_eval=sts_eval)
                writer.add_scalar('Loss/Val', val_loss, training_steps)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save(suff = f'{name}_best_score')

        # evaluate at the end of the epoch
        val_loss = validation_step(val_dataloader, model, loss_fn, sts_eval=sts_eval)
        writer.add_scalar('Loss/Val', val_loss, training_steps)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(suff=f'{name}_best_score')

    print(f"Best Validation Loss: {best_val_loss}")
    print('Traning done. Saving model')
    model.save(suff = f'{name}_final')

    return model



def training_step(model, batch, optimizer, scheduler, loss_fn, max_grad_norm=1):
    input_1 = batch['input_ids_sentence_1'].to(device)
    input_2 = batch['input_ids_sentence_2'].to(device)
    attention_1 = batch['attention_mask_sentence_1'].to(device)
    attention_2 = batch['attention_mask_sentence_2'].to(device)
    labels = batch['labels'].to(device) # .to(torch.long)  # Ensure labels are of type long when training NLI

    optimizer.zero_grad()
    outputs = model(input_1, attention_1, input_2, attention_2)
    print(f'output shape: {outputs.shape} | label shape: {labels.shape}')

    loss = loss_fn(outputs, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    return loss.item()

def validation_step(val_dataloader, model, loss_fn, sts_eval=False):
    model = model.to(device)
    if sts_eval:
        model.mode_sts_eval()

    model.eval()
    running_vloss = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            input_1 = batch['input_ids_sentence_1'].to(device)
            input_2 = batch['input_ids_sentence_2'].to(device)
            attention_1 = batch['attention_mask_sentence_1'].to(device)
            attention_2 = batch['attention_mask_sentence_2'].to(device)
            labels = batch['labels'].to(device)# .to(torch.long)  # Ensure labels are of type long when training NLI

            outputs = model(input_1, attention_1, input_2, attention_2)
            if sts_eval:
                sim_loss = nn.MSELoss()
                loss = sim_loss(outputs, labels)
                running_vloss += loss.item()
            else:
                loss = loss_fn(outputs, labels)
                running_vloss += loss.item()

    avg_vloss = running_vloss / len(val_dataloader)
    model.zero_grad()
    model.train()

    if sts_eval:
        model.mode_normal()
    return avg_vloss
