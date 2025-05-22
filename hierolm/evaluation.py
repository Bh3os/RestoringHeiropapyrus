#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from .utils import batch_iter


def evaluate_ppl(model, dev_data, batch_size, device):
    """
    Evaluate the perplexity of a model on a dataset.
    
    Args:
        model: The model to evaluate
        dev_data: The dataset to evaluate on
        batch_size: The batch size to use for evaluation
        device: The device to run the evaluation on
        
    Returns:
        ppl: The perplexity of the model on the dataset
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents, device).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def evaluate_accuracy_and_f1(model, dev_data, batch_size, device):
    """
    Evaluate the accuracy and F1 score of a model on a dataset.
    
    Args:
        model: The model to evaluate
        dev_data: The dataset to evaluate on
        batch_size: The batch size to use for evaluation
        device: The device to run the evaluation on
        
    Returns:
        accuracy: The accuracy of the model on the dataset
        f1: The macro F1 score of the model on the dataset
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        total_correct = 0
        total_base = 0

        all_preds = []
        all_truths = []

        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            predictions, target_masks, target_padded, source_lengths = model.predict(src_sents, tgt_sents, device)

            correct_num = ((predictions == target_padded) * target_masks).sum().item()
            base_num = target_masks.sum().item()

            total_correct += correct_num
            total_base += base_num

            # Collect predictions and truths for F1 score
            for i in range(predictions.shape[1]):
                sent_len = source_lengths[i]
                pred = predictions[:sent_len, i].cpu()
                truth = target_padded[:sent_len, i].cpu()
                mask = target_masks[:sent_len, i].cpu().bool()
                
                all_preds.append(pred[mask])
                all_truths.append(truth[mask])

        # Calculate accuracy
        accuracy = total_correct / total_base if total_base > 0 else 0
        
        # Combine all predictions and truths
        all_preds = torch.cat(all_preds).numpy()
        all_truths = torch.cat(all_truths).numpy()
        
        # Calculate F1 score
        f1 = f1_score(all_truths, all_preds, average="macro") if len(all_truths) > 0 else 0

    if was_training:
        model.train()

    return accuracy, f1


def evaluate_multishot(model, test_sents, device):
    """
    Evaluate the multi-shot accuracy of a model on a dataset.
    
    Args:
        model: The model to evaluate
        test_sents: The sentences to evaluate on
        device: The device to run the evaluation on
        
    Returns:
        accs: List of accuracies for 1-shot, 2-shot, 3-shot, and 4-shot predictions
    """
    bases = [0 for _ in range(4)]
    counts = [0 for _ in range(4)]
    
    for i in tqdm(range(len(test_sents))):
        sent = test_sents[i]
        for idx in range(len(sent)-1):
            max_shot = min(4, len(sent)-idx-1)
            
            # Increment the base counter for all shots that can be attempted
            for s in range(max_shot):
                bases[s] += 1
                
            # Try to predict each shot
            for s in range(max_shot):
                tgt_idx = idx+s+1
                src = [sent[:tgt_idx]]
                prediction = model.predict_realtime(src, device)
                
                # If prediction is correct, increment the counter for this shot
                if prediction == sent[tgt_idx]:
                    counts[s] += 1
                else:
                    # If we fail at shot s, we'll fail at all higher shots
                    break
                    
    # Calculate accuracy for each shot
    return [counts[i]/bases[i] if bases[i] > 0 else 0 for i in range(4)]
