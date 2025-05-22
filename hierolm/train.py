#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .utils import read_corpus, batch_iter
from .evaluation import evaluate_ppl, evaluate_accuracy_and_f1
from .model import HieroLM


def train(args, vocab, device):
    """
    Train the model.
    
    Args:
        args: Command line arguments
        vocab: Vocabulary
        device: Device to train on
    """
    print('Training HieroLM model...')
    
    # Load data
    train_file = f"data/{args.dataset}/{args.train_file}"
    train_data_src, train_data_tgt = read_corpus(train_file)
    print(f"Loaded training set from {train_file}")

    dev_file = f"data/{args.dataset}/{args.dev_file}"
    dev_data_src, dev_data_tgt = read_corpus(dev_file)
    print(f"Loaded dev set from {dev_file}")

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    # Model save path
    model_save_path = f"saved_models/{args.embed_size}_{args.hidden_size}_{args.dropout}_{args.dataset}_{args.model_save_path}"

    # Initialize model
    model = HieroLM(embed_size=args.embed_size,
                   hidden_size=args.hidden_size,
                   dropout_rate=args.dropout,
                   vocab=vocab)
    model.train()

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=f"./runs/{args.dataset}_{args.embed_size}_{args.hidden_size}_{args.dropout}")

    # Initialize model parameters
    if np.abs(args.uniform_init) > 0.:
        print(f'Uniformly initializing parameters in range [{-args.uniform_init}, {args.uniform_init}]')
        for p in model.parameters():
            p.data.uniform_(-args.uniform_init, args.uniform_init)

    # Move model to device
    model = model.to(device)
    print(f'Using device: {device}')

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training variables
    num_trial = 0
    train_iter = patience = cum_loss = report_loss = 0
    cum_tgt_words = report_tgt_words = cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('Beginning Maximum Likelihood training')

    # Train the model
    while epoch < args.max_epoch:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=args.train_batch_size, shuffle=True):
            train_iter += 1
            optimizer.zero_grad()

            batch_size = len(src_sents)

            # Forward pass
            example_losses = -model(src_sents, tgt_sents, device)  # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            # Update statistics
            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            report_tgt_words += tgt_word_num_to_predict
            cum_tgt_words += tgt_word_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            # Log training progress
            if train_iter % args.log_every == 0:
                print(f'Epoch {epoch}, iteration {train_iter}, avg. loss {report_loss / report_examples:.2f}, ' \
                      f'avg. ppl {math.exp(report_loss / report_tgt_words):.2f}, ' \
                      f'time elapsed {time.time() - begin_time:.2f} sec')

                # Write to TensorBoard
                writer.add_scalar('Loss/train', report_loss / report_examples, train_iter)
                writer.add_scalar('PPL/train', math.exp(report_loss / report_tgt_words), train_iter)

                report_loss = report_tgt_words = report_examples = 0.

            # Validate the model
            if train_iter % args.valid_niter == 0:
                print(f'Validating model at epoch {epoch}, iteration {train_iter}')
                
                # Calculate validation perplexity
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128, device=device)
                
                # Calculate validation accuracy and F1 score
                dev_accuracy, dev_f1 = evaluate_accuracy_and_f1(model, dev_data, batch_size=128, device=device)
                
                # Write to TensorBoard
                writer.add_scalar('PPL/dev', dev_ppl, train_iter)
                writer.add_scalar('Accuracy/dev', dev_accuracy, train_iter)
                writer.add_scalar('F1/dev', dev_f1, train_iter)
                
                print(f'Validation: ppl {dev_ppl:.2f}, accuracy {dev_accuracy:.4f}, F1 {dev_f1:.4f}')

                is_better = len(hist_valid_scores) == 0 or dev_ppl < min(hist_valid_scores)
                hist_valid_scores.append(dev_ppl)

                if is_better:
                    patience = 0
                    print(f'Saving model to {model_save_path}')
                    model.save(model_save_path)
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                else:
                    patience += 1
                    print(f'Not saving model, patience: {patience}/{args.patience}')
                    
                    if patience >= args.patience:
                        num_trial += 1
                        print(f'Patience exceeded, trying with learning rate {args.lr * args.lr_decay}')
                        if num_trial == args.max_num_trial:
                            print('Maximum number of trials reached. Stopping training.')
                            return

                        # Load the previously best model and optimizer
                        model = HieroLM.load(model_save_path)
                        model = model.to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * args.lr_decay)
                        
                        # Load the optimizer state
                        try:
                            optimizer.load_state_dict(torch.load(model_save_path + '.optim'))
                        except:
                            print('Failed to load optimizer states, using new optimizer with reduced learning rate.')
                            
                        # Update learning rate
                        args.lr *= args.lr_decay
                        patience = 0
                        
                # If the validation score is the best so far, or every 10 validations, print the training time
                if is_better or valid_num % 10 == 0:
                    print(f'Training time: {(time.time() - train_time) / 60:.2f} min')
                    train_time = time.time()
                
                valid_num += 1

        # Print epoch stats
        print(f'Finished epoch {epoch}, iteration {train_iter}')
        epoch_time = time.time() - begin_time
        print(f'Epoch time: {epoch_time / 60:.2f} min')
        
        # Write cumulative stats to TensorBoard
        writer.add_scalar('Loss/epoch', cum_loss / cum_examples, epoch)
        writer.add_scalar('PPL/epoch', math.exp(cum_loss / cum_tgt_words), epoch)
        
        cum_loss = cum_tgt_words = cum_examples = 0.
        
    writer.close()
    print('Training complete!')
