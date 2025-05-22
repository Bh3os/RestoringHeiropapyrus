#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='HieroLM - A Hierarchical Language Model for Hieroglyphic Data')
    
    # Training parameters
    parser.add_argument('--max_epoch', default=50, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--train_batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--clip_grad', default=5.0, type=float, help='Gradient clipping value')
    parser.add_argument('--dropout', default=0, type=float, help='Dropout rate')
    parser.add_argument('--uniform_init', default=0.1, type=float, help='Uniform initialization range')
    parser.add_argument('--patience', default=5, type=int, help='Patience for early stopping')
    parser.add_argument('--max_num_trial', default=5, type=int, help='Maximum number of trials')
    parser.add_argument('--lr_decay', default=0.5, type=float, help='Learning rate decay')
    
    # Model parameters
    parser.add_argument('--embed_size', default=1024, type=int, help='Word embedding size')
    parser.add_argument('--hidden_size', default=1024, type=int, help='Hidden layer size')
    
    # Data parameters
    parser.add_argument('--dataset', default="aes", type=str, help='Dataset to use (aes, mixed, ramses)')
    parser.add_argument('--train_file', default="train.txt", type=str, help='Training data file')
    parser.add_argument('--dev_file', default="val.txt", type=str, help='Development data file')
    parser.add_argument('--test_file', default="test.txt", type=str, help='Test data file')
    parser.add_argument('--vocab_file', default="vocab.json", type=str, help='Vocabulary file')
    
    # Logging parameters
    parser.add_argument('--valid_niter', default=200, type=int, help='Validation frequency (iterations)')
    parser.add_argument('--log_every', default=10, type=int, help='Logging frequency (iterations)')
    
    # Model saving/loading
    parser.add_argument('--model_save_path', default="model.bin", type=str, help='Path to save the model')
    parser.add_argument('--model_path', default="model.bin", type=str, help='Path to load the model')
    
    # Other parameters
    parser.add_argument('--cuda', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--mode', default="train", type=str, choices=['train', 'decode', 'realtime'], help='Mode to run the model in')
    
    return parser.parse_args()
