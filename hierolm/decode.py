#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from tqdm import tqdm

from .utils import read_corpus
from .evaluation import evaluate_ppl, evaluate_accuracy_and_f1, evaluate_multishot
from .model import HieroLM


def read_sents(file_path):
    """
    Read sentences from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        sents: List of sentences, where each sentence is a list of tokens
    """
    sents = []

    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            s = line.split(" ")
            sents.append(s)

    return sents


def decode(args, device):
    """
    Decode using the model.
    
    Args:
        args: Command line arguments
        device: Device to run decoding on
    """
    # Load test data
    test_file = f"data/{args.dataset}/{args.test_file}"
    print(f"Loading test data from {test_file}")
    test_data_src, test_data_tgt = read_corpus(test_file)
    test_data = list(zip(test_data_src, test_data_tgt))

    # Load the model
    model_path = f"saved_models/{args.embed_size}_{args.hidden_size}_{args.dropout}_{args.dataset}_{args.model_path}"
    print(f"Loading model from {model_path}")
    model = HieroLM.load(model_path)
    model = model.to(device)

    # Evaluate the model
    test_ppl = evaluate_ppl(model, test_data, batch_size=128, device=device)
    test_accuracy, test_f1 = evaluate_accuracy_and_f1(model, test_data, batch_size=128, device=device)

    print(f'Test results: ppl {test_ppl:.2f}, accuracy {test_accuracy:.4f}, F1 {test_f1:.4f}')
    
    # Save results to a file
    with open(f"{args.dataset}_result.txt", "a+") as f:
        f.write(f"{args.embed_size},{args.hidden_size},{args.dropout},{test_ppl},{test_accuracy},{test_f1}\n")


def multi_shot_decode(args, device):
    """
    Evaluate multi-shot accuracy using the model.
    
    Args:
        args: Command line arguments
        device: Device to run decoding on
    """
    # Load test sentences
    test_file = f"data/{args.dataset}/{args.test_file}"
    print(f"Loading test sentences from {test_file}")
    test_sents = read_sents(test_file)

    # Load the model
    model_path = f"saved_models/{args.embed_size}_{args.hidden_size}_{args.dropout}_{args.dataset}_{args.model_path}"
    print(f"Loading model from {model_path}")
    model = HieroLM.load(model_path)
    model = model.to(device)

    # Evaluate multi-shot accuracy
    print("Evaluating multi-shot accuracy...")
    accs = evaluate_multishot(model, test_sents, device)
    
    # Print results
    for i in range(4):
        print(f"{i+1}-shot accuracy: {accs[i]:.4f}")
    
    # Save results to a file
    with open(f"{args.dataset}_multishot_result.txt", "a+") as f:
        f.write(f"{args.embed_size},{args.hidden_size},{args.dropout},{','.join([str(acc) for acc in accs])}\n")


def realtime_decode(args, device):
    """
    Run the model in real-time interactive mode.
    
    Args:
        args: Command line arguments
        device: Device to run decoding on
    """
    # Load the model
    model_path = f"saved_models/{args.embed_size}_{args.hidden_size}_{args.dropout}_{args.dataset}_{args.model_path}"
    print(f"Loading model from {model_path}")
    model = HieroLM.load(model_path)
    model = model.to(device)
    
    print("Enter a sequence of tokens separated by spaces (or 'q' to quit):")
    
    with torch.no_grad():
        while True:
            try:
                # Get input from user
                user_input = input("> ")
                
                if user_input.lower() == 'q':
                    break
                
                # Convert input to list of tokens
                tokens = user_input.strip().split()
                
                if not tokens:
                    print("Please enter at least one token.")
                    continue
                
                # Predict next token
                src = [tokens]
                prediction = model.predict_realtime(src, device)
                
                print(f"Predicted next token: {prediction}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
