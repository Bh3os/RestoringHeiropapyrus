#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch

from hierolm.parse import parse_args
from hierolm.vocab import Vocab
from hierolm.train import train
from hierolm.decode import decode, multi_shot_decode, realtime_decode


def main():
    """
    Main entry point for the program.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    # Load vocabulary
    vocab_file = f"data/{args.dataset}/{args.vocab_file}"
    print(f"Loading vocabulary from {vocab_file}")
    try:
        vocab = Vocab.load(vocab_file)
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        sys.exit(1)
    
    # Choose mode
    if args.mode == "train":
        train(args, vocab, device)
    elif args.mode == "decode":
        decode(args, device)
    elif args.mode == "multishot":
        multi_shot_decode(args, device)
    elif args.mode == "realtime":
        realtime_decode(args, device)
    else:
        print(f"Invalid mode: {args.mode}")
        print("Valid modes: train, decode, multishot, realtime")
        sys.exit(1)


if __name__ == "__main__":
    main()
