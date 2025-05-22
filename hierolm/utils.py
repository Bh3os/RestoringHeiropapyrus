#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_sents(sents, pad_token):
    """
    Pad a batch of sentences to the maximum length.
    
    Args:
        sents: List of sentences, where each sentence is a list of tokens
        pad_token: The token to use for padding
        
    Returns:
        sents_padded: List of sentences, where each sentence is padded to the length of the longest sentence
    """
    sents_padded = []

    max_len = max([len(sent) for sent in sents])
    for sent in sents:
        sents_padded.append(sent + [pad_token]*(max_len-len(sent)))

    return sents_padded


def read_corpus(file_path):
    """
    Read in a corpus file where each line has a sequence of tokens.
    
    Args:
        file_path: Path to the corpus file
        
    Returns:
        src: List of source sentences, where each sentence is a list of tokens
        tgt: List of target sentences, where each sentence is a list of tokens (shifted by one position)
    """
    src = []
    tgt = []

    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            s = line.split(" ")
            src.append(s[:-1])
            tgt.append(s[1:])

    return src, tgt


def batch_iter(data, batch_size, shuffle=False):
    """
    Yield batches of source and target sentences reverse sorted by length.
    
    Args:
        data: List of (source_sent, target_sent) pairs
        batch_size: Batch size
        shuffle: Whether to shuffle the data before batching
        
    Yields:
        src_sents: List of source sentences
        tgt_sents: List of target sentences
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
