#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class HieroLM(nn.Module):
    """
    Hierarchical Language Model for hieroglyphic data.
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate):
        """
        Initialize the model.
        
        Args:
            embed_size: Size of the word embeddings
            hidden_size: Size of the hidden states
            vocab: Vocabulary
            dropout_rate: Dropout rate
        """
        super(HieroLM, self).__init__()
        src_pad_token_idx = vocab.vocab['<pad>']
        self.embed_size = embed_size
        self.model_embeddings = nn.Embedding(len(vocab.vocab), embed_size, padding_idx=src_pad_token_idx)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        
        # Encoder LSTM
        self.encoder = nn.LSTM(embed_size, hidden_size, bias=True, bidirectional=False)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Projection layer to vocabulary
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.vocab), bias=False)

    def forward(self, source: List[List[str]], target: List[List[str]], device) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            source: List of source sentences
            target: List of target sentences
            device: Device to run the model on
            
        Returns:
            scores: Log-likelihood scores for each sentence in the batch
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.vocab.to_input_tensor(source, device=device)  # Tensor: (src_len, b)
        target_padded = self.vocab.vocab.to_input_tensor(target, device=device)  # Tensor: (tgt_len, b)

        # Encode the source sentences
        enc_hiddens = self.encode(source_padded, source_lengths)
        
        # Apply dropout
        enc_hiddens = self.dropout(enc_hiddens)

        # Project to vocabulary space and apply log-softmax
        P = F.log_softmax(self.target_vocab_projection(enc_hiddens), dim=-1)

        # Zero out probabilities for padding tokens
        target_masks = (target_padded != self.vocab.vocab['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded.unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """
        Encode the source sentences.
        
        Args:
            source_padded: Tensor of (src_len, b) containing the padded source sentences
            source_lengths: List of source sentence lengths
            
        Returns:
            enc_hiddens: Tensor of (src_len, b, h) containing the encoder hidden states
        """
        # Embed the source sentences
        X = self.model_embeddings(source_padded)
        
        # Pack the sentences
        X_packed = pack_padded_sequence(X, source_lengths)
        
        # Pass through LSTM
        enc_hiddens_packed, (last_hidden, last_cell) = self.encoder(X_packed)
        
        # Unpack the hidden states
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens_packed)

        return enc_hiddens

    def predict(self, source: List[List[str]], target: List[List[str]], device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Predict target words given source sentences.
        
        Args:
            source: List of source sentences
            target: List of target sentences
            device: Device to run the model on
            
        Returns:
            predictions: Tensor of word predictions
            target_masks: Tensor of masks for target words
            target_padded: Tensor of padded target sentences
            source_lengths: List of source sentence lengths
        """
        source_lengths = [len(s) for s in source]
        source_padded = self.vocab.vocab.to_input_tensor(source, device=device)  # Tensor: (src_len, b)
        target_padded = self.vocab.vocab.to_input_tensor(target, device=device)  # Tensor: (tgt_len, b)
        
        # Encode the source sentences
        enc_hiddens = self.encode(source_padded, source_lengths)
        
        # Apply dropout
        enc_hiddens = self.dropout(enc_hiddens)

        # Project to vocabulary space and apply log-softmax
        P = F.log_softmax(self.target_vocab_projection(enc_hiddens), dim=-1)

        # Zero out probabilities for padding tokens
        target_masks = (target_padded != self.vocab.vocab['<pad>']).float()

        # Get the predictions
        predictions = torch.argmax(P, dim=-1) * target_masks

        return predictions, target_masks, target_padded, source_lengths
    
    def predict_realtime(self, source: List[List[str]], device) -> str:
        """
        Predict the next word in real-time given a source sentence.
        
        Args:
            source: List of source sentences
            device: Device to run the model on
            
        Returns:
            prediction: Predicted next word
        """
        source_lengths = [len(s) for s in source]
        source_padded = self.vocab.vocab.to_input_tensor(source, device=device)  # Tensor: (src_len, b)
        
        # Encode the source sentences
        enc_hiddens = self.encode(source_padded, source_lengths)
        
        # Apply dropout
        enc_hiddens = self.dropout(enc_hiddens)

        # Project to vocabulary space and apply log-softmax
        P = F.log_softmax(self.target_vocab_projection(enc_hiddens), dim=-1)

        # Get the prediction for the last token in the sequence
        prediction_idx = torch.argmax(P, dim=-1)[-1][0].cpu().item()
        prediction = self.vocab.vocab.id2word[prediction_idx]

        return prediction

    @property
    def device(self) -> torch.device:
        """
        Return the device on which the model is.
        
        Returns:
            device: The device on which the model is
        """
        return self.model_embeddings.weight.device

    @staticmethod
    def load(model_path: str):
        """
        Load a model from a file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            model: The loaded model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = HieroLM(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])
        return model

    def save(self, path: str):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
