#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter
from itertools import chain
import json
import torch
from typing import List
from .utils import read_corpus, pad_sents


class VocabEntry(object):
    """
    Vocabulary entry for a language.
    """
    def __init__(self, word2id=None):
        """
        Initialize VocabEntry.
        
        Args:
            word2id: Dictionary mapping words to indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0
            self.word2id['<unk>'] = 1
            self.word2id['<s>'] = 2
            self.word2id['</s>'] = 3
            
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def add(self, word):
        """
        Add a word to the vocabulary.
        
        Args:
            word: The word to add
            
        Returns:
            wid: The ID of the added word
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """
        Convert list of words or list of sentences to list of indices.
        
        Args:
            sents: List of words or list of sentences
            
        Returns:
            word_ids: List of indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """
        Convert list of indices to words.
        
        Args:
            word_ids: List of indices
            
        Returns:
            sents: List of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """
        Convert a list of sentences to a tensor.
        
        Args:
            sents: List of sentences
            device: Device to put the tensor on
            
        Returns:
            sents_var: Tensor of (max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        """
        Create a VocabEntry from a corpus.
        
        Args:
            corpus: List of sentences
            size: Size of the vocabulary
            freq_cutoff: Frequency cutoff for words in the vocabulary
            
        Returns:
            vocab_entry: VocabEntry created from the corpus
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry
    
    @staticmethod
    def from_subword_list(subword_list):
        """
        Create a VocabEntry from a list of subwords.
        
        Args:
            subword_list: List of subwords
            
        Returns:
            vocab_entry: VocabEntry created from the subword list
        """
        vocab_entry = VocabEntry()
        for subword in subword_list:
            vocab_entry.add(subword)
        return vocab_entry


class Vocab(object):
    """
    Vocabulary for a language model.
    """
    def __init__(self, vocab: VocabEntry):
        """
        Initialize Vocab.
        
        Args:
            vocab: The vocabulary entry
        """
        self.vocab = vocab

    @staticmethod
    def build(src_sents, tgt_sents) -> 'Vocab':
        """
        Build a vocabulary from source and target sentences.
        
        Args:
            src_sents: List of source sentences
            tgt_sents: List of target sentences
            
        Returns:
            vocab: Vocabulary built from the source and target sentences
        """
        print('initialize source vocabulary ..')
        src = VocabEntry.from_subword_list(src_sents)

        print('initialize target vocabulary ..')
        tgt = VocabEntry.from_subword_list(tgt_sents)

        return Vocab(src, tgt)

    def save(self, file_path):
        """
        Save the vocabulary to a file.
        
        Args:
            file_path: Path to save the vocabulary
        """
        with open(file_path, 'w') as f:
            json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), f, indent=2)

    @staticmethod
    def load(file_path):
        """
        Load the vocabulary from a file.
        
        Args:
            file_path: Path to the vocabulary file
            
        Returns:
            vocab: Vocabulary loaded from the file
        """
        entry = json.load(open(file_path, 'r'))
        word2id = entry

        return Vocab(VocabEntry(word2id))

    def __repr__(self):
        return 'Vocab(words %d)' % (len(self.vocab))
