"""
HieroLM - A language model for hieroglyphic data.

This package implements a hierarchical language model for hieroglyphic data.
It supports training, evaluation, and interactive use of the model.
"""

from .model import HieroLM, Hypothesis
from .vocab import Vocab, VocabEntry
from .utils import read_corpus, batch_iter, pad_sents
from .evaluation import evaluate_ppl, evaluate_accuracy_and_f1, evaluate_multishot
from .decode import decode, multi_shot_decode, realtime_decode
from .train import train

__version__ = '0.1.0'
