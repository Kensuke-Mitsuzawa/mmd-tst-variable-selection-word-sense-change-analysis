""""A script for getting statistics of the word embeddings."""

import numpy as np
import json
from pathlib import Path
import typing
import logzero
import pickle
import itertools

logger = logzero.logger

# ----------------------------------------
# you define the input path here
PATH_WORD_EMBEDDING = './resources/original/wv.npy'
PATH_DICTIONARY_PICKLE = './resources/original/freq100.pkl'

# path check
assert Path(PATH_WORD_EMBEDDING).exists(), f"Path not found: {PATH_WORD_EMBEDDING}"
assert Path(PATH_DICTIONARY_PICKLE).exists(), f"Path not found: {PATH_DICTIONARY_PICKLE}"

# loading the dictionary
file_dictionary = pickle.load(Path(PATH_DICTIONARY_PICKLE).open('rb'))
logger.info(f'#vocabulary: {len(file_dictionary)}')
# end with

# loading the numpy file
embedding_array = np.load(PATH_WORD_EMBEDDING)
logger.info(f'embedding_array.shape: {embedding_array.shape}')

# dividing the array sample size by the vocabulary size
n_size_v = len(file_dictionary)
n_time_epoch = embedding_array.shape[0] / n_size_v
assert n_time_epoch.is_integer(), f"n_time_epoch is not an integer: {n_time_epoch}"
logger.info(f'n_time_epoch: {n_time_epoch}')

seq_comb = itertools.combinations(range(int(n_time_epoch)), 2)
n_comb = len(list(seq_comb))
logger.info(f'n_comb: {n_comb}')