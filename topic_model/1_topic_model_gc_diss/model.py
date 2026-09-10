#!/usr/bin/env python3

import os
import sys
from pathlib import Path
# General import list pulled from the collab code
import numpy as np
import pandas as pd
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
import matplotlib.pyplot as plt


DATA_PATH = 'data/gc_dissertations_combined_v2.csv'
TEXT_COLUMN = ''

# Validate that the source file exists.
print('Data source: ', DATA_PATH)

if not os.path.exists(DATA_PATH):
	print('Data source is not available or does not exist.')
	sys.exit(1)

# Settings pulled from the collab code
# np.random.seed is not a reliable way to handle randomness in Python/Numpy
rnd_gen = np.random.default_rng(seed=2026)

nltk.download('wordnet')
nltk.download('omw-1.4')
stemmer = SnowballStemmer('english')

# model settings
NUM_TOPICS = 20 # Should be a reasonable number of topic buckets
PASSES = 10 # How many times topics get bucketized?
WORKERS = 8

# data settings
USE_SUBSET = True
DO_TESTING = True
TRUNCATE_SIZE = 50000 # For testing
INSPECT_ROW = 1234 # A random row to inspect
UNSEEN_TEXT = '' # A novel stentence to fit to a bucket via the model

# dictionary settings
MAX_WORDS = 100000
MIN_WORDS = 15
MAX_PERCENTABE = 0.5

# Helper functions for Lemmatizing and Stemming
def lemmatize_stemming(test):
	return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
	if pd.isna(text) or not isinstance(text,str):
		return []

	result = []
	for token in gensim.utils.simple_preprocess(text):
		if token not in STOPWORDS and len(token) > 3:
			result.append(lemmatize_stemming(token))
	return result

# Load data
data = pd.read_csv(DATA_PATH)
print(data.head())
print('Missing values: \n', data.isna().sum())
