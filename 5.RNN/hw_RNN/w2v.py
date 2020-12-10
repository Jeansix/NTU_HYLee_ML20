import os
import numpy as np
import pandas as pd
import argparse
from gensim.models import word2vec


def train_word2vec(x):
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model

