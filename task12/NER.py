# Installing extra-dependencies
# pip -q install git+https://www.github.com/keras-team/keras-contrib.git sklearn-crfsuite
import tensorflow as tf
import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("reviews.csv", encoding="UTF-8")
data = data.fillna(method="ffill")

print("Number of sentences: ", len(data.groupby(['Sentence #'])))

words = list(set(data["Word"].values))
n_words = len(words)
print("Number of words in the dataset: ", n_words)

tags = list(set(data["Tag"].values))
print("Tags:", tags)
n_tags = len(tags)
print("Number of Labels: ", n_tags)

print("What the dataset looks like:")
# Show the first 10 rows
data.tail(10)
# [OUTPUT]
# '''Number of sentences:  47959
# Number of words in the dataset:  35178
# Tags: ['O', 'I-nat', 'I-eve', 'B-nat', 'I-art', 'B-gpe', 'I-org', 'I-gpe', 'B-per', 'I-per', 'B-eve', 'I-tim', 'B-geo', 'I-geo', 'B-org', 'B-art', 'B-tim']
# Number of Labels:  3