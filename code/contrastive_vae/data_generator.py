"""Library to generate training data for DeepJoke Contrastive VAE model.

This library reads data from two sources:
  -positive examples: in json format
  -contrastive example: in csv format

It then cleans them, assign score 0 to the contrastive examples, assigns score
1 to positive class, combine them, split into training and validation sets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import bleach
import logging
import numpy as np
import os
import pandas as pd
import re

from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from unidecode import unidecode

# Set numpy seed
np.random.seed(123)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGenerator(object):
  """Class to generate data for DeepJoke Contrastive VAE model.
  """

  def __init__(self,
               positive_data_path,
               contrastive_data_path,
               max_nb_words=50000,
               max_nb_examples=None,
               max_sequence_length=200,
               validation_split=0.1):
      self.positive_data_path = positive_data_path
      self.contrastive_data_path = contrastive_data_path
      self.max_nb_words = max_nb_words
      self.max_nb_examples = max_nb_examples
      self.max_sequence_length = max_sequence_length
      self.validation_split = validation_split

  def read_and_clean(self):
    """Reads in postive and contrastive data sets and cleans them.

    Returns:
      texts: A list of positive and contrastive example texts. Order is
          randomized.
      scores: A list of scores corresponding to texts.
    """
    def clean_punc(s):
      """Helper function to clean up punctuations.
      """
      s = s.replace("...", " ")
      s = s.replace("..", " ")
      s = s.replace("?", " ? ")
      s = s.replace(".", " . ")
      s = s.replace(",", " , ")
      s = s.replace("!", " , ")
      return s

    def clean_html(s):
      """Helper function to clean up html.
      """
      s = BeautifulSoup(s, "lxml")
      s = bleach.clean(s.text, tags=[], strip=True)
      s = re.sub(r'http\S+', 'url', s)
      return s

    # Read in positive and contrastive data
    positive_data = pd.read_csv(self.positive_data_path, encoding='utf-8')
    contrastive_data = pd.read_csv(self.contrastive_data_path, encoding='utf-8')
    logger.info("Read in {} positive examples".format(positive_data.shape[0]))
    logger.info("Read in {} contrastive examples".format(
        contrastive_data.shape[0]))

    # Unify column names, assign appropriate scores, and then concat the two
    # datasets
    positive_data["text"] = positive_data["Joke"]
    positive_data["score"] = 1
    positive_data = positive_data[["text", "score"]]
    contrastive_data["score"] = 0
    contrastive_data = contrastive_data[["text", "score"]]
    data = pd.concat([positive_data, contrastive_data], axis=0,
                     ignore_index=True)
    logger.info("Combined data has {} examples".format(data.shape[0]))
    texts = data["text"].tolist()
    scores = data["score"].tolist()

    # Clean punctuations, html, get rid of html, convert to lower case
    texts = map(clean_html, texts)
    texts = [unidecode(text) for text in texts]
    texts = map(clean_punc, texts)
    texts = map(lambda s: s.lower(), texts)
    logger.info("Done cleaning data")

    return texts, scores

  def tokenize(self, texts):
    """Tokenizes text strings into integers.

    Args:
      texts: A list of text strings.

    Returns:
      sequences: A list of tokenized texts
      tokenizer: A Keras tokenizer object
    """
    filters = '"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n'
    tokenizer = Tokenizer(num_words=self.max_nb_words, filters=filters)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    num_words = min(self.max_nb_words, len(word_index))
    logger.info('Found {} unique words; using {} unique words.'.format(len(word_index), num_words))
    return sequences, tokenizer

  def pad(self, sequences):
    """Pads sentences to a certain length, for LSTM to consume.
    """
    padded_sequences = pad_sequences(
        sequences, maxlen=self.max_sequence_length,
        padding='post', truncating='post')
    # labels for l model
    return padded_sequences

  def split(self, sequences, scores):
    """Shuffles and splits data into training and validation sets. If
    max_nb_examples is not None, shrink the number of examples to speed up
    dev runs.

    Args:
      sequences: An array of sequences.
      scores: An array of scores.

    Returns:
      x_train: Training set sequences
      s_train: Training set scores
      x_val: Validation set sequences
      s_val: Validation set scores
    """

    # Shuffle
    idx = np.arange(sequences.shape[0])
    np.random.shuffle(idx)
    x = sequences[idx]
    s = scores[idx]

    # Maybe shrink
    if self.max_nb_examples:
      nb_examples = min(self.max_nb_examples, x.shape[0])
      x = x[:nb_examples]
      s = s[:nb_examples]
      logger.info("Using {} examples".format(nb_examples))

    # Split
    num_val = int(self.validation_split * x.shape[0])
    x_train = x[:-num_val]
    s_train = s[:-num_val]
    x_val = x[-num_val:]
    s_val = s[-num_val:]

    return x_train, s_train, x_val, s_val

  def generate(self):
    """Main function to generate data for deepjoke models.

    Returns:
      x_train:
      s_train:
      x_val:
      s_val:
      tokenizer:
      texts:
    """

    texts, scores = self.read_and_clean()
    sequences, tokenizer = self.tokenize(texts)
    sequences = self.pad(sequences)
    scores = np.asarray(scores)
    x_train, s_train, x_val, s_val = self.split(sequences, scores)

    logger.info("Shape of training features: {}".format(x_train.shape))
    logger.info("Shape of training scores: {}".format(s_train.shape))
    logger.info("Shape of training features: {}".format(x_val.shape))
    logger.info("Shape of training scores: {}".format(s_val.shape))

    return x_train, s_train, x_val, s_val, tokenizer, texts, scores


def run():
  """Construct a DataGenerator object and genrate data using default settings.
  """
  cur_path = os.getcwd()
  short_jokes_path = cur_path.replace(
      "/code/contrastive_vae",
      "/data/short-jokes-dataset/shortjokes.csv")
  hacker_news_path = cur_path.replace(
      "/code/contrastive_vae",
      "/data/hacker-news-dataset/hacker_news_subset_10_to_200.csv")
  data_gen = DataGenerator(
      positive_data_path=short_jokes_path,
      contrastive_data_path=hacker_news_path)
  data_generated = data_gen.generate()
  file_name = "data_generated.p"
  pickle.dump(data_generated, open(file_name, "wb"))
  logger.info("Generated data stored as {}".format(file_name))


if __name__ == "__main__":
  run()