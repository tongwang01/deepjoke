"""Drive to train Contrastive VAE model for Deep Joke.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import logging
import numpy as np
import os
from datetime import datetime

import data_generator
import contrastive_vae_model

# Set numpy sedd
np.random.seed(123)


def train(generated_data_path=None, epochs=1, examples_cap=None):
  """Function to train lstm cvae model.

  Args:
    generated_data_path: If not None, load pre generated data from this path
        instead of generating it anew.
    epochs: Epochs to train the model.
    examples_cap: If not None, restrict to only this many training examples.

  Return:
    Nothing.
  """

  # Prepare ModelConfig
  base_dir = os.getcwd().replace("/contrastive_vae", "")
  glove_dir = base_dir.replace("/code", "/data/glove.twitter.27B/")
  embedding_path = os.path.join(glove_dir, 'glove.twitter.27B.200d.txt')
  short_jokes_path = os.path.join(
      base_dir.replace("/code", '/data/short-jokes-dataset/'),
      "shortjokes.csv")
  hacker_news_path = os.path.join(
      base_dir.replace("/code", '/data/hacker-news-dataset/'),
      "hacker_news_subset_10_to_200.csv")
  model_dir = (base_dir +
               "/model_checkpoints/contrastive_vae/{:%Y%m%d_%H%M%S}".format(
          datetime.now()))

  model_config = contrastive_vae_model.ModelConfig(
      positive_data_path=short_jokes_path,
      contrastive_data_path=hacker_news_path,
      embedding_path=embedding_path,
      model_dir=model_dir,
      embedding_dim=200,
      batch_size=32,
      max_nb_words=100000,
      max_nb_examples=None,
      max_sequence_length=50,
      encoder_lstm_dims = [256, 128],
      decoder_lstm_dims = [128, 256],
      latent_dim=64,
      kl_weight=1.,
      optimizer="RMSprop")

  # Set up logging
  try:
    os.makedirs(model_config.model_dir)
  except:
    logger.info("Did not successfully make new model dir")
  logger = logging.getLogger("contrastive_vae")
  logger.setLevel(logging.INFO)
  logging.basicConfig(filename=(model_config.model_dir + '/model_log.log'),
                      level=logging.INFO)

  # Load or generate data
  logger.info("Loading or generating data...")
  if generated_data_path:
    x_train, s_train, x_val, s_val, tokenizer, _, _ = pickle.load(
        open(generated_data_path, "r"))
  else:
    data_gen = data_generator.DataGenerator(
        positive_data_path=short_jokes_path,
        contrastive_data_path=hacker_news_path)
    x_train, s_train, x_val, s_val, tokenizer, _, _ = data_gen.generate()
  logger.info("Done loading or generating data.")

  # Build and fit model
  contra_vae = contrastive_vae_model.ContraVAE(model_config, tokenizer)
  hist = contra_vae.fit(
      x_train=x_train,
      s_train=s_train,
      x_val=x_val,
      s_val=s_val,
      epochs=epochs,
      examples_cap=examples_cap)
  logger.info(hist.history)
  print("Done.")


if __name__ == "__main__":
  train(
      generated_data_path="data_generated.p",
      #generated_data_path=None,
      epochs=1,
      examples_cap=100000)