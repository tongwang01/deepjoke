"""Library to encapsulate Contrastive VAE model for Deep Joke.

Main changes compared to the previous lstm_cvae_model:
    -Use RecurrentShop to implement readout
    -Deeper LSTM networks
    -Use Twitter embedding.
    -VAE conditional on a binary "funny or not flag"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import logging
import numpy as np
import os

from keras import backend as K
from keras import optimizers
from keras.layers import Dense, LSTM, Embedding, Input, RepeatVector, Lambda, TimeDistributed
from keras.models import Model
from keras.layers.merge import concatenate
from recurrentshop import LSTMCell, RecurrentSequential

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfig():
  """Class to hold all model configs. Required to instantiate a LSTM CVAE
  model.
  """
  def __init__(self,
               positive_data_path,
               contrastive_data_path,
               embedding_path,
               model_dir,
               embedding_dim=200,
               batch_size=32,
               max_nb_words=100000,
               max_nb_examples=None,
               max_sequence_length=200,
               encoder_lstm_dims = [256, 128],
               decoder_lstm_dims = [128, 256],
               latent_dim=64,
               kl_weight=1.,
               optimizer=optimizers.RMSprop(clipnorm=1.)):
    self.positive_data_path = positive_data_path
    self.contrastive_data_path = contrastive_data_path
    self.embedding_path = embedding_path
    self.model_dir = model_dir
    self.embedding_dim = embedding_dim
    self.batch_size = batch_size
    self.max_nb_words = max_nb_words
    self.max_nb_examples = max_nb_examples
    self.max_sequence_length = max_sequence_length
    self.encoder_lstm_dims = encoder_lstm_dims
    self.decoder_lstm_dims = decoder_lstm_dims
    self.latent_dim = latent_dim
    self.kl_weight = kl_weight
    self.optimizer = optimizer


class ContraVAE(object):
  """Class to hold Contrastive VAE model for DeepJoke.
  """

  def __init__(self, model_config, tokenizer):
    self.config = model_config
    self.word_index = tokenizer.word_index
    self.num_words = min(int(model_config.max_nb_words),
                         len(tokenizer.word_index))
    self.tokenizer = tokenizer
    self._make_model()

  def _load_embedding(self):
    """Load and prepare embedding matrix"""
    logger.info("Loading embedding...")
    embeddings_index = {}
    f = open(self.config.embedding_path)
    for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros(
      (self.num_words + 1, self.config.embedding_dim))
    for word, i in self.word_index.items():
      if i >= self.config.max_nb_words:
        continue
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    return embedding_matrix

  def _make_model(self):
    """Constructs ContraVAE model.

    Returns:
      Nothing.
    """

    # Load embedding in Embedding layer
    logger.info("Making model...")
    embedding_matrix = self._load_embedding()
    embedding_layer = Embedding(self.num_words + 1,
                                self.config.embedding_dim,
                                weights=[embedding_matrix],
                                input_length=self.config.max_sequence_length,
                                trainable=False)

    # Q(z|X,c) -- encoder
    # Embedded sequence input
    sequence_inputs = Input(
        batch_shape=(self.config.batch_size, self.config.max_sequence_length),
        dtype='int32')
    embedded_sequence_inputs = embedding_layer(sequence_inputs)
    # Merge with score inputs
    score_inputs = Input(batch_shape=(self.config.batch_size, 1))
    score_inputs_repeated = RepeatVector(self.config.max_sequence_length)(
        score_inputs)
    last_layer = concatenate([embedded_sequence_inputs, score_inputs_repeated],
                             axis=2)
    # LSTM layers
    for dim in self.config.encoder_lstm_dims[:-1]:
      last_layer = LSTM(dim, return_sequences=True)(last_layer)
    last_layer = LSTM(self.config.encoder_lstm_dims[-1],
                      return_sequences=False)(last_layer)
    # Mean and std of z
    z_mean = Dense(self.config.latent_dim, activation='tanh')(last_layer)
    z_log_sigma = Dense(self.config.latent_dim, activation='tanh')(last_layer)

    # Sample z ~ Q(z|X,c)
    def sampling(args):
      z_mean, z_log_sigma = args
      epsilon = K.random_normal_variable(
          shape=(self.config.batch_size, self.config.latent_dim),
          mean=0., scale=1.)
      return z_mean + K.exp(z_log_sigma / 2.) * epsilon

    z = Lambda(sampling)([z_mean, z_log_sigma])

    # Second score inputs - at training time this is simply equal to
    # score_inputs; at sampling time this could vary.
    score_inputs2 = Input(batch_shape=(self.config.batch_size, 1))
    z_c = concatenate([z, score_inputs2], axis=1)
    # Repeat z_c so every timestep has access to it
    #z_c_repeated = RepeatVector(self.config.max_sequence_length)(z_c)

    # P(X|z,c) -- decoder.
    rnn = RecurrentSequential(decode=True,
                              output_length=self.config.max_sequence_length)
    rnn.add(LSTMCell(self.config.decoder_lstm_dims[0],
                     input_dim=self.config.latent_dim+1))
    for dim in self.config.decoder_lstm_dims[1:]:
      rnn.add(LSTMCell(dim))
    decoder_out = TimeDistributed(Dense(self.num_words + 1), activation='tanh')

    # Decoder output
    # x_decoded = rnn(z_c_repeated, ground_truth=sequence_inputs)
    h_decoded = rnn(z_c)
    x_decoded = decoder_out(h_decoded)

    # Construct models
    # VAE
    vae = Model([sequence_inputs, score_inputs,
                 score_inputs2], x_decoded)
    # Encoder
    encoder = Model([sequence_inputs, score_inputs], z_mean)
    # Generator
    generator_z_inputs = Input(
        batch_shape=(self.config.batch_size, self.config.latent_dim))
    generator_z_c = concatenate([generator_z_inputs, score_inputs2], axis=1)
    generator_h_decoded = rnn(generator_z_c)
    generator_x_decoded = decoder_out(generator_h_decoded)
    generator = Model([generator_z_inputs, score_inputs2], generator_x_decoded)

    # Define loss function
    kl_weight = self.config.kl_weight

    def recon_loss(y_true, y_pred):
      """E[log P(X|z,y)].
      """
      recon = K.mean(K.sparse_categorical_crossentropy(
          output=y_pred, target=y_true, from_logits=True), axis=1)
      return recon

    def kl_loss(y_true, y_pred):
      """D_KL(Q(z|X,y) || P(z|X)); calculate in closed form as both
      dist. are Gaussian.
      """
      kl = 0.5 * K.mean(
          K.exp(z_log_sigma) + K.square(z_mean) - 1. - z_log_sigma, axis=1)
      kl = kl * kl_weight
      return kl

    def vae_loss(y_true, y_pred):
      """Calculate loss = reconstruction loss + KL loss.
      """
      recon = recon_loss(y_true, y_pred)
      kl = kl_loss(y_true, y_pred)
      return recon + kl

    # Compile model
    vae.compile(loss=vae_loss, optimizer=self.config.optimizer,
                metrics=[recon_loss, kl_loss])

    self.vae = vae
    self.encoder = encoder
    self.generator = generator
    logger.info("Done making model.")

  def fit(self, x_train, s_train, x_val=None, s_val=None, epochs=1,
          examples_cap=None):
    """Fits vae model. Store output models, configs and associated
    tokenizer.

    Args:
      x_train: Sequence inputs for training.
      s_train: Score inputs for training.
      x_val: Sequence inputs for validation.
      s_val: Score inputs for validation.
      epochs: epochs to train the model for.
      examples_cap: If not None, restrict to only this many training examples.

    Returns:
      Nothing.
    """

    # Optionally restrict the number of training exmaples, to speed things up
    if examples_cap:
      x_train = x_train[:examples_cap]
      s_train = s_train[:examples_cap]
    logger.info("Training model on {} examples...".format(x_train.shape[0]))

    # Cut training and validation sets to multiples of batch_size
    train_cap = int(np.floor(
        x_train.shape[0] / self.config.batch_size) * self.config.batch_size)
    val_cap = int(np.floor(
        x_val.shape[0] / self.config.batch_size) * self.config.batch_size)
    x_train = x_train[0:train_cap, :]
    s_train = s_train[0:train_cap]
    x_train_3d = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val = x_val[0:val_cap, :]
    s_val = s_val[0:val_cap]
    x_val_3d = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

    # Train the vae model
    hist = self.vae.fit(
        x=[x_train, s_train, s_train],
        y=x_train_3d,
        batch_size=self.config.batch_size,
        epochs=epochs,
        validation_data=([x_val, s_val, s_val], x_val_3d))

    # Save outputs
    try:
      os.makedirs(self.config.model_dir)
    except:
      logger.info("Did not make model dir")
    for i in range(3):
      try:
        self.vae.save(self.config.model_dir + "/vae_checkpoint")
      except:
        print("Did not successfully save vae")

    self.encoder.save(self.config.model_dir + "/encoder_checkpoint")
    self.generator.save(self.config.model_dir + "/generator_checkpoint")
    pickle.dump(self.config,
                open(self.config.model_dir + "/model_config.p", "wb"))
    pickle.dump(self.tokenizer,
                open(self.config.model_dir + "/tokenizer.p", "wb"))
    pickle.dump(hist.history,
                open(self.config.model_dir + "/history.p", "wb"))
    logger.info(hist.history)
    return hist