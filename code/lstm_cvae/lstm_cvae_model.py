#TODO
#-Split the KL loss term and Recon loss term
#-Pickle model_config intead of json it

from __future__ import print_function
import os
from datetime import datetime
import cPickle as pickle
import json

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import Dense, LSTM, Embedding, Input, RepeatVector, Lambda
from keras.models import Model, load_model
from keras.layers.merge import concatenate
from keras import optimizers
from data_generator import DataGenerator

class ModelConfig():
    """Class to hold all model configs; required to instantiate a LSTM CVAE model"""
    def __init__(self,
        input_path,
        embedding_path,
        model_dir,
        embedding_dim=100,
        batch_size=32,
        epochs=5,
        min_score=0,
        max_nb_words=50000,
        max_nb_examples=None,
        max_sequence_length=300,
        lstm_size_encoder=256,
        lstm_size_decoder=256,
        intermediate_size=128,
        latent_size=64,
        kl_weight=1.,
        optimizer="RMSprop",
        validation_split=0.1):
        self.input_path = input_path
        self.embedding_path = embedding_path
        self.model_dir = model_dir
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.min_score = min_score
        self.max_nb_words = max_nb_words
        self.max_nb_examples = max_nb_examples
        self.max_sequence_length = max_sequence_length
        self.lstm_size_encoder = lstm_size_encoder
        self.lstm_size_decoder = lstm_size_decoder
        self.intermediate_size = intermediate_size
        self.latent_size = latent_size
        self.kl_weight = kl_weight
        self.optimizer = optimizer
        self.validation_split = validation_split

class UncondDecodeLstmCvae(object):
    """Class to hold LSTM CVAE model
    This implementation users the repeated code z as the only input to the decoder 
    (i.e. unconditional decoder)
    """
    def __init__(self, model_config, tokenizer):
        self.config = model_config
        self.word_index = tokenizer.word_index
        self.num_words = min(model_config.max_nb_words, 
                                    len(tokenizer.word_index))
        self.tokenizer = tokenizer

    def load_embedding(self):
        """Load and prepare embedding matrix"""
        embeddings_index = {}
        f = open(self.config.embedding_path)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        embedding_matrix = np.zeros((self.num_words + 1, self.config.embedding_dim))
        for word, i in self.word_index.items():
            if i >= self.config.max_nb_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
    
    def build(self):
        """Construct lstm cvae model"""
        # Load embedding in Embedding layer
        embedding_matrix = self.load_embedding()
        embedding_layer = Embedding(self.num_words + 1,
                                    self.config.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.config.max_sequence_length,
                                    trainable=False)
        
        # Q(z|X,y) -- encoder
        # embedded sequence input
        sequence_inputs = Input(batch_shape=(self.config.batch_size, self.config.max_sequence_length), dtype='int32')
        embedded_inputs = embedding_layer(sequence_inputs)
        x = LSTM(self.config.lstm_size_encoder, return_sequences=False)(embedded_inputs)
        score_inputs = Input(batch_shape=(self.config.batch_size, 1))
        x_joint = concatenate([x, score_inputs], axis=1)
        x_encoded = Dense(self.config.intermediate_size, activation='tanh')(x_joint)
        z_mean = Dense(self.config.latent_size)(x_encoded)
        z_log_sigma = Dense(self.config.latent_size)(x_encoded)

        # Sample z ~ Q(z|X,y)
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(self.config.batch_size, self.config.latent_size), 
                                       mean=0., stddev=1.)            
            return z_mean + K.exp(z_log_sigma/2.) * epsilon
        
        z = Lambda(sampling)([z_mean, z_log_sigma])
        z_cond = concatenate([z, score_inputs], axis=1)

        # P(X|z,y) -- decoder
        z_repeated = RepeatVector(self.config.max_sequence_length)(z_cond)
        
        decoder_h = LSTM(self.config.lstm_size_decoder, return_sequences=True)
        decoder_out = Dense(self.num_words + 1)
        
        h_decoded = decoder_h(z_repeated)
        x_decoded = decoder_out(h_decoded)
        # Construct three models
        # vae
        vae = Model([sequence_inputs, score_inputs], x_decoded)
        # encoder
        encoder = Model([sequence_inputs, score_inputs], z_mean)
        # generator
        generator_z_inputs = Input(batch_shape=(self.config.batch_size, self.config.latent_size))
        generator_z_cond = concatenate([generator_z_inputs, score_inputs], axis=1)
        generator_z_repeated = RepeatVector(self.config.max_sequence_length)(generator_z_cond)
        generator_h_decoded = decoder_h(generator_z_repeated)
        generator_x_decoded = decoder_out(generator_h_decoded)
        generator = Model([generator_z_inputs, score_inputs], generator_x_decoded)

        kl_weight = self.config.kl_weight
        def recon_loss(y_true, y_pred):
            """E[log P(X|z,y)]"""
            recon = K.mean(K.sparse_categorical_crossentropy(
                output=y_pred, target=y_true, from_logits=True), axis=1)
            return recon

        def kl_loss(y_true, y_pred):
            """D_KL(Q(z|X,y) || P(z|X)); calculate in closed form as both dist. are Gaussian"""
            kl = 0.5 * K.mean(K.exp(z_log_sigma) + K.square(z_mean) - 1. - z_log_sigma, axis=1)
            kl = kl * kl_weight            
            return kl

        def vae_loss(y_true, y_pred):
            """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
            recon = recon_loss(y_true, y_pred)
            kl = kl_loss(y_true, y_pred)
            return recon + kl
        
        vae.compile(loss=vae_loss, optimizer=self.config.optimizer, metrics=[recon_loss, kl_loss])
        
        self.vae = vae
        self.encoder = encoder
        self.generator = generator
        
    def fit(self, x_train, y_s_train, x_val, y_s_val):
        """Fit vae model, and store ouput models, configs and associated tokenizer"""
        # Cut training and validation sets to multiples of batch_size
        train_cap = int(np.floor(x_train.shape[0] / self.config.batch_size) * self.config.batch_size)
        val_cap = int(np.floor(x_val.shape[0] / self.config.batch_size) * self.config.batch_size)
        x_train = x_train[0:train_cap, :]
        y_s_train = y_s_train[0:train_cap]
        x_val = x_val[0:val_cap, :]
        y_s_val = y_s_val[0:val_cap]
        # Reshape a version of x as targets
        x_train_reshaped = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_val_reshaped = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
        # Fit 
        hist = self.vae.fit([x_train, y_s_train], x_train_reshaped,
                     batch_size=self.config.batch_size,
                     epochs=self.config.epochs,
                     validation_data=([x_val, y_s_val], x_val_reshaped))
        # Save outputs
        try:
            os.makedirs(self.config.model_dir)
        except:
            print("Did not make model dir")
        
        for i in range(3):        
            try:
                self.vae.save(self.config.model_dir + "/vae_checkpoint")
            except:
                print("Did not successfully save vae")

        self.encoder.save(self.config.model_dir + "/encoder_checkpoint")
        self.generator.save(self.config.model_dir + "/generator_checkpoint")
        pickle.dump(self.config, open(self.config.model_dir + "/model_config.p", "wb" ))
        pickle.dump(self.tokenizer, open(self.config.model_dir + "/tokenizer.p", "wb" ))
        return hist

def test_UncondDecodeLstmCvae():
    """Function to test UncondDecodeLstmCvae class"""
    # Construct an ModelConfig object
    BASE_DIR = os.getcwd().replace("/lstm_cvae", "")
    GLOVE_DIR = BASE_DIR.replace("/code", "/glove.6B/")
    EMBEDDING_PATH=os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')
    TEXT_DATA_DIR = os.path.join(BASE_DIR.replace("/code", '/joke-dataset/'), "reddit_jokes.json")
    MODEL_DIR = BASE_DIR + "/model_checkpoints/lstm_cvae/test/{:%Y%m%d_%H%M%S}".format(datetime.now())
    model_config = ModelConfig(input_path=TEXT_DATA_DIR,
                           embedding_path=EMBEDDING_PATH,
                           model_dir=MODEL_DIR,
                           max_nb_examples=500,
                           epochs=2,
                           batch_size=32)

    # Generate data
    data_generator = DataGenerator(
        input_path=model_config.input_path,
        min_score=model_config.min_score, 
        max_nb_words=model_config.max_nb_words,
        max_nb_examples=model_config.max_nb_examples,
        max_sequence_length=model_config.max_sequence_length,
        validation_split=model_config.validation_split)
    x_train, y_l_train, y_s_train, x_val, y_l_val, y_s_val, tokenizer = data_generator.generate()

    # Build and fit model
    cvae = UncondDecodeLstmCvae(model_config, tokenizer)
    cvae.build()
    cvae.fit(x_train, y_s_train, x_val, y_s_val)
    print("Test passed! Go check model_dir for outputs")


if __name__ == "__main__":
    test_UncondDecodeLstmCvae()
