# Module to train LSTM CVAE model

from __future__ import print_function
import os
from datetime import datetime
import logging

import numpy as np

from data_generator import DataGenerator
from lstm_cvae_model import ModelConfig, UncondDecodeLstmCvae, CondDecodeLstmCvae

np.random.seed(123)

def train():
    """Function to train lstm cvae model """
    # Prepare ModelConfig
    base_dir = os.getcwd().replace("/lstm_cvae", "")
    glove_dir = base_dir.replace("/code", "/glove.6B/")
    embedding_path=os.path.join(glove_dir, 'glove.6B.100d.txt')
    text_data_dir = os.path.join(base_dir.replace("/code", '/joke-dataset/'), "reddit_jokes.json")
    model_dir = base_dir + "/model_checkpoints/lstm_cvae/{:%Y%m%d_%H%M%S}".format(datetime.now())

    model_config = ModelConfig(input_path=text_data_dir,
                           embedding_path=embedding_path,
                           model_dir=model_dir,
                           lstm_size_encoder=256,
                           lstm_size_decoder=256,
                           intermediate_size=128,
                           latent_size=64,
                           max_nb_examples=500,
                           min_score=0,
                           kl_weight=100,
                           score_transform="log",
                           epochs=5,
                           batch_size=32)

    # Set up logging
    try:
        os.makedirs(model_config.model_dir)
    except:
        pass
    logger = logging.getLogger("lstm_cvae")
    logger.setLevel(logging.INFO)
    logging.basicConfig(filename=(model_config.model_dir + '/model_log.log'), 
        level=logging.INFO)

    # Generate data
    data_generator = DataGenerator(
        input_path=model_config.input_path,
        min_score=model_config.min_score, 
        max_nb_words=model_config.max_nb_words,
        max_nb_examples=model_config.max_nb_examples,
        max_sequence_length=model_config.max_sequence_length,
        validation_split=model_config.validation_split)
    x_train, y_l_train, y_s_train, x_val, y_l_val, y_s_val, tokenizer, _ = data_generator.generate()

    def log_transform(x):
        return np.log(x + np.e)

    if model_config.score_transform == "log":
        y_s_train = log_transform(y_s_train)
        y_s_val = log_transform(y_s_val)

    # Build and fit model
    cvae = CondDecodeLstmCvae(model_config, tokenizer)
    cvae.build()
    hist = cvae.fit(x_train, y_s_train, y_l_train, x_val, y_s_val, y_l_val)
    logger.info(hist.history)
    print("Done.")

if __name__ == "__main__":
    train()
 



    
