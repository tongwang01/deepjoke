# Module to sample frmo trained LSTM CVAE model

from __future__ import print_function

from keras import backend as K
from keras.models import Model, load_model
import cPickle as pickle

# Load model and tokenizer
def load_model():
    MODEL_DIR = ""
    encoder_path = MODEL_DIR + "/encoder_checkpoint"
    generator_path = MODEL_DIR + "/generator_checkpoint"

    encoder = load_model(encoder_path)
    generator = load_model(generator_path)

def sample_cvae(generator):
    