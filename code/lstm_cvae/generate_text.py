# Generate texts from trained models

from __future__ import print_function
import cPickle as pickle
import csv
from datetime import datetime

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences

from data_generator import DataGenerator
from lstm_cvae_model import ModelConfig

def get_sample_config():
    sample_config = {
        "model_dir": "/Users/tongwang/Playground/deepjoke/code/model_checkpoints/lstm_cvae/20170618_072219",
        "starter_sentences": ["a man", "a girl", "a sexy", "what", "why", "i have a dream", "once upon a time"],
        "temperatures": [None, 0.2, 0.5, 1.0, 1.5],
        "scores": [0, 1, 5, 10, 20],
        "num": 5
    }
    return sample_config

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sample(preds, temperature=None):
    """Helper function to sample an index from a probability array; if temperature is None, 
    then sample greedily"""
    if temperature is None:
        return np.argmax(preds)
    else:
        preds = np.asarray(preds).astype('float64')
        preds = softmax(preds)  # Convert logits into probabilities
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def tokens_to_words(tokens, tokenizer, eos=""):
    """Helper function to turn an 1-d array of tokens tokenized by tokenizer back to words"""
    reverse_word_index = {index: word for word, index in tokenizer.word_index.iteritems()}
    reverse_word_index[0] = eos
    words = [reverse_word_index.get(token) for token in tokens]
    text = " ".join(words)
    return text

def generate_text(target_score, generator, model_config, tokenizer,
                  starter_sentence="", temperature=None, eos=""):
    """Function to generate paragraphs given a target score, a random latent vector,
    and (optionally) a starter sentence.
    
    Args:
        -target_score
        -generator
        -model_config
        -tokenizer
        -temperature: if None, generate text greedily; otherwise sample stochastically
    Returns:
        -model_config.batch_size many pieces of text
    """
    # Prepare inputs
    z = np.random.normal(size=(model_config.batch_size, model_config.latent_size))
    scores = np.repeat(target_score, model_config.batch_size)
    cur_sentence = [starter_sentence]
    cur_sequence = tokenizer.texts_to_sequences(cur_sentence)
    cur_sequence = pad_sequences(cur_sequence, maxlen=model_config.max_sequence_length,
                                 padding='post', truncating='post')
    cur_sequence = np.repeat(cur_sequence, model_config.batch_size, axis=0)
    
    reverse_word_index = {index: word for word, index in tokenizer.word_index.iteritems()}

    # Iteratively predict the next word
    while True:
        true_len = len(cur_sequence[0][cur_sequence[0]>0])
        if true_len == model_config.max_sequence_length:
            break
        next_preds = generator.predict([cur_sequence, scores, z])[0, true_len-1, :] # predicted next word
        next_token = sample(next_preds, temperature)
        if next_token == 0:
            break
        cur_sequence[0][true_len] = next_token
    pred_sequence = cur_sequence[0][cur_sequence[0]>0]

    # Translate tokens to words
    pred_text = tokens_to_words(pred_sequence, tokenizer=tokenizer, eos=eos)
    
    return pred_text

def run_all():
    """Function to run ops"""
    np.random.seed(123)
    sample_config = get_sample_config()

    # Load models and configs
    model_dir = sample_config["model_dir"]
    encoder_path = model_dir + "/encoder_checkpoint"
    generator_path = model_dir + "/generator_checkpoint"
    tokenizer_path = model_dir + "/tokenizer.p"
    model_config_path = model_dir + "/model_config.p"
    encoder = load_model(encoder_path)
    generator = load_model(generator_path)
    tokenizer = pickle.load(open(tokenizer_path, "r"))
    model_config = pickle.load(open(model_config_path, "r"))
    generated_text_path = model_dir + "/generated_text_{:%Y%m%d_%H%M%S}.csv".format(datetime.now())

    # Generate text for each config, write to a csv file
    with open(generated_text_path, 'wb') as csvfile:
        fieldnames = ["starter_sentence", "temperature", "score", "iter", "text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()
        cnt = 0
        print("Generating texts...\n")    
        for starter_sentence in sample_config["starter_sentences"]:
            for temperature in sample_config["temperatures"]:
                for score in sample_config["scores"]:
                    for i in range(sample_config["num"]):
                        text = generate_text(generator=generator, model_config=model_config, tokenizer=tokenizer,
                            target_score=score, starter_sentence=starter_sentence, temperature=temperature)
                        new_dict_row = {
                        "starter_sentence": starter_sentence,
                        "temperature": str(temperature),
                        "score": score,
                        "iter": i,
                        "text": text}
                        writer.writerow(new_dict_row)
                        csvfile.flush()
                        cnt += 1
                        print("Generated {} pieces of text".format(cnt))
                        print(new_dict_row)

if __name__ == "__main__":
    run_all()








