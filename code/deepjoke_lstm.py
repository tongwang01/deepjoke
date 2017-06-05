from __future__ import print_function
import os
import sys
import logging
from datetime import datetime
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, LSTM, Flatten, Embedding, Input
from keras.models import Model, load_model
from unidecode import unidecode

np.random.seed(123)

# Base params
BASE_DIR = os.getcwd()
GLOVE_DIR = BASE_DIR.replace("/code", "/glove.6B/")
TEXT_DATA_DIR = BASE_DIR.replace("/code", '/joke-dataset/')
MODEL_DIR = BASE_DIR + "/models/lstm/{:%Y%m%d_%H%M%S}/".format(datetime.now())
LOAD_MODEL_DIR = None 

# Model params
VALIDATION_SPLIT = 0.2
MAX_SEQUENCE_LENGTH = 300
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
LSTM_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
MAX_NB_EXAMPLES = None # Sample a fraction of examples to speed up training
TRAIN_SCORE_THRESHOLD = 5
NB_SHARDS = 3

# Sampling params
STARTER_SENTENCES = [
    "a man walks into a bar", 
    "my girlfriend told me to take the spider out instead of killing it",
    "a lot of women turn into great drivers"]

try:
    os.makedirs(MODEL_DIR)
except:
    print("Did not make model dir")

# Logging
logger = logging.getLogger("deepjoke_lstm")
logger.setLevel(logging.INFO)
logging.basicConfig(filename = (MODEL_DIR + 'model_log.log'), level=logging.INFO)

# Log model params
logger.info("VALIDATION_SPLIT = {} \
    MAX_SEQUENCE_LENGTH = {} \
    MAX_NB_WORDS = {} \
    EMBEDDING_DIM = {} \
    LSTM_SIZE = {} \
    BATCH_SIZE = {} \
    EPOCHS = {} \
    MAX_NB_EXAMPLES = {} \
    TRAIN_SCORE_THRESHOLD = {} \
    NB_SHARDS = {}".format(VALIDATION_SPLIT, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, EMBEDDING_DIM, 
    LSTM_SIZE, BATCH_SIZE, EPOCHS, MAX_NB_EXAMPLES, TRAIN_SCORE_THRESHOLD, NB_SHARDS))

# Read in data
reddit_data = pd.read_json(TEXT_DATA_DIR + "reddit_jokes.json", encoding='utf-8')

# Function to combine title and body when appropriate
def combine_title_body((title, body), verify_chars=15):
    """Given title and body:
    - discard title if the first verify_chars chars of title is the same as that of body
    - otherwise add title to body"""
    title_lower = title.lower()
    body_lower = body.lower()
    if title_lower[0:verify_chars] == body_lower[0:verify_chars]:
        combined = body
    else:
        combined = title + " " + body
    return combined

def clean_punc(s):
    # Function to clean up punctuations
    s = s.replace("...", " ")
    s = s.replace("..", " ")
    s = s.replace("?", " ? ")
    s = s.replace(".", " . ")
    s = s.replace(",", " , ")
    return s

# Extract texts and scores
texts = map(combine_title_body, zip(reddit_data["title"].tolist(), reddit_data["body"].tolist()))
texts = [unidecode(text) for text in texts] # Get rid of unicode characters
texts = map(clean_punc, texts) # clean up punctuations 
scores = reddit_data["score"].tolist()
logger.info("Read in {} jokes.".format(len(texts)))
logger.info("Read in {} scores.".format(len(scores)))
print("Read in {} jokes.".format(len(texts)))
print("Read in {} scores.".format(len(scores)))


# Shrink dataset to speed up training, if needed
if TRAIN_SCORE_THRESHOLD > 0:
    idx = [i for i in range(len(scores)) if scores[i]>=TRAIN_SCORE_THRESHOLD]
    texts = [texts[i] for i in idx]
    scores = [scores[i] for i in idx]
if MAX_NB_EXAMPLES is not None:
    nb_examples = min(MAX_NB_EXAMPLES, len(texts))
    texts = texts[:nb_examples]
    scores = scores[:nb_examples]

# Tokenzie texts and labels
filters = '!"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n'
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters=filters)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
reserse_word_index = {index: word for word, index in word_index.iteritems()}
logger.info('Found %s unique tokens.' % len(word_index));
print('Found %s unique tokens.' % len(word_index))
num_words = min(MAX_NB_WORDS, len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,
                    padding='post', truncating='post')

l_labels = np.append(data[:,1:], np.zeros((data.shape[0],1)), 
                     axis=1).astype("int32").reshape(data.shape[0], data.shape[1], 1)
s_labels = np.asarray(scores) # labels for the scoring model

logger.info('Shape of data tensor: {}'.format(data.shape))
logger.info('Shape of language model label tensor: {}'.format(l_labels.shape))
logger.info('Shape of scoring model label tensor: {}'.format(s_labels.shape))
print('Shape of data tensor: {}'.format(data.shape))
print('Shape of language model label tensor: {}'.format(l_labels.shape))
print('Shape of scoring model label tensor: {}'.format(s_labels.shape))


# Split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
l_labels = l_labels[indices]
s_labels = s_labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train_l = l_labels[:-num_validation_samples]
y_train_s = s_labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val_l = l_labels[-num_validation_samples:]
y_val_s = s_labels[-num_validation_samples:]

def sample_weight_func(scores):
    return scores + 1

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = softmax(preds)  # Convert logits into probabilities
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_sentence(model, starter_sentence,
                      diversities=[0.2, 0.5, 1.0, 1.2, 2.0]):
    # function to generate sentences from trained lstm
    for diversity in diversities:
        cur_sentence = [starter_sentence]
        cur_sequence = tokenizer.texts_to_sequences(cur_sentence)
        cur_sequence = pad_sequences(cur_sequence, maxlen=MAX_SEQUENCE_LENGTH,
                    padding='post', truncating='post')
        logger.info(' '); print()
        logger.info('----- diversity: {}'.format(diversity)); print('----- diversity: {}'.format(diversity))
        logger.info('----- Generating with seed: "{}"'.format(cur_sentence[0]))
        print('----- Generating with seed: "{}"'.format(cur_sentence[0]))
        logger.info(' ')
        print()
        
        while True:
            true_len = len(cur_sequence[0][cur_sequence[0]>0])
            if true_len == MAX_SEQUENCE_LENGTH:
                break
            next_preds = model.predict(cur_sequence, verbose=0)[0, true_len-1, :] # predicted next word
            next_token = sample(next_preds, diversity)
            if next_token == 0:
                break
            cur_sequence[0][true_len] = next_token
        
        pred_sequence = cur_sequence[0][cur_sequence[0]>0]
        pred_sentence = [reserse_word_index[pred_sequence[i]] for i in range(len(pred_sequence))]
        pred_sentence = " ".join(pred_sentence)
        logger.info(pred_sentence)
        logger.info(' ')
        print(pred_sentence)
        print(' ')   

# Read in Glove vectors
logger.info('Indexing word vectors.')
print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# Prepare embedding matrix
embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Load embedding in Embedding layer
embedding_layer = Embedding(num_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# Build language model
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = LSTM(LSTM_SIZE, return_sequences=True)(embedded_sequences)
preds_l = Dense(num_words + 1, activation=None)(x) # Generate logits only 
    
# Build new model or load existing model
def tf_sparse_categorical_crossentropy(y_true, y_pred):
    return tf.contrib.keras.backend.sparse_categorical_crossentropy(
        output=y_pred, target=y_true, from_logits=True)

logger.info('Training model.'); print('Training model.')
if LOAD_MODEL_DIR is None:
    l_model = Model(sequence_input, preds_l)
    l_model.compile(loss=tf_sparse_categorical_crossentropy, optimizer='adam')
else:
    l_model = load_model(LOAD_MODEL_DIR, 
        custom_objects={'tf_sparse_categorical_crossentropy': tf_sparse_categorical_crossentropy})
    logger.info("Loaded model from {}".format(LOAD_MODEL_DIR))
    print("Loaded model from {}".format(LOAD_MODEL_DIR))

# Split training data into shards to get more frequent feedback
examples_per_shard = int(x_train.shape[0] / (NB_SHARDS-1))

for epoch in range(EPOCHS):
    start_time = time.time()
    logger.info(' ')
    print('')
    logger.info("Grand epoch {}".format(epoch))
    print("Grand epoch {}".format(epoch))
    logger.info(' ')
    print('')
    for shard in range(NB_SHARDS):
        logger.info("Training on shard {}/{}".format(shard, NB_SHARDS))
        print("Training on shard {}/{}".format(shard, NB_SHARDS))
        if shard != NB_SHARDS - 1:
            x_train_now = x_train[shard * examples_per_shard: (shard + 1) * examples_per_shard]
            y_train_l_now = y_train_l[shard * examples_per_shard: (shard + 1) * examples_per_shard]
            y_train_s_now = y_train_s[shard * examples_per_shard: (shard + 1) * examples_per_shard]
        else:
            x_train_now = x_train[(shard - 1) * examples_per_shard: ]
            y_train_l_now = y_train_l[(shard - 1) * examples_per_shard: ]
            y_train_s_now = y_train_s[(shard - 1) * examples_per_shard: ]

        hist = l_model.fit(x_train_now, y_train_l_now,
            batch_size=BATCH_SIZE,
            sample_weight = sample_weight_func(y_train_s_now),
            epochs=1,
            validation_data=(x_val, y_val_l))
        l_model.save(MODEL_DIR + "checkpoint_latest")
        logger.info(hist.history)
        print(hist.history)
        try:
            for starter_sentence in STARTER_SENTENCES:
                generate_sentence(l_model, starter_sentence=starter_sentence)
        except:
            logger.info("Error generating sentence")
            print("Error generating sentence")
    logger.info("Epoch Took {} seconds\n".format(time.time() - start_time))
    print("Epoch Took {} seconds\n".format(time.time() - start_time))


