from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, LSTM, Flatten, Embedding, Input
from keras.models import Model
from unidecode import unidecode

np.random.seed(123)

# Base params
BASE_DIR = os.getcwd()
GLOVE_DIR = BASE_DIR.replace("/code", "/glove.6B/")
TEXT_DATA_DIR = BASE_DIR.replace("/code", '/joke-dataset/')
MODEL_DIR = BASE_DIR + "/models/{:%Y%m%d_%H%M%S}/".format(datetime.now()) 
try:
	os.mkdir(MODEL_DIR)
except:
	print("Did not make model dir")


# Model params
VALIDATION_SPLIT = 0.2
MAX_SEQUENCE_LENGTH = 300
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 100
LSTM_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5
MAX_NB_EXAMPLES = None  # Sample a fraction of examples to speed up training

# Sampling params
STARTER_SENTENCE = "donald trump walks into a bar"

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
 	# Function to preserve some punctuation marks
    s = s.replace("?", " ?")
    s = s.replace(".", " .")
    s = s.replace(",", " ,")
    return s

# Extract texts and scores
texts = map(combine_title_body, zip(reddit_data["title"].tolist(), reddit_data["body"].tolist()))
texts = [unidecode(text) for text in texts] # Get rid of unicode characters
texts = map(clean_punc, texts) # clean up punctuations 
scores = reddit_data["score"].tolist()
print("Read in {} jokes.".format(len(texts)))
print("Read in {} scores.".format(len(scores)))

# Shrink dataset to speed up training, if needed
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
print('Found %s unique tokens.' % len(word_index))
num_words = min(MAX_NB_WORDS, len(word_index))


data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,
                    padding='post', truncating='post')

l_labels = np.append(data[:,1:], np.zeros((data.shape[0],1)), 
                     axis=1).astype("int32").reshape(data.shape[0], data.shape[1], 1)
s_labels = np.asarray(scores) # labels for the scoring model

print('Shape of data tensor:', data.shape)
print('Shape of language model label tensor:', l_labels.shape)
print('Shape of scoring model label tensor:', s_labels.shape)

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

# Read in Glove vectors
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
preds_l = Dense(num_words + 1, activation='softmax')(x)

def sample_weight_func(scores):
    return scores + 1

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_sentence(model, starter_sentence=STARTER_SENTENCE,
                      diversities=[0.2, 0.5, 1.0, 1.2]):
    # function to generate sentences from trained lstm
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        cur_sentence = [starter_sentence]
        cur_sequence = tokenizer.texts_to_sequences(cur_sentence)
        cur_sequence = pad_sequences(cur_sequence, maxlen=MAX_SEQUENCE_LENGTH,
                    padding='post', truncating='post')
        print()
        print('----- diversity:', diversity)
        print('----- Generating with seed: "' + cur_sentence[0] + '"')
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
        print(pred_sentence)
        print()

# Train model
print('Training model.')
l_model = Model(sequence_input, preds_l)
l_model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam')

for epoch in range(EPOCHS):
	l_model.fit(x_train, y_train_l,
          batch_size=BATCH_SIZE,
          epochs=1,
          validation_data=(x_val, y_val_l))
	l_model.save(MODEL_DIR + "checkpoint_epoch_{}".format(epoch))
	try:
		generate_sentence(l_model)
	except:
		print("Error generating sentence")


