from __future__ import print_function

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from unidecode import unidecode

np.random.seed(123)


class DataGenerator(object):
    """Class to generate data for deepjoke models
    
    Main method: generate()
    Returns:
        -x_train
        -y_train_l
        -y_train_s
        -x_val
        -y_val_l
        -y_val_s
        -tokenizer
    """
    def __init__(self,
                 input_path,
                 min_score=0, 
                 max_nb_words=50000,
                 max_nb_examples=None,
                 max_sequence_length=300,
                 validation_split=0.1):
        self.input_path = input_path
        self.min_score = min_score
        self.max_nb_words = max_nb_words
        self.max_nb_examples = max_nb_examples
        self.max_sequence_length = max_sequence_length
        self.validation_split = validation_split

    def read_and_clean(self):
        """Read in and clean reddit jokes data
        
        Returns:
            -texts: list of joke text strings
            -scores: list of correpsonding upvote scores
        """
        def combine_title_body((title, body), verify_chars=15):
            """Helper function to process input data. Given title and body:
                - discard title if the first verify_chars chars of title is the same as that of body
                - otherwise add title to body
            """
            title_lower = title.lower()
            body_lower = body.lower()
            if title_lower[0:verify_chars] == body_lower[0:verify_chars]:
                combined = body
            else:
                combined = title + " " + body
            return combined
            return input_data
        
        def clean_punc(s):
            """Helper Function to clean up punctuations"""
            s = s.replace("...", " ")
            s = s.replace("..", " ")
            s = s.replace("?", " ? ")
            s = s.replace(".", " . ")
            s = s.replace(",", " , ")
            return s
        
        def extract_text_and_score(data):
            """Extract text and score from pd dataframe"""
            texts = map(combine_title_body, zip(data["title"].tolist(), data["body"].tolist()))
            # Get rid of unicode characters
            texts = [unidecode(text) for text in texts] 
            # Clean up punctuations 
            texts = map(clean_punc, texts) 
            scores = data["score"].tolist()
            return texts, scores
        
        input_data = pd.read_json(self.input_path, encoding='utf-8')
        texts, scores = extract_text_and_score(input_data)
        print("Read in {} jokes.".format(len(texts)))
        print("Read in {} scores.\n".format(len(scores)))
        return texts, scores
    
    def maybe_shrink(self, texts, scores):
        """Shrink data by either minimum score or max number of examples
        to speed up development runs.
        """
        if self.min_score > 0:
            idx = [i for i in range(len(scores)) if scores[i] >= self.min_score]
            texts = [texts[i] for i in idx]
            scores = [scores[i] for i in idx]
        if self.max_nb_examples is not None:
            nb_examples = min(self.max_nb_examples, len(texts))
            texts = texts[:nb_examples]
            scores = scores[:nb_examples]
        return texts, scores
    
    def tokenize(self, texts):
        """Tokenize text strings into integers
        
        Returns:
            -sequences: tokenized texts
            -tokenzier: keras tokenrizer object
        """
        filters = '!"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n'
        tokenizer = Tokenizer(num_words=self.max_nb_words, filters=filters)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index
        num_words = min(self.max_nb_words, len(word_index))
        print('Found {} unique words; using {} unique words\n'.format(len(word_index), num_words))
        return sequences, tokenizer
    
    def pad(self, sequences):
        padded_sequences = pad_sequences(sequences, 
                                        maxlen=self.max_sequence_length,
                                        padding='post', truncating='post')
        # labels for l model
        x = padded_sequences
        # features
        y_l = np.append(padded_sequences[:,1:], np.zeros((padded_sequences.shape[0],1)), 
            axis=1).astype("int32")
        y_l = y_l.reshape(y_l.shape[0], y_l.shape[1], 1)
        return x, y_l
    
    def split(self, x, y_l, y_s):
        """split data into train and test"""
        # Shuffle
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        x = x[idx]
        y_l = y_l[idx]
        y_s = y_s[idx]
        # Split
        num_val = int(self.validation_split * x.shape[0])
        x_train = x[:-num_val]
        y_l_train = y_l[:-num_val]
        y_s_train = y_s[:-num_val]
        x_val = x[-num_val:]
        y_l_val = y_l[-num_val:]
        y_s_val = y_s[-num_val:]
        return x_train, y_l_train, y_s_train, x_val, y_l_val, y_s_val

    def generate(self):
        """Main function to generate data for deepjoke models"""
        texts, scores = self.read_and_clean()
        texts, scores = self.maybe_shrink(texts, scores)
        sequences, tokenizer = self.tokenize(texts)
        x, y_l = self.pad(sequences)
        y_s = np.asarray(scores)
        x_train, y_l_train, y_s_train, x_val, y_l_val, y_s_val = self.split(x, y_l, y_s)
        print('Shape of training features: {}'.format(x_train.shape))
        print('Shape of training language model labels: {}'.format(y_l_train.shape))
        print('Shape of training score labels: {}'.format(y_s_train.shape))
        print('Shape of validation features: {}'.format(x_val.shape))
        print('Shape of validation language model labels: {}'.format(y_l_val.shape))
        print('Shape of validation score labels: {}'.format(y_s_val.shape))        
        return x_train, y_l_train, y_s_train, x_val, y_l_val, y_s_val, tokenizer, texts

                 