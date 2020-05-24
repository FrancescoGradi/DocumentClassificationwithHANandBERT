import string
import pickle
import os.path

import pandas as pd
import numpy as np
import tensorflow_datasets as tfds

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from nltk.corpus import stopwords

from utils import cleanString, splitDataframe, wordToSeq, toCategorical


MAX_FEATURES = 200000  # maximum number of unique words that should be included in the tokenized word index
MAX_SENTENCE_NUM = 40  # maximum number of sentences in one document
MAX_WORD_NUM = 50  # maximum number of words in each sentence
EMBED_SIZE = 100  # vector size of word embedding


if __name__ == '__main__':
    '''
    # Reading JSON dataset with Pandas
    
    dataset_name = "imdb_complete"
    data_df = pd.read_json(dataset_name + "json")
    data_df = data_df[["rating", "review"]]
    data_df.columns = ["label", "text"]
    '''

    dataset_name = 'imdb_reviews'
    ds = tfds.load(dataset_name, split='train')
    reviews = []
    for element in ds.as_numpy_iterator():
        reviews.append((element['text'].decode('utf-8'), element['label']))

    data_df = pd.DataFrame(data=reviews, columns=['text', 'label'])


    # Cleaning text (no uppercase words), removing stopwords

    reviews = []
    stop_words = set(stopwords.words('english'))
    data_cleaned = data_df.copy()

    n = data_df['text'].shape[0]
    col = data_df.columns.get_loc('text')
    for i in range(n):
        reviews.append(cleanString(data_df.iloc[i, col], stop_words))

    # We copy our clean reviews in data_cleaned pandas dataframe

    data_cleaned.loc[:, 'text'] = pd.Series(reviews, index=data_df.index)
    data_cleaned.loc[:, 'label'] = pd.Categorical(data_cleaned.label)

    # Adding a normalized code from 0 to len(label) - 1

    data_cleaned['code'] = data_cleaned.label.cat.codes
    categoryToCode = dict(enumerate(data_cleaned['label'].cat.categories))

    print(data_cleaned)

    # We construct a word index that associates a word to a integer number and it is saved

    if os.path.isfile('indices/word_index_' + dataset_name + '.txt'):
        with open('indices/word_index_' + dataset_name + '.txt', 'rb') as f:
            word_index = pickle.load(f)
    else:
        texts = []
        n = data_cleaned['text'].shape[0]
        for i in range(n):
            s = data_cleaned['text'].iloc[i]
            s = ' '.join([word.strip(string.punctuation) for word in s.split() if word.strip(string.punctuation) is not ""])
            texts.append(s)
        tokenizer = Tokenizer(num_words=MAX_FEATURES, lower=True, oov_token=None)
        tokenizer.fit_on_texts(texts)
        word_index = tokenizer.word_index
        os.makedirs(os.path.dirname('indices/word_index_' + dataset_name + '.txt'), exist_ok=True)
        with open('indices/word_index_' + dataset_name + '.txt', 'wb') as f:
            pickle.dump(word_index, f)

    # We read a pre-trained dataset (Glove) that contains a words list: every word is associated with a numeric vector
    # of dim 100 (in this implementation). It is downloadable at https://github.com/stanfordnlp/GloVe

    embeddings_index = {}
    with open(os.path.join(os.getcwd(), 'glove.6B.100d.txt'), encoding='UTF-8') as f:
        for line in f:
            values = line.split()
            embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

    # We create an embedding matrix for the network

    embedding_matrix = np.zeros((len(word_index) + 1, EMBED_SIZE))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


    train, validation, test = splitDataframe(data_cleaned, 'code', 0.8, 0.1, 0.1)

    # Training
    paras = []
    for i in range(train['text'].shape[0]):
        sequence = wordToSeq(train['text'].iloc[i], word_index, MAX_SENTENCE_NUM, MAX_WORD_NUM, MAX_FEATURES)
        paras.append(sequence)
    x_train = np.array(paras)
    y_train = toCategorical(train['code'], categoryToCode)

    # Validation
    paras = []
    for i in range(validation['text'].shape[0]):
        sequence = wordToSeq(validation['text'].iloc[i], word_index, MAX_SENTENCE_NUM, MAX_WORD_NUM, MAX_FEATURES)
        paras.append(sequence)
    x_val = np.array(paras)
    y_val = toCategorical(validation['code'], categoryToCode)

    # Test
    paras = []
    for i in range(test['text'].shape[0]):
        sequence = wordToSeq(test['text'].iloc[i], word_index, MAX_SENTENCE_NUM, MAX_WORD_NUM, MAX_FEATURES)
        paras.append(sequence)
    x_test = np.array(paras)
    y_test = toCategorical(test['code'], categoryToCode)