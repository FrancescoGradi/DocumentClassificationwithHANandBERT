import string
import pickle
import os.path

import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from nltk.corpus import stopwords

from utils import cleanString, splitDataframe, wordToSeq, toCategorical


def preprocessing(dataset_name, data_cleaned, save_all=False, MAX_FEATURES=200000, MAX_SENTENCE_NUM=40, MAX_WORD_NUM=50,
                  EMBED_SIZE=100):
    '''
    :param dataset_name: a string that represents the name of the dataset (it used to save some stuff).
    :param data_df: dataset in DataFrame Pandas format, with two columns: 'text' and 'label'.
    :param MAX_FEATURES: maximum number of unique words that should be included in the tokenized word index
    :param MAX_SENTENCE_NUM: maximum number of sentences in one document
    :param MAX_WORD_NUM: maximum number of words in each sentence
    :param EMBED_SIZE: vector size of word embedding
    :return: train, validation and test cleaned and ready for the network. Also it returns embedding_matrix (weights for
    the network), word_index and n_classes in dataset.
    '''

    # Cleaning text (no uppercase words), removing stopwords
    '''
    reviews = []
    stop_words = set(stopwords.words('english'))
    data_cleaned = data_df.copy()

    n = data_df['text'].shape[0]
    col = data_df.columns.get_loc('text')
    for i in range(n):
        reviews.append(cleanString(data_df.iloc[i, col], stop_words))

    # We copy our clean reviews in data_cleaned pandas dataframe

    data_cleaned.loc[:, 'text'] = pd.Series(reviews, index=data_df.index)

    '''
    data_cleaned = data_cleaned[["label", "text"]]
    data_cleaned.loc[:, 'label'] = pd.Categorical(data_cleaned.label)
    print(data_cleaned)

    # Adding a normalized code from 0 to len(label) - 1. We create a dict {label: code}.

    data_cleaned['code'] = data_cleaned.label.cat.codes
    categoryToCode = dict(enumerate(data_cleaned['label'].cat.categories))
    n_classes = len(categoryToCode)

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

    # We split data_cleaned pandas dataframe with code column as y

    train, validation, test = splitDataframe(data_cleaned, 'code', 0.8, 0.1, 0.1)

    # Every text is converted to a numeric sequence (a numpy matrix with dimension MAX_SENTENCE_NUM x MAX_WORD_NUM)
    # thanks to word_index just created. Every matrix is added to a list and converted in a numpy array of matrices.

    # Training
    sequences = []
    for i in range(train['text'].shape[0]):
        sequences.append(wordToSeq(train['text'].iloc[i], word_index, MAX_SENTENCE_NUM, MAX_WORD_NUM, MAX_FEATURES))
    x_train = np.array(sequences)
    y_train = toCategorical(train['code'], categoryToCode)

    # Validation
    sequences = []
    for i in range(validation['text'].shape[0]):
        sequences.append(wordToSeq(validation['text'].iloc[i], word_index, MAX_SENTENCE_NUM, MAX_WORD_NUM, MAX_FEATURES))
    x_val = np.array(sequences)
    y_val = toCategorical(validation['code'], categoryToCode)

    # Test
    sequences = []
    for i in range(test['text'].shape[0]):
        sequences.append(wordToSeq(test['text'].iloc[i], word_index, MAX_SENTENCE_NUM, MAX_WORD_NUM, MAX_FEATURES))
    x_test = np.array(sequences)
    y_test = toCategorical(test['code'], categoryToCode)

    if save_all is True:
        os.makedirs(os.path.dirname('datasets/' + dataset_name + '_cleaned.txt'), exist_ok=True)
        with open('datasets/' + dataset_name + '_cleaned.txt', 'wb') as f:
            pickle.dump([x_train, y_train, x_val, y_val, x_test, y_test, embedding_matrix, word_index, n_classes], f)


    return x_train, y_train, x_val, y_val, x_test, y_test, embedding_matrix, word_index, n_classes