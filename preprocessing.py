import string
import pickle
import os.path
import time
import torch

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from torch import cuda
from torch.utils.data import DataLoader
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from transformers import BertTokenizer, TFBertForSequenceClassification

from bertModel import BertModel
from utils import cleanString, splitDataframe, wordToSeq, toCategorical, CustomDataset, CustomDatasetWithSoftTargets, \
    formatTime


def hanPreprocessing(dataset_name, data_df, save_all=False, cleaned=False, MAX_FEATURES=200000, MAX_SENTENCE_NUM=40,
                  MAX_WORD_NUM=50, EMBED_SIZE=100):
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
    if not cleaned:
        reviews = []
        stop_words = set(stopwords.words('english'))
        data_cleaned = data_df.copy()

        n = data_df['text'].shape[0]
        col = data_df.columns.get_loc('text')
        for i in range(n):
            reviews.append(cleanString(data_df.iloc[i, col], stop_words))

        # We copy our clean reviews in data_cleaned pandas dataframe
        data_cleaned.loc[:, 'text'] = pd.Series(reviews, index=data_df.index)
    else:
        data_cleaned = data_df

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


def bertPreprocessing(dataset_name, data_df, MAX_LEN=128, save_all=True):
    """
    Dataset preparation for Bert Model. It is splitted (0.8 train, 0.1 valid and 0.1 test) and sets are returned. Every
    set is a CustomDataset class (see utils.py) that return data in Bert format.
    :param dataset_name: string of dataset name.
    :param data_df: dataset in dataframe pandas format.
    :param MAX_LEN: it represents total words represented in bert encoding (other words will be ignored).
    :param save_all: boolean that specifies if save all data for time saving before training or network evaluating.
    :return: training_set, validation_set, test_set in CustomDataset format.
    """

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_size = 0.8
    train_dataset = data_df.sample(frac=train_size, random_state=200).reset_index(drop=True)
    tmp_dataset = data_df.drop(train_dataset.index).reset_index(drop=True)
    test_dataset = tmp_dataset.sample(frac=0.5, random_state=200).reset_index(drop=True)
    val_dataset = tmp_dataset.drop(test_dataset.index).reset_index(drop=True)

    print("FULL Dataset: {}".format(data_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))
    print("VALID Dataset: {}".format(val_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    validation_set = CustomDataset(val_dataset, tokenizer, MAX_LEN)
    test_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

    if save_all is True:
        os.makedirs(os.path.dirname('datasets/' + dataset_name + '_bert_cleaned.txt'), exist_ok=True)
        with open('datasets/' + dataset_name + '_bert_cleaned.txt', 'wb') as f:
            pickle.dump([training_set, validation_set, test_set, MAX_LEN], f)

    return training_set, validation_set, test_set


def kdPreprocessing(dataset_name, n_classes, data_df, teacher_path, MAX_LEN=128, save_all=True, isCheckpoint=False):
    """
    Dataset preparation for Bert Model and KD models. It is splitted (0.8 train, 0.1 valid and 0.1 test) and sets are
    returned. Every set is a CustomDatasetWithSoftTargets class (see utils.py) that return data in Bert format.
    :param dataset_name: string of dataset name.
    :param data_df: dataset in dataframe pandas format.
    :param MAX_LEN: it represents total words represented in bert encoding (other words will be ignored).
    :param save_all: boolean that specifies if save all data for time saving before training or network evaluating.
    :return: training_set, validation_set, test_set in CustomDataset format.
    """
    device = 'cuda' if cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    data_df['soft_targets'] = 0

    train_size = 0.8
    train_dataset = data_df.sample(frac=train_size, random_state=200).reset_index(drop=True)
    tmp_dataset = data_df.drop(train_dataset.index).reset_index(drop=True)
    test_dataset = tmp_dataset.sample(frac=0.5, random_state=200).reset_index(drop=True)
    val_dataset = tmp_dataset.drop(test_dataset.index).reset_index(drop=True)

    print("FULL Dataset: {}".format(data_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))
    print("VALID Dataset: {}".format(val_dataset.shape))

    training_set = CustomDatasetWithSoftTargets(train_dataset, tokenizer, MAX_LEN)
    validation_set = CustomDatasetWithSoftTargets(val_dataset, tokenizer, MAX_LEN)
    test_set = CustomDatasetWithSoftTargets(test_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': 32,
                    'shuffle': False,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)

    teacher_model = BertModel(n_classes=n_classes, dropout=0.3)
    if isCheckpoint:
        teacher_model.load_state_dict(torch.load(teacher_path)['model_state_dict'])
    else:
        teacher_model.load_state_dict(torch.load(teacher_path))

    print(teacher_model)
    total_params = sum(p.numel() for p in teacher_model.parameters())
    print('Teacher total parameters: {:}'.format(total_params))

    teacher_model.to(device)
    teacher_model.eval()

    soft_targets = []
    t0 = time.time()

    print('Creating soft targets...')
    with torch.no_grad():
        for step, batch in enumerate(training_loader):
            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)

            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = formatTime(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.   Elapsed: {:}.'.format(step, len(training_loader), elapsed))

            soft_target = torch.softmax(teacher_model(ids, mask, token_type_ids), dim=1)
            soft_targets.extend(soft_target.cpu().detach().numpy().tolist())

    del teacher_model

    training_set.setSoftTargets(soft_targets)

    if save_all is True:
        os.makedirs(os.path.dirname('datasets/' + dataset_name + '_kd_cleaned.txt'), exist_ok=True)
        with open('datasets/' + dataset_name + '_kd_cleaned.txt', 'wb') as f:
            pickle.dump([training_set, validation_set, test_set, MAX_LEN], f)

    return training_set, validation_set, test_set


if __name__ == '__main__':

    '''
    dataset_name = 'imdb_reviews'
    ds = tfds.load(dataset_name, split='train')
    reviews = []
    for element in ds.as_numpy_iterator():
        reviews.append((element['text'].decode('utf-8'), element['label']))

    data_df = pd.DataFrame(data=reviews, columns=['text', 'label'])
    

    MAX_FEATURES = 200000  # maximum number of unique words that should be included in the tokenized word index
    MAX_SENTENCE_NUM = 20  # maximum number of sentences in one document
    MAX_WORD_NUM = 40  # maximum number of words in each sentence
    EMBED_SIZE = 100  # vector size of word embedding
    BATCH_SIZE = 64
    NUM_EPOCHS = 60

    dataset_name = 'IMDB'
    train_df = pd.read_csv('datasets/' + dataset_name + '/train.tsv', sep='\t')
    train_df.columns = ['label', 'text']
    test_df = pd.read_csv('datasets/' + dataset_name + '/test.tsv', sep='\t')
    test_df.columns = ['label', 'text']
    dev_df = pd.read_csv('datasets/' + dataset_name + '/dev.tsv', sep='\t')
    dev_df.columns = ['label', 'text']
    data_df = pd.concat([train_df, test_df, dev_df], ignore_index=True)
    data_df['label'] = data_df['label'].apply(lambda x: len(str(x)) - 1)
    print(data_df)
    '''

    dataset_name = "yelp_2014"
    data_df = pd.read_csv("datasets/" + dataset_name + ".csv")
    data_df = data_df[['label', 'text']]
    for index, row in data_df.iterrows():
        try:
            row['label'] = int(float(row['label'])) - 1
        except:
            row['label'] = 0

    bertPreprocessing(dataset_name=dataset_name, data_df=data_df, save_all=True, MAX_LEN=128)
    #preprocessing(dataset_name=dataset_name, data_df=data_df, save_all=True, MAX_SENTENCE_NUM=20, MAX_WORD_NUM=40, EMBED_SIZE=100)