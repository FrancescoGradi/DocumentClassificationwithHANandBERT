import pandas as pd
import numpy as np
import ijson
import json
import sty
import torch
import time
import datetime

from nltk import tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.utils import to_categorical
from torch.utils.data import Dataset


def cleanString(text, stop_words):
    """
        Cleans input string using set rules.
        Cleaning rules:         Every word is lemmatized and lowercased. Stopwords and non alpha-numeric words are
                                removed.
                                Each sentence ends with a period.
        Input:   text       - string(in sentence structure)
                 stop_words   - set of strings which should be removed from text
        Output:  returnString - cleaned input string
    """

    lemmatizer = WordNetLemmatizer()
    cleaned_string = ''
    sentence_token = tokenize.sent_tokenize(text)

    for j in range(len(sentence_token)):
        single_sentence = tokenize.word_tokenize(sentence_token[j])
        sentences_filtered = [(i, lemmatizer.lemmatize(w.lower())) for i, w in enumerate(single_sentence)
                              if w.lower() not in stop_words and w.isalnum()]

        word_list = [x[1] for x in sentences_filtered]
        cleaned_string += ' '.join(word_list) + ' . '

    return cleaned_string


def wordAndSentenceCounter(data_df):
    """
    Print some stats useful to choose problem parameters.
    :param data_df: pandas dataframe of dataset, with column named 'text'.
    :return: None
    """
    n_sent = 0
    n_words = 0
    for i in range(data_df.shape[0]):
        sent = tokenize.sent_tokenize(data_df.loc[i, 'text'])
        for satz in sent:
            n_words += len(tokenize.word_tokenize(satz))
        n_sent += len(sent)
    print("Average number of words in each sentence: ", round(n_words / n_sent))
    print("Average number of sentences in each document: ", round(n_sent / data_df.shape[0]))


def splitDataframe(dataframe, column_name, training_split=0.6, validation_split=0.2, test_split=0.2):
    """
    Splits a pandas dataframe into trainingset, validationset and testset in specified ratio.
    All sets are balanced, which means they have the same ratio for each categorie as the full set.
    Input:   dataframe        - Pandas Dataframe, should include a column for data and one for categories
             column_name      - Name of dataframe column which contains the categorical output values
             training_split   - from ]0,1[, default = 0.6
             validation_split - from ]0,1[, default = 0.2
             test_split       - from ]0,1[, default = 0.2
                                Sum of all splits need to be 1
    Output:  train            - Pandas DataFrame of trainset
             validation       - Pandas DataFrame of validationset
             test             - Pandas DataFrame of testset
    """
    if training_split + validation_split + test_split != 1.0 and training_split > 0 and validation_split > 0 and \
            test_split > 0:
        raise ValueError('Split paramter sum should be 1.0')

    total = len(dataframe.index)

    train = dataframe.reset_index().groupby(column_name).apply(lambda x: x.sample(frac=training_split)) \
        .reset_index(drop=True).set_index('index')
    train = train.sample(frac=1)
    temp_df = dataframe.drop(train.index)
    validation = temp_df.reset_index().groupby(column_name) \
        .apply(lambda x: x.sample(frac=validation_split / (test_split + validation_split))) \
        .reset_index(drop=True).set_index('index')
    validation = validation.sample(frac=1)
    test = temp_df.drop(validation.index)
    test = test.sample(frac=1)

    return train, validation, test


def wordToSeq(text, word_index, max_sentences, max_words, max_features):
    """
    Converts a string to a numpy matrix where each word is tokenized.
    Arrays are zero-padded to max_sentences and max_words length.

    Input:    text           - string of sentences
              word_index     - trained word_index
              max_sentences  - maximum number of sentences allowed per document for HAN
              max_words      - maximum number of words in each sentence for HAN
              max_features   - maximum number of unique words to be tokenized
    Output:   data           - Numpy Matrix of size [max_sentences x max_words]
    """
    sentences = tokenize.sent_tokenize(text)
    data = np.zeros((max_sentences, max_words), dtype='int32')
    for j, sent in enumerate(sentences):
        if j < max_sentences:
            wordTokens = tokenize.word_tokenize(sent.rstrip('.'))
            wordTokens = [w for w in wordTokens]
            k = 0
            for _, word in enumerate(wordTokens):
                try:
                    if k < max_words and word_index[word] < max_features:
                        data[j, k] = word_index[word]
                        k = k + 1
                except:
                    pass
    return data


def toCategorical(series, class_dict):
    """
    Converts category labels to vectors,
    Input:     series     - pandas Series containing numbered category labels
               class_dict - dictionary of integer to category string
                            e.g. {0: 'business', 1: 'entertainment', 2: 'politics', 3: 'sport', 4: 'tech'}
    Output:    Array      - numpy array containing categories converted to lists
                            e.g. 0:'business'      -> [1 0 0 0 0]
                                 1:'entertainment' -> [0 1 0 0 0]
                                 2:'politics'      -> [0 0 1 0 0]
                                 3:'sport'         -> [0 0 0 1 0]
                                 4:'tech'          -> [0 0 0 0 1]
    """
    n_classes = len(class_dict)
    new_dict = {}
    for key, value in class_dict.items():
        cat_list = [0] * n_classes
        cat_list[key] = 1
        new_dict[key] = cat_list
    y_cat = []
    for key, value in series.iteritems():
        y_cat.append(new_dict[value])
    return np.array(y_cat)


def wordAttentionWeights(sequence_sentence, weights):
    """
    Function to calculate a_it (same of Attention Layer)
    :param sequence_sentence:
    :param weights:
    :return: a_it
    """
    u_it = np.dot(sequence_sentence, weights[0]) + weights[1]
    u_it = np.tanh(u_it)

    a_it = np.dot(u_it, weights[2])
    a_it = np.squeeze(a_it)
    a_it = np.exp(a_it)
    a_it /= np.sum(a_it)

    return a_it


def yelpYear(dataset_name, year):
    """
    Select from Yelp complete dataset only rows till a specific year in input and save it in json standard. With large
    dataset can be useful splitting and using one piece at the time. In Linux terminal: split -l and after cat.
    :param dataset_name: string name of dataset, contained in datasets local directory
    :param year: year until
    :return: None
    """
    data_df = pd.read_json("datasets/temp/" + dataset_name + ".json", lines=True)
    data_df = data_df[["stars", "text", "date"]]
    data_df = data_df[(data_df['date'] <= str(year) + '-12-30') & (data_df['date'] >= str(year) + '-01-01')]
    data_df = data_df[["stars", "text"]]
    data_df.columns = ["label", "text"]

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

    data_cleaned.to_csv('datasets/' + dataset_name + '_' + str(year) + '.csv')

    '''
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    for f in files:
        print(f)
        yelpYear('xa' + f, 2014)
        
    dataset_name = 'xab'
    year = 2014
    
    data_df = pd.read_json("datasets/temp/" + dataset_name + ".json", lines=True)
    data_df = data_df[["stars", "text", "date"]]
    data_df = data_df[(data_df['date'] > str(year) + '-12-30') | (data_df['date'] < str(year) + '-01-01')]
    data_df = data_df[["stars", "text"]]
    data_df.columns = ["label", "text"]
    
    data_df.to_csv('datasets/yelp_reviews_container.csv')
    '''


def printAttentionedWordsAndSentences(review, all_sent_index, sent_index, sorted_wordlist, MAX_SENTENCE_NUM):
    """
    Utility function for hanPredict that provides a colored terminal printing (thanks to Sty Python library) of most
    attentioned sentences and words in a predicted review (with partial weights from attention layers of Han network
    model).
    :param review: a string of the review.
    :param all_sent_index: all sentences index.
    :param sent_index: most important sencentences index.
    :param sorted_wordlist: most important words list, sorted by importance.
    :param MAX_SENTENCE_NUM: same parameter of network.
    :return: None
    """

    sentences = tokenize.sent_tokenize(review)
    all_sent_index = np.array(all_sent_index[:len(sentences)])

    nothing = '     '
    low = sty.bg.li_blue + '     ' + sty.bg.rs
    medium = sty.bg(27) + '     ' + sty.bg.rs
    high = sty.bg.da_blue + '     ' + sty.bg.rs

    high_word, medium_word, low_word = np.array_split(sorted_wordlist, 3)
    high_sent, medium_sent, low_sent, nothing_sent = np.array_split(all_sent_index, 4)

    sent_color = nothing
    for idx, sent in enumerate(sentences):
        if idx in high_sent and idx <= MAX_SENTENCE_NUM:
            sent_color = high
        elif idx in medium_sent and idx <= MAX_SENTENCE_NUM:
            sent_color = medium
        elif idx in low_sent and idx <= MAX_SENTENCE_NUM:
            sent_color = low
        else:
            sent_color = nothing

        sent_to_print = ''
        for idy, word in enumerate(tokenize.word_tokenize(sent)):
            if word in high_word and idx in sent_index:
                sent_to_print += (sty.bg.da_red + word + sty.bg.rs + ' ')
            elif word in medium_word and idx in sent_index:
                sent_to_print += (sty.bg(255, 0, 0) + word + sty.bg.rs + ' ')
            elif word in low_word and idx in sent_index:
                sent_to_print += (sty.bg(255, 92, 92) + word + sty.bg.rs + ' ')
            else:
                sent_to_print += (word + ' ')
        print(sent_color, idx, sent_to_print)


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    :param elapsed: time in seconds.
    :return: time in hh:mm::ss format.
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


class CustomDataset(Dataset):
    """
    Custum class that provides tokenization (with appropriatly Bert tokenized in input) of every text of a pandas
    dataframe in input. Also it convertes a int label (target value for the network) in one hot encoding format.
    __getitem__ returns ids (bert encoding of max_len), the relative mask and token_type_ids. Also the processed label
    for the network, renamed 'target'.
    """
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.label = to_categorical(self.data.label)
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.label[index], dtype=torch.long)
        }