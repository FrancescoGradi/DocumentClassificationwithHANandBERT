import pandas as pd
import numpy as np
import ijson
import json

from nltk import tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


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

    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    for f in files:
        print(f)
        yelpYear('xa' + f, 2014)