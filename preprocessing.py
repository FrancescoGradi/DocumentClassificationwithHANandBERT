import string
import pickle

import pandas as pd
import tensorflow_datasets as tfds

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from nltk.corpus import stopwords

from utils import cleanString


MAX_FEATURES = 200000  # maximum number of unique words that should be included in the tokenized word index
MAX_SENTENCE_NUM = 40  # maximum number of sentences in one document
MAX_WORD_NUM = 50  # maximum number of words in each sentence
EMBED_SIZE = 100  # vector size of word embedding


if __name__ == '__main__':
    '''
    # Reading JSON dataset with Pandas

    data_df = pd.read_json("IMDB.json")
    data_df = data_df[["rating", "review"]]
    data_df.columns = ["label", "text"]
    '''

    ds = tfds.load('imdb_reviews', split='train')
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

    # We construct a word index that associates a word to a integer number, and it is saved

    texts = []
    n = data_cleaned['text'].shape[0]
    for i in range(n):
        s = data_cleaned['text'].iloc[i]
        s = ' '.join([word.strip(string.punctuation) for word in s.split() if word.strip(string.punctuation) is not ""])
        texts.append(s)
    tokenizer = Tokenizer(num_words=MAX_FEATURES, lower=True, oov_token=None)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    with open('word_index.txt', 'wb') as f:
        pickle.dump(word_index, f)