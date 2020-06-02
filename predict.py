import pickle
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model, Model
from nltk.corpus import stopwords
from nltk import tokenize
from sklearn.metrics import classification_report

from hanModel import AttentionLayer, HanModel
from utils import cleanString, wordToSeq, wordAttentionWeights, printAttentionedWordsAndSentences

import tensorflow as tf


def hanPredict(review, review_label, dataset_name, model_path, n_sentences=3, n_words=5, MAX_FEATURES=200000,
               MAX_SENTENCE_NUM=15, MAX_WORD_NUM=25):
    """
    This function tests Han Model Neural Network, reads the pretrained parameters and predicts (with a review in input).
    Also evaluates most attentionable sentences and words thanks to Attention Layer of HAN (it uses weights of some
    layers of the network).
    :param review: a text string to evaluate.
    :param dataset_name: string of dataset name.
    :param model_path: path where is saved the pretrained file .h5.
    :param n_sentences: most attentionable sentences.
    :param n_words: number of most important words to print in this function.
    :param MAX_FEATURES: same parameter used to preprocessing dataset.
    :param MAX_SENTENCE_NUM: same parameter used to preprocessing dataset.
    :param MAX_WORD_NUM: same parameter used to preprocessing dataset.
    :return: None
    """
    # Load model from saved hdf5 file and word index (saved during preprocessing)
    model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    with open('indices/word_index_' + dataset_name + '.txt', 'rb') as f:
        word_index = pickle.load(f)
    stopWords = set(stopwords.words('english'))

    # We clean review and convert to numeric array
    review_cleaned = cleanString(review, stopWords)
    input_array = wordToSeq(review_cleaned, word_index, MAX_SENTENCE_NUM, MAX_WORD_NUM, MAX_FEATURES)

    # We load intermediate models with weights in output capable to evalutae sentence importance
    sent_att_weights = Model(model.inputs, model.get_layer('sent_attention').output, name='SentenceAttention')

    # We predict now the most important sentences, according to trained network
    output_array = sent_att_weights.predict(np.resize(input_array, (1, MAX_SENTENCE_NUM, MAX_WORD_NUM)))[1]

    # We get n_sentences with most attention in document
    sent_index = output_array.flatten().argsort()[-n_sentences:]
    sent_index = np.sort(sent_index)
    sent_index = sent_index.tolist()

    all_sent_index = list(reversed(output_array.flatten().argsort()))

    # Create summary using n sentences
    sent_list = tokenize.sent_tokenize(review)
    try:
        summary = [sent_list[i] for i in sent_index]
    except IndexError:
        print('Number of sentences in this review is to low respect parameter n_sentences choosen.')
        return

    # Summary (n most important sentences) as input for word attention
    summary_cleaned = cleanString(' '.join(summary), stopWords)
    word_input_array = wordToSeq(summary_cleaned, word_index, MAX_SENTENCE_NUM, MAX_WORD_NUM, MAX_FEATURES)

    # We load the word encoder and recreate model for word attention
    word_encoder = model.get_layer('sent_linking').layer
    hidden_word_encoding_out = Model(inputs=word_encoder.input, outputs=word_encoder.get_layer('word_dense').output)

    # Load weights from trained attention layer
    word_context = word_encoder.get_layer('word_attention').get_weights()

    # Compute output of dense layer
    hidden_word_encodings = hidden_word_encoding_out.predict(word_input_array)

    # Compute context vector using output of dense layer
    a_it = wordAttentionWeights(hidden_word_encodings, word_context)

    # Get n words with most attention in document
    flattenlist = []
    words_unpadded = []
    for idx, sent in enumerate(tokenize.sent_tokenize(summary_cleaned)):
        if (idx >= MAX_SENTENCE_NUM):
            break
        attword_list = tokenize.word_tokenize(sent.rstrip('.'))
        a_it_short = (1000 * a_it[idx][:len(attword_list)]).tolist()
        words_unpadded.extend(a_it_short)
        flattenlist.extend(attword_list)


    words_unpadded = np.array(words_unpadded)
    sorted_wordlist = [flattenlist[i] for i in words_unpadded.argsort()]

    sorted_wordlist = list(reversed(sorted_wordlist))

    mostAtt_words = []
    i = 0
    while (i < n_words):
        mostAtt_words.append(sorted_wordlist[i])
        i += 1

    res = model.predict(np.expand_dims(input_array, axis=0)).flatten()
    cat = np.argmax(res.flatten())

    print('')
    print('Review: ' + review)

    if dataset_name == 'yelp_2014':
        print('Stars: ' + str(review_label))
        print('')
        print('Predicted Stars: ' + str(cat + 1))
        print(res)
        print('')
    else:
        print('Category: ' + str(review_label))
        print('')
        print('Predicted Category: ' + str(cat + 1))
        print(res)
        print('')

    print(str(n_sentences) + ' most important sentences: ' + str(summary))
    print(str(n_words) + ' most important words: ' + str(mostAtt_words))
    print('')

    printAttentionedWordsAndSentences(review, all_sent_index, sent_index, sorted_wordlist, MAX_SENTENCE_NUM)


def bertPredict():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    dataset_name = 'imdb_reviews'
    n_classes = 2

    with open('datasets/' + dataset_name + '_bert_cleaned.txt', 'rb') as f:
        data_cleaned = pickle.load(f)

    train_inputs = data_cleaned[0]
    train_mask = data_cleaned[1]
    train_labels = data_cleaned[2]
    validation_inputs = data_cleaned[3]
    validation_mask = data_cleaned[4]
    validation_labels = data_cleaned[5]
    test_inputs = data_cleaned[6]
    test_mask = data_cleaned[7]
    test_labels = data_cleaned[8]

    NUM_EPOCHS = 1
    BATCH_SIZE = 16

    model = load_model('models/model_imdb_reviews_bert/20200602-210813')

    # evaluate the network
    print("Evaluating network...")
    predictions = model.predict(test_inputs, batch_size=BATCH_SIZE)
    # model.evaluate(test_inputs, test_labels, batch_size=BATCH_SIZE, use_multiprocessing=True)
    print(predictions)
    print(classification_report(test_labels, predictions))



def getRandomReview(container_path):
    """
    Function that returns a text and relative label of a review from a test dataset in .csv format.
    :param container_path: path to .csv file contained some test dataset reviews, with columns 'text' and 'label'
    :return: review, label_review
    """
    data_df = pd.read_csv(container_path)
    sample = data_df.sample(1)
    return sample.text.iloc[0], sample.label.iloc[0]


if __name__ == '__main__':

    '''
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    MAX_FEATURES = 200000  # maximum number of unique words that should be included in the tokenized word index
    MAX_SENTENCE_NUM = 15  # maximum number of sentences in one document
    MAX_WORD_NUM = 25  # maximum number of words in each sentence
    EMBED_SIZE = 100  # vector size of word embedding
    BATCH_SIZE = 50
    NUM_EPOCHS = 10
    INIT_LR = 1e-2

    dataset_name = 'yelp_2014'
    model_path = 'models/model_yelp_2014/20200527-220312.h5'
    n_sentences = 2
    n_words = 5

    review, review_label = getRandomReview('datasets/yelp_reviews_container.csv')

    hanPredict(review=review, review_label=review_label, dataset_name=dataset_name, model_path=model_path,
               n_sentences=n_sentences, n_words=n_words, MAX_FEATURES=MAX_FEATURES, MAX_SENTENCE_NUM=MAX_SENTENCE_NUM,
               MAX_WORD_NUM=MAX_WORD_NUM)
    '''

    bertPredict()