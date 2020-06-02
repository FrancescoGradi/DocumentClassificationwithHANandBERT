import pandas as pd
import numpy as np
import datetime
import os.path
import pathlib
import shutil
import pickle
import json
import tensorflow_datasets as tfds
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from transformers import TFBertForSequenceClassification

from preprocessing import preprocessing
from hanModel import HanModel
from utils import wordAndSentenceCounter


def bertTrain():
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

    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=n_classes,
                                                            output_attentions=False, output_hidden_states=False)
    optimizer = Adam()
    loss = tf.keras.losses.CategoricalCrossentropy()
    metric = tf.keras.metrics.CategoricalAccuracy(name='accuracy')
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    print(model.summary())

    log_dir = "logs/" + dataset_name + "_bert/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.rmtree(log_dir, ignore_errors=True)

    callbacks = [
        EarlyStopping(monitor='acc', patience=4, restore_best_weights=True),
        TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0, update_freq=10),
    ]

    h = model.fit(train_inputs, train_labels,
                  validation_data=(validation_inputs, validation_labels),
                  epochs=NUM_EPOCHS,
                  batch_size=BATCH_SIZE,
                  callbacks=callbacks)

    os.makedirs(
        os.path.dirname('models/model_' + dataset_name + '_bert/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                        + '.h5'), exist_ok=True)
    model.save('models/model_' + dataset_name + '_bert/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
               save_format='tf')

    # evaluate the network
    print("Evaluating network...")
    predictions = model.predict(test_inputs, batch_size=BATCH_SIZE)
    model.evaluate(test_inputs, test_labels, batch_size=BATCH_SIZE, use_multiprocessing=True)
    print(predictions)
    print(test_labels)
    print(classification_report(test_labels.argmax(axis=1), predictions.argmax(axis=1)))


def hanTrain():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    '''
    MAX_FEATURES = 200000  # maximum number of unique words that should be included in the tokenized word index
    MAX_SENTENCE_NUM = 15  # maximum number of sentences in one document
    MAX_WORD_NUM = 25  # maximum number of words in each sentence
    EMBED_SIZE = 100  # vector size of word embedding
    BATCH_SIZE = 64
    NUM_EPOCHS = 25
    INIT_LR = 1e-2

    # Reading dataset with Pandas

    dataset_name = "yelp_2014"
    data_df = pd.read_csv("datasets/" + dataset_name + ".csv")
    cleaned = True
    '''
    
    MAX_FEATURES = 200000  # maximum number of unique words that should be included in the tokenized word index
    MAX_SENTENCE_NUM = 20  # maximum number of sentences in one document
    MAX_WORD_NUM = 40  # maximum number of words in each sentence
    EMBED_SIZE = 100  # vector size of word embedding
    BATCH_SIZE = 64
    NUM_EPOCHS = 60
    INIT_LR = 1e-2

    # Reading JSON dataset with Pandas

    dataset_name = "imdb_complete"
    data_df = pd.read_json("datasets/" + dataset_name + ".json")
    data_df = data_df[["rating", "review"]]
    data_df.columns = ["label", "text"]
    
    cleaned = False
    '''

    MAX_FEATURES = 200000  # maximum number of unique words that should be included in the tokenized word index
    MAX_SENTENCE_NUM = 25  # maximum number of sentences in one document
    MAX_WORD_NUM = 40  # maximum number of words in each sentence
    EMBED_SIZE = 100  # vector size of word embedding
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    INIT_LR = 1e-2
    
    dataset_name = 'imdb_reviews'
    ds = tfds.load(dataset_name, split='train')
    reviews = []
    for element in ds.as_numpy_iterator():
        reviews.append((element['text'].decode('utf-8'), element['label']))

    data_df = pd.DataFrame(data=reviews, columns=['text', 'label'])
    cleaned = False
    '''

    if (os.path.isfile('datasets/' + dataset_name + '_cleaned.txt')):
        with open('datasets/' + dataset_name + '_cleaned.txt', 'rb') as f:
            data_cleaned = pickle.load(f)
        x_train = data_cleaned[0]
        y_train = data_cleaned[1]
        x_val = data_cleaned[2]
        y_val = data_cleaned[3]
        x_test = data_cleaned[4]
        y_test = data_cleaned[5]
        embedding_matrix = data_cleaned[6]
        word_index = data_cleaned[7]
        n_classes = data_cleaned[8]
    else:
        x_train, y_train, x_val, y_val, x_test, y_test, embedding_matrix, word_index, n_classes = preprocessing(
            dataset_name=dataset_name, data_df=data_df, save_all=True, cleaned=cleaned, MAX_FEATURES=MAX_FEATURES,
            MAX_SENTENCE_NUM=MAX_SENTENCE_NUM, MAX_WORD_NUM=MAX_WORD_NUM, EMBED_SIZE=EMBED_SIZE)


    model = HanModel(n_classes=n_classes, len_word_index=len(word_index), embedding_matrix=embedding_matrix,
                     MAX_SENTENCE_NUM=MAX_SENTENCE_NUM, MAX_WORD_NUM=MAX_WORD_NUM, EMBED_SIZE=EMBED_SIZE)
    optimizer = SGD(momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    print(model.summary())

    log_dir = "logs/" + dataset_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.rmtree(log_dir, ignore_errors=True)

    callbacks = [
        EarlyStopping(monitor='acc', patience=4, restore_best_weights=True),
        TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0),
    ]

    h = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=callbacks)

    # evaluate the network
    print("Evaluating network...")
    predictions = model.predict(x_test, batch_size=BATCH_SIZE)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))

    os.makedirs(os.path.dirname('models/model_' + dataset_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                                + '.h5'), exist_ok=True)
    model.save('models/model_' + dataset_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')


    # determine the number of epochs and then construct the plot title
    N = np.arange(0, NUM_EPOCHS)
    title = "Training Loss and Accuracy on {0} with {1}".format(dataset_name, model.name)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, h.history["loss"], label="train_loss")
    plt.plot(N, h.history["val_loss"], label="val_loss")
    plt.plot(N, h.history["acc"], label="train_acc")
    plt.plot(N, h.history["val_acc"], label="val_acc")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    bertTrain()