import pandas as pd
import numpy as np
import datetime

import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from keras.callbacks import TensorBoard

from preprocessing import preprocessing
from hanModel import HanModel


if __name__ == '__main__':

    MAX_FEATURES = 200000  # maximum number of unique words that should be included in the tokenized word index
    MAX_SENTENCE_NUM = 40  # maximum number of sentences in one document
    MAX_WORD_NUM = 50  # maximum number of words in each sentence
    EMBED_SIZE = 100  # vector size of word embedding
    BATCH_SIZE = 50
    NUM_EPOCHS = 2
    INIT_LR = 1e-2

    # Reading JSON dataset with Pandas
    '''
    dataset_name = "imdb_complete"
    data_df = pd.read_json(dataset_name + ".json")
    data_df = data_df[["rating", "review"]]
    data_df.columns = ["label", "text"]

    '''
    dataset_name = 'imdb_reviews'
    ds = tfds.load(dataset_name, split='train')
    reviews = []
    for element in ds.as_numpy_iterator():
        reviews.append((element['text'].decode('utf-8'), element['label']))

    data_df = pd.DataFrame(data=reviews, columns=['text', 'label'])


    x_train, y_train, x_val, y_val, x_test, y_test, embedding_matrix, word_index, n_classes = preprocessing(
        dataset_name=dataset_name,
        data_df=data_df)


    model = HanModel(n_classes=n_classes, len_word_index=len(word_index), embedding_matrix=embedding_matrix)
    optimizer = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    print(model.summary())

    h = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    # evaluate the network
    print("Evaluating network...")
    predictions = model.predict(x_test, batch_size=BATCH_SIZE)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))

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

    model.save('model_' + str(dataset_name) + '_' + str(NUM_EPOCHS) + '_epoch.h5')
