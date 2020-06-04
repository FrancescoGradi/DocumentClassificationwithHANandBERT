import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

import datetime
import os.path
import pathlib
import shutil
import pickle
import json
import time
import datetime
import torch

from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling1D, Dense, Input, Dropout
from tensorflow.keras.models import Model
from transformers import TFBertForSequenceClassification, TFBertModel, BertConfig, TFBertForTokenClassification, \
    BertForSequenceClassification
from torch import cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from preprocessing import preprocessing
from hanModel import HanModel
from bertModel import BertModel
from utils import wordAndSentenceCounter, format_time, loss_fn


def bertTrainNew():
    device = 'cuda' if cuda.is_available() else 'cpu'

    dataset_name = 'imdb_complete'
    n_classes = 11

    with open('datasets/' + dataset_name + '_bert_cleaned.txt', 'rb') as f:
        data_cleaned = pickle.load(f)

    training_set = data_cleaned[0]
    validation_set = data_cleaned[1]
    test_set = data_cleaned[2]
    MAX_LEN = data_cleaned[3]

    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 4
    EPOCHS = 3
    LEARNING_RATE = 1e-05

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **test_params)
    testing_loader = DataLoader(test_set, **test_params)

    model = BertModel(n_classes=n_classes, dropout=0.3)
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # Stats with Tensorboard
    log_dir = "logs/" + dataset_name + "_bert/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.rmtree(log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=log_dir)

    total_t0 = time.time()

    for epoch in range(EPOCHS):
        print("")
        print('============================== Epoch {:} / {:} =============================='.format(epoch + 1, EPOCHS))
        print('Training...')
        t0 = time.time()
        model.train()

        for step, batch in enumerate(training_loader):

            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.long)

            model.zero_grad()

            outputs = model(ids, mask, token_type_ids)
            loss = criterion(outputs, torch.max(targets, 1)[1])

            if step % 200 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.   Loss: {:>19,}   Elapsed: {:}.'.format(step, len(training_loader),
                                                                                           loss, elapsed))

                writer.add_scalar('batch_loss', loss, step + (epoch * len(training_loader)))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_time = format_time(time.time() - t0)
        print("  Training epoch took: {:}".format(training_time))
        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for batch in validation_loader:

                ids = batch['ids'].to(device, dtype=torch.long)
                mask = batch['mask'].to(device, dtype=torch.long)
                token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                targets = batch['targets'].to(device, dtype=torch.long)

                outputs = model(ids, mask, token_type_ids)

                total_eval_loss += criterion(outputs, torch.max(targets, 1)[1])

                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy().tolist())

        valid_loss = total_eval_loss / len(validation_loader)

        fin_outputs = np.array(fin_outputs)
        fin_targets =np.array(fin_targets)

        accuracy = accuracy_score(fin_targets.argmax(axis=1), fin_outputs.argmax(axis=1))

        print("  Validation Accuracy: {0:.2f}".format(accuracy))
        print("  Validation Loss: {0:.2f}".format(valid_loss))
        writer.add_scalar('epoch_loss', valid_loss, epoch)
        writer.add_scalar('epoch_accuracy', accuracy, epoch)

    print("")
    print("Training complete!")
    print("Saving model...")
    torch.save(model.state_dict(),
               'models/model_' + dataset_name + '_bert/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


def bertTrain():

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
    MAX_LEN = 128

    print(train_labels)


    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=n_classes)
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)

    '''
    
    token_inputs = Input((MAX_LEN), dtype=tf.int32, name='input_word_ids')
    bert_layers = TFBertModel.from_pretrained("bert-base-uncased")
    bert_output, _ = bert_layers(token_inputs)
    dense = Dense(128, activation='relu')(bert_output)
    dense_drop = Dropout(rate=0.2, name='dropout')(dense)
    cls_output = Dense(n_classes, activation='softmax', name='cls_output')(dense_drop)

    model = Model(token_inputs, cls_output)
    '''

    optimizer = Adam()
    loss = tf.keras.losses.CategoricalCrossentropy()
    metric = tf.keras.metrics.BinaryAccuracy(name='accuracy')
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
    print(model.summary())

    log_dir = "logs/" + dataset_name + "_bert/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.rmtree(log_dir, ignore_errors=True)

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True),
        TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0, update_freq=50),
    ]

    h = model.fit(train_inputs, train_labels,
                  validation_data=(validation_inputs, validation_labels),
                  epochs=NUM_EPOCHS,
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
    bertTrainNew()