import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import torch.nn.functional as F

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
    BertForSequenceClassification, BertTokenizer
from torch import cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from preprocessing import hanPreprocessing
from hanModel import HanModel
from bertModel import BertModel
from lstmModel import LSTMBase
from utils import wordAndSentenceCounter, formatTime


def kdLstmTrain(dataset_name, n_classes, teacher_path, validation=True, from_checkpoint=False, student_path=None):
    device = 'cuda' if cuda.is_available() else 'cpu'

    with open('datasets/' + dataset_name + '_bert_cleaned.txt', 'rb') as f:
        data_cleaned = pickle.load(f)

    training_set = data_cleaned[0]
    validation_set = data_cleaned[1]
    test_set = data_cleaned[2]
    MAX_LEN = data_cleaned[3]

    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 8
    EPOCHS = 30
    LEARNING_RATE = 1e-03
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 256
    LAMBDA = 1
    start_epoch = 0

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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    student_model = LSTMBase(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=tokenizer.vocab_size,
                     n_classes=n_classes)
    print(student_model)
    total_params = sum(p.numel() for p in student_model.parameters())
    total_trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print('Student total parameters: {:}'.format(total_params))
    print('Student trainable parameters: {:}'.format(total_trainable_params))

    teacher_model = BertModel(n_classes=n_classes, dropout=0.3)
    print(teacher_model)
    total_params = sum(p.numel() for p in teacher_model.parameters())
    print('Teacher total parameters: {:}'.format(total_params))

    optimizer = torch.optim.Adam(params=student_model.parameters(), lr=LEARNING_RATE)

    classification_criterion = torch.nn.CrossEntropyLoss()
    distillation_criterion = torch.nn.KLDivLoss()

    if from_checkpoint == True:
        print('Restoring model from checkpoint...')
        torch.cuda.empty_cache()
        checkpoint = torch.load(student_path, map_location=device)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
        print('Start epoch: {:}'.format(start_epoch + 1))

    student_model.to(device)
    teacher_model.to(device)

    teacher_model.eval()

    # Stats with Tensorboard
    log_dir = "logs/" + dataset_name + "_kdLstm/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.rmtree(log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=log_dir)

    total_t0 = time.time()

    for epoch in range(start_epoch, EPOCHS):
        print("")
        print('============================== Epoch {:} / {:} =============================='.format(epoch + 1, EPOCHS))
        print('Training...')
        t0 = time.time()
        student_model.train()

        for step, batch in enumerate(training_loader):

            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.long)

            student_model.zero_grad()

            outputs = student_model(ids)
            outputs = torch.softmax(outputs, dim=1)

            distillation = LAMBDA * F.kl_div(outputs, torch.softmax(teacher_model(ids, mask, token_type_ids), dim=1), 'batchmean')

            # Loss = classification + lambda * distillation
            loss = classification_criterion(outputs, torch.max(targets, 1)[1]) + distillation

            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = formatTime(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.   Loss: {:>19,}   Elapsed: {:}.'.format(step, len(training_loader),
                                                                                           loss, elapsed))

                writer.add_scalar('batch_loss', loss, step + (epoch * len(training_loader)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_time = formatTime(time.time() - t0)
        print("  Training epoch took: {:}".format(training_time))
        print("  Saving checkpoint...")
        os.makedirs(os.path.dirname(
            'models/model_' + dataset_name + '_kdLstm/ckp_' + str(epoch) + 'epochs_' + datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S")), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss},
            'models/model_' + dataset_name + '_kdLstm/ckp_' + str(epoch) + 'epochs_' + datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S"))
        print("")

        if validation == True:
            print("Running Validation...")

            student_model.eval()

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

                    outputs = student_model(ids)
                    outputs = torch.softmax(outputs, dim=1)

                    total_eval_loss += classification_criterion(outputs, torch.max(targets, 1)[1]) + \
                                       (LAMBDA * F.kl_div(outputs, torch.softmax(teacher_model(ids, mask, token_type_ids), dim=1), 'batchmean'))

                    fin_targets.extend(targets.cpu().detach().numpy().tolist())
                    fin_outputs.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy().tolist())

            valid_loss = total_eval_loss / len(validation_loader)

            fin_outputs = np.array(fin_outputs)
            fin_targets = np.array(fin_targets)

            accuracy = accuracy_score(fin_targets.argmax(axis=1), fin_outputs.argmax(axis=1))

            print("  Validation Accuracy: {0:.2f}".format(accuracy))
            print("  Validation Loss: {0:.2f}".format(valid_loss))
            writer.add_scalar('epoch_loss', valid_loss, epoch)
            writer.add_scalar('epoch_accuracy', accuracy, epoch)

    print("")
    print("Training complete!")
    print("Saving model...")
    os.makedirs(
        os.path.dirname('models/model_' + dataset_name + '_kdLstm/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
        exist_ok=True)
    torch.save(student_model.state_dict(),
               'models/model_' + dataset_name + '_kdLstm/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print("Total training took {:} (h:mm:ss)".format(formatTime(time.time() - total_t0)))


def lstmTrain(dataset_name, n_classes, validation=True, from_checkpoint=False, model_path=None):
    device = 'cuda' if cuda.is_available() else 'cpu'

    with open('datasets/' + dataset_name + '_bert_cleaned.txt', 'rb') as f:
        data_cleaned = pickle.load(f)

    training_set = data_cleaned[0]
    validation_set = data_cleaned[1]
    test_set = data_cleaned[2]
    MAX_LEN = data_cleaned[3]

    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 8
    EPOCHS = 40
    LEARNING_RATE = 1e-03
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 256
    start_epoch = 0

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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = LSTMBase(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=tokenizer.vocab_size,
                     n_classes=n_classes)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters: {:}'.format(total_params))
    print('Total trainable parameters: {:}'.format(total_trainable_params))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    if from_checkpoint == True:
        print('Restoring model from checkpoint...')
        torch.cuda.empty_cache()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
        print('Start epoch: {:}'.format(start_epoch + 1))

    model.to(device)

    # Stats with Tensorboard
    log_dir = "logs/" + dataset_name + "_lstm/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.rmtree(log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=log_dir)

    total_t0 = time.time()

    for epoch in range(start_epoch, EPOCHS):
        print("")
        print('============================== Epoch {:} / {:} =============================='.format(epoch + 1, EPOCHS))
        print('Training...')
        t0 = time.time()
        model.train()

        for step, batch in enumerate(training_loader):

            ids = batch['ids'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.long)

            model.zero_grad()

            outputs = model(ids)

            loss = criterion(torch.softmax(outputs, dim=1), torch.max(targets, 1)[1])

            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = formatTime(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.   Loss: {:>19,}   Elapsed: {:}.'.format(step, len(training_loader), loss, elapsed))

                writer.add_scalar('batch_loss', loss, step + (epoch * len(training_loader)))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_time = formatTime(time.time() - t0)
        print("  Training epoch took: {:}".format(training_time))
        print("  Saving checkpoint...")
        os.makedirs(os.path.dirname('models/model_' + dataset_name + '_lstm/ckp_' + str(epoch) + 'epochs_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), exist_ok=True)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss},
                    'models/model_' + dataset_name + '_lstm/ckp_' + str(epoch) + 'epochs_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        print("")

        if validation == True:
            print("Running Validation...")

            model.eval()

            total_eval_accuracy = 0
            total_eval_loss = 0

            fin_targets = []
            fin_outputs = []
            with torch.no_grad():
                for batch in validation_loader:

                    ids = batch['ids'].to(device, dtype=torch.long)
                    targets = batch['targets'].to(device, dtype=torch.long)

                    outputs = model(ids)

                    total_eval_loss += criterion(torch.softmax(outputs, dim=1), torch.max(targets, 1)[1])

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
    os.makedirs(os.path.dirname('models/model_' + dataset_name + '_lstm/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), exist_ok=True)
    torch.save(model.state_dict(),
               'models/model_' + dataset_name + '_lstm/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print("Total training took {:} (h:mm:ss)".format(formatTime(time.time() - total_t0)))


def bertTrain(dataset_name, n_classes, validation=True, from_checkpoint=False, model_path=None):
    device = 'cuda' if cuda.is_available() else 'cpu'

    with open('datasets/' + dataset_name + '_bert_cleaned.txt', 'rb') as f:
        data_cleaned = pickle.load(f)

    training_set = data_cleaned[0]
    validation_set = data_cleaned[1]
    test_set = data_cleaned[2]
    MAX_LEN = data_cleaned[3]

    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 1e-05
    start_epoch = 0

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

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    if from_checkpoint == True:
        print('Restoring model from checkpoint...')
        torch.cuda.empty_cache()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
        print('Start epoch: {:}'.format(start_epoch + 1))

    model.to(device)

    # Stats with Tensorboard
    log_dir = "logs/" + dataset_name + "_bert/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.rmtree(log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=log_dir)

    total_t0 = time.time()

    for epoch in range(start_epoch, EPOCHS):
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
            outputs = torch.softmax(outputs, dim=1)

            loss = criterion(outputs, torch.max(targets, 1)[1])

            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = formatTime(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.   Loss: {:>19,}   Elapsed: {:}.'.format(step, len(training_loader), loss, elapsed))

                writer.add_scalar('batch_loss', loss, step + (epoch * len(training_loader)))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_time = formatTime(time.time() - t0)
        print("  Training epoch took: {:}".format(training_time))
        print("  Saving checkpoint...")
        os.makedirs(os.path.dirname('models/model_' + dataset_name + '_bert/ckp_' + str(epoch) + 'epochs_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), exist_ok=True)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss},
                    'models/model_' + dataset_name + '_bert/ckp_' + str(epoch) + 'epochs_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        print("")

        if validation == True:
            print("Running Validation...")

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

                    total_eval_loss += criterion(torch.softmax(outputs, dim=1), torch.max(targets, 1)[1])

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
    os.makedirs(os.path.dirname('models/model_' + dataset_name + '_bert/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), exist_ok=True)
    torch.save(model.state_dict(),
               'models/model_' + dataset_name + '_bert/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print("Total training took {:} (h:mm:ss)".format(formatTime(time.time() - total_t0)))


def hanTrain(dataset_name, n_classes, cleaned=False):
    #physical_devices = tf.config.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
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
    
    
    MAX_FEATURES = 200000  # maximum number of unique words that should be included in the tokenized word index
    MAX_SENTENCE_NUM = 20  # maximum number of sentences in one document
    MAX_WORD_NUM = 40  # maximum number of words in each sentence
    EMBED_SIZE = 100  # vector size of word embedding
    BATCH_SIZE = 64
    NUM_EPOCHS = 60
    INIT_LR = 1e-2

    # Reading JSON dataset with Pandas

    dataset_name = 'IMDB'
    train_df = pd.read_csv('datasets/' + dataset_name + '/train.tsv', sep='\t')
    train_df.columns = ['label', 'text']
    test_df = pd.read_csv('datasets/' + dataset_name + '/test.tsv', sep='\t')
    test_df.columns = ['label', 'text']
    dev_df = pd.read_csv('datasets/' + dataset_name + '/dev.tsv', sep='\t')
    dev_df.columns = ['label', 'text']
    data_df = pd.concat([train_df, test_df, dev_df], ignore_index=True)
    data_df['label'] = data_df['label'].apply(lambda x: len(str(x)) - 1)
    
    cleaned = False
    '''

    MAX_FEATURES = 200000  # maximum number of unique words that should be included in the tokenized word index
    MAX_SENTENCE_NUM = 25  # maximum number of sentences in one document
    MAX_WORD_NUM = 40  # maximum number of words in each sentence
    EMBED_SIZE = 100  # vector size of word embedding
    BATCH_SIZE = 128
    NUM_EPOCHS = 45
    INIT_LR = 1e-2
    '''
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
        x_train, y_train, x_val, y_val, x_test, y_test, embedding_matrix, word_index, n_classes = hanPreprocessing(
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
    dataset_name = 'yelp_2014'
    n_classes = 5
    hanTrain(dataset_name, n_classes)
    #lstmTrain(dataset_name=dataset_name, n_classes=n_classes)
    #kdLstmTrain(dataset_name, n_classes, from_checkpoint=True, student_path='models/model_yelp_2014_kdLstm/ckp_1epochs_20200611-165400', teacher_path='models/model_yelp_2014_bert/20200607-201214')
    #bertTrain(dataset_name, n_classes)