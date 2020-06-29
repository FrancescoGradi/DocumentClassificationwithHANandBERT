from preprocessing import hanPreprocessing, bertPreprocessing, kdPreprocessing
from train import hanTrain, bertTrain, lstmTrain, kdLstmTrain
from predict import hanPredict, lstmEvaluate, bertPredict, bertEvaluate, hanEvaluate, softTargetsEvaluate
from utils import readImdbSmall, readIMDB, readYelp


if __name__ == '__main__':

    # dataset_name, n_classes, data_df = readImdbSmall()
    dataset_name, n_classes, data_df = readIMDB()
    # dataset_name, n_classes, data_df = readYelp()

    # lstmTrain(dataset_name, n_classes)
    # lstmEvaluate(dataset_name, n_classes, model_path='models/model_yelp_2014_lstm/ckp_8epochs_20200618-183826', isCheckpoint=True)

    # kdPreprocessing(dataset_name, n_classes, data_df, teacher_path='models/model_IMDB_bert/20200605-184848', MAX_LEN=128)
    # softTargetsEvaluate(dataset_name, n_classes)
    # kdLstmTrain(dataset_name, n_classes, EPOCHS=40, TRAIN_BATCH_SIZE=128, LAMBDA=1)
    # lstmEvaluate(dataset_name, n_classes, model_path='models/model_IMDB_kdLstm/ckp_39epochs_20200626-100027', isCheckpoint=True)

    hanPreprocessing(dataset_name, data_df, save_all=True, cleaned=False, MAX_WORD_NUM=40, MAX_SENTENCE_NUM=20, EMBED_SIZE=100, MAX_FEATURES=50000)
    hanTrain(dataset_name, n_classes, MAX_WORD_NUM=40, MAX_SENTENCE_NUM=20, EMBED_SIZE=100, NUM_EPOCHS=30, MAX_FEATURES=50000,
             LEARNING_RATE=1e-03)
    # hanEvaluate(dataset_name, model_path='models/model_IMDB/20200626-140528.h5')

    # bertTrain(dataset_name, n_classes)
    # bertEvaluate(dataset_name, n_classes, model_path='models/model_IMDB_bert/20200605-184848')