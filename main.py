from preprocessing import hanPreprocessing, bertPreprocessing, kdPreprocessing
from train import hanTrain, bertTrain, lstmTrain, kdLstmTrain
from predict import hanPredict, lstmEvaluate, bertPredict, bertEvaluate, hanEvaluate
from utils import readImdbSmall, readIMDB, readYelp


if __name__ == '__main__':

    # dataset_name, n_classes, data_df = readImdbSmall()
    dataset_name, n_classes, data_df = readIMDB()
    # dataset_name, n_classes, data_df = readYelp()

    # lstmTrain(dataset_name, n_classes)
    # lstmEvaluate(dataset_name, n_classes, model_path='models/model_imdb_reviews_lstm/ckp_12epochs_20200618-133908', isCheckpoint=True)

    kdPreprocessing(dataset_name, data_df, MAX_LEN=128)
    kdLstmTrain(dataset_name, n_classes, teacher_path='models/model_IMDB_bert/20200605-184848',
                student_path='models/model_IMDB_kdLstm/ckp_29epochs_20200622-190819', from_checkpoint=True, EPOCHS=40)
    # lstmEvaluate(dataset_name, n_classes, model_path='models/model_imdb_reviews_kdLstm/ckp_17epochs_20200618-222526', isCheckpoint=True)

    # hanPreprocessing(dataset_name, data_df, save_all=True, cleaned=True, MAX_WORD_NUM=25, MAX_SENTENCE_NUM=15, EMBED_SIZE=100)
    # hanTrain(dataset_name, n_classes, MAX_WORD_NUM=25, MAX_SENTENCE_NUM=15, EMBED_SIZE=100)

    # hanEvaluate(dataset_name, model_path='models/model_yelp_2014/20200615-210355.h5')

    # bertTrain(dataset_name, n_classes)
    # bertEvaluate(dataset_name, n_classes, model_path='models/model_IMDB_bert/20200605-184848')