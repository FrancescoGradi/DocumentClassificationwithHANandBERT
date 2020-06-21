from preprocessing import hanPreprocessing, bertPreprocessing
from train import hanTrain, bertTrain, lstmTrain, kdLstmTrain
from predict import hanPredict, lstmEvaluate, bertPredict, bertEvaluate, hanEvaluate
from utils import readImdbSmall, readIMDB, readYelp


if __name__ == '__main__':

    # dataset_name, n_classes, data_df = readImdbSmall()
    dataset_name, n_classes, data_df = readIMDB()
    # dataset_name, n_classes, data_df = readYelp()

    # lstmTrain(dataset_name, n_classes)
    # lstmEvaluate(dataset_name, n_classes, model_path='models/model_imdb_reviews_lstm/ckp_12epochs_20200618-133908', isCheckpoint=True)

    kdLstmTrain(dataset_name, n_classes, teacher_path='models/model_IMDB_bert/20200605-184848', student_path='models/model_IMDB_kdLstm/ckp_17epochs_20200620-231848', from_checkpoint=True)
    # lstmEvaluate(dataset_name, n_classes, model_path='models/model_imdb_reviews_kdLstm/ckp_17epochs_20200618-222526', isCheckpoint=True)

    # hanTrain(dataset_name, n_classes)
    # hanEvaluate(dataset_name, model_path='models/model_yelp_2014/20200615-210355.h5')

    # bertTrain(dataset_name, n_classes)
    # bertEvaluate(dataset_name, n_classes, model_path='models/model_IMDB_bert/20200605-184848')