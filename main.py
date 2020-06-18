from preprocessing import hanPreprocessing, bertPreprocessing
from train import hanTrain, bertTrain, lstmTrain, kdLstmTrain
from predict import hanPredict, lstmEvaluate, bertPredict, bertEvaluate, hanEvaluate
from utils import readImdbSmall, readIMDB, readYelp


if __name__ == '__main__':

    dataset_name, n_classes, data_df = readImdbSmall()
    # dataset_name, n_classes, data_df = readIMDB()
    # dataset_name, n_classes, data_df = readYelp()

    # lstmTrain(dataset_name, n_classes)
    # lstmEvaluate(dataset_name, n_classes, model_path='models/model_yelp_2014_lstm/ckp_8epochs_20200618-183826', isCheckpoint=True)

    # kdLstmTrain(dataset_name, n_classes, teacher_path='models/model_imdb_reviews_bert/20200604-141128')
    lstmEvaluate(dataset_name, n_classes, model_path='models/model_imdb_reviews_kdLstm/ckp_17epochs_20200618-222526', isCheckpoint=True)

    # hanTrain(dataset_name, n_classes)
    # hanEvaluate(dataset_name, model_path='models/model_yelp_2014/20200615-210355.h5')

    # bertTrain(dataset_name, n_classes)
    # bertEvaluate(dataset_name, n_classes, model_path='models/model_yelp_2014_bert/ckp_2epochs_20200607-201214', isCheckpoint=True)