from preprocessing import hanPreprocessing, bertPreprocessing
from train import hanTrain, bertTrain, lstmTrain, kdLstmTrain
from predict import hanPredict, lstmEvaluate, bertPredict, bertEvaluate, hanEvaluate
from utils import readImdbSmall, readIMDB, readYelp


if __name__ == '__main__':

    dataset_name, n_classes, data_df = readImdbSmall()
    # hanTrain(dataset_name, n_classes)
    hanEvaluate(dataset_name, model_path='models/model_imdb_reviews/20200617-123807.h5')