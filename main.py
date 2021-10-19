import os
from dataset import Dataset
from models_handler import ModelsHandler
import utils

from ml_models import models

csv_data =  "./data/noises.csv"
csv_train_data = "./data/train_data.csv"
csv_test_data = "./data/test_data.csv"

knn_params = {
    'n_neighbors':[5, 10, 20, 30, 50, 60, 100, 150],
    'weights': ['uniform', 'distance'],
}

rf_params = {
    'n_estimators': [100, 200, 300, 500]
}


def main():
    dataset = Dataset()

    if not os.path.exists(csv_train_data):
        if not os.path.exists(csv_data):
            utils.create_data_csv(csv_data)
        dataset = utils.generate_features(csv_data, csv_train_data, csv_test_data)
    else:
        dataset.load_train_test_data(csv_train_data, csv_test_data)

    models = ModelsHandler(dataset)

    # models.create_knn(knn_params)
    models.create_rf(rf_params)
    # models.create_sgd({})



if __name__ == '__main__':
    main()
