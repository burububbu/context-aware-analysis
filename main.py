import os
import utils

from classes.dataset import Dataset
from classes.models_handler import ModelsHandler

csv_data =  "./data/noises.csv"
csv_train_data = "./data/train_data.csv"
csv_test_data = "./data/test_data.csv"

knn_params = {
    'n_neighbors':[5, 10, 20, 30, 50, 60, 100, 150],
    'weights': ['uniform', 'distance'],
}

rf_params = {
    'n_estimators': [200, 300, 500]
}

nn_params = {
    "hidden_sizes": [5, 10, 64],
    "num_epochs": [50, 100, 500],
    "batch_sizes": [64, 128],
    "learning_rates": [0.1, 0.001]
}

# nn_params = {
#     "hidden_sizes": [5, 10, 64],
#     "num_epochs": [5],
#     "batch_sizes": [64],
#     "learning_rates": [0.1]
# }



def main():
    dataset = Dataset()

    if not os.path.exists(csv_train_data):
        if not os.path.exists(csv_data):
            utils.create_data_csv(csv_data)
        dataset = utils.generate_features(csv_data, csv_train_data, csv_test_data)
    else:
        dataset.load_train_test_data(csv_train_data, csv_test_data)

    dataset.select_best_features(5)

    models_handler = ModelsHandler(dataset)
    
    models_handler.create_models('knn', knn_params)
    models_handler.create_models('rf', rf_params)
    models_handler.create_models('sgd', {})
    models_handler.create_neural_networks(nn_params)



if __name__ == '__main__':
    main()
