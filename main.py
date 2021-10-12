import os
from features_extraction import Features_handler

import models
import pandas as pd
import dataset_handler as dh

from dataset import Dataset

csv_data =  "./data/noises.csv"
csv_data_new_features = "./data/noises_features.csv"

def main():    
    if not os.path.exists(csv_data_new_features):
    
        if not os.path.exists(csv_data):
            dh.create_csv()

        create_csv_features()

def create_csv_features():
    dataset = Dataset(csv_data)
    dataset.split(0.20, 42)

    featurehandler = Features_handler()   
   
    # convert to radiants -> for haversine
    dataset.x_train = featurehandler.to_radiants(dataset.x_train)
    dataset.y_train = featurehandler.round_lvalues(dataset.y_train)

    featurehandler.init_learners(dataset.x_train, [5, 10])
    
    new_neighbors_features = featurehandler.get_neighbors_features(
        dataset.x_train,
        dataset.y_train)

    for feature_name, values in new_neighbors_features.items():
        dataset.x_train[feature_name] = values
    
    dataset.train_to_csv(csv_data_new_features)

    # y_train = data['noise']

    # models.train_knn(x_train, y_train)
    # # models.train_rf(x_train, y_train)
    # models.train_sgd(x_train, y_train)

    
    # # subsets = dh.create_subsets(data)

if __name__ == '__main__':
    main()
