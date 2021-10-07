import os

import models
import pandas as pd
import dataset_handler as dh

from sklearn.model_selection import train_test_split

def main():
    csv_data =  "./data/noises.csv"
    
    if not os.path.exists(csv_data):
        dh.create_csv()
    
    data = pd.read_csv(csv_data)

    x_train, x_test, y_train, y_test = train_test_split(
        data[['longitude', 'latitude']],
        data['noise'],
        test_size=0.20,
        random_state=42)

    train_dataset = x_train.assign(noise=y_train)
    
    # x_train, y_train = dh.features_engineering(train_dataset)
    print(x_train)
    
    data = pd.read_csv("./data/noises_features.csv")
    print(data.columns)

    x_train = data[[
    'latitude',
    'longitude',

    'nearest_5_points_distance_mean',
    'nearest_5_points_distance_std',
    'nearest_5_points_noise_mean',
    'nearest_5_points_noise_std',
    
    'nearest_10_points_distance_mean',
    'nearest_10_points_distance_std',
    'nearest_10_points_noise_mean',
    'nearest_10_points_noise_std'
    ]]

    y_train = data['noise']

    models.train_knn(x_train, y_train)
    # models.train_rf(x_train, y_train)
    models.train_sgd(x_train, y_train)

    
    # subsets = dh.create_subsets(data)

if __name__ == '__main__':
    main()
