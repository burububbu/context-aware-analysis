import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import database.database as db

from dataset import Dataset
from learners import Learners

from dotenv import load_dotenv, find_dotenv

def create_data_csv(csv_data):
    load_dotenv(find_dotenv())
    
    results = db.getAll()

    longs = []
    lats = []
    noises = []
 
    for res in results:
        longs.append(res.longitude)
        lats.append(res.latitude)
        noises.append(res.noise)

    data = pd.DataFrame({'latitude':lats, 'longitude': longs, 'noise':noises})

    print('Noises data saved at path {0}'.format(csv_data))
    
    data.to_csv(csv_data, index=False)

def generate_features(csv_data, csv_train_data, csv_test_data, to_csv = True):
    ''' Also returns the Dataset object containig the data'''
    dataset = Dataset()
    dataset.load_data(csv_data)
    
    dataset.split(0.20, 42)

    features_handler = Learners()
    
    _add_train_features(dataset, features_handler)
    _add_test_features(dataset, features_handler)
    
    if to_csv:
        dataset.train_to_csv(csv_train_data)
        print('Train data saved at path {0}'.format(csv_train_data))
        
        dataset.test_to_csv(csv_test_data)
        print('Test data saved at path {0}'.format(csv_test_data))
    
    return dataset

def _add_train_features(dataset, features_handler):
    # convert altitude and longitude to radiants -> for haversine
    dataset.x_train = _to_radiants(dataset.x_train)
    dataset.y_train = _round_values(dataset.y_train)

    features_handler.init_learners(dataset.x_train[['latitude', 'longitude']], [5, 10])
    
    new_neighbors_features = features_handler.get_neighbors_features(
        dataset.x_train[['latitude', 'longitude']],
        dataset.y_train)

    for feature_name, values in new_neighbors_features.items():
        dataset.add_column_to_train(feature_name, values)

def _add_test_features(dataset, features_handler):
    # convert altitude and longitude to radiants -> for haversine
    dataset.x_test = _to_radiants(dataset.x_test)
    dataset.y_test = _round_values(dataset.y_test)
    
    new_neighbors_features = features_handler.get_neighbors_features(
        dataset.x_test[['latitude', 'longitude']], # take x data from test (query created)
        dataset.y_train) # take noises from train

    for feature_name, values in new_neighbors_features.items():
        dataset.add_column_to_test(feature_name, values)

def _round_values(data):
    return data.apply(lambda value: round(value,5))
    
def _to_radiants(data):    
    data['latitude'] = data['latitude'].map(np.radians) 
    data['longitude'] = data['longitude'].map(np.radians)
    return data


def plot(x_label, y_label, title, file_name):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig("./plots/{}.png".format(file_name))

    plt.clf()
