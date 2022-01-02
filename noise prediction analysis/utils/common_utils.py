import pandas as pd
import numpy as np
from sqlalchemy.sql.sqltypes import String
import database_connection.database as db

from utils.dataset import Dataset

from sklearn.neighbors import NearestNeighbors

from dotenv import load_dotenv, find_dotenv

from utils.dataset import Dataset
import const
import os

from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

from utils.nn_classes import DatasetNN


def load_dataset(num_subset):
    dataset = Dataset()

    if os.path.exists(const.csv_train_data):
        dataset.load_train_test_from_csv(
            const.csv_train_data, const.csv_test_data, num_sub_features=num_subset)
    else:
        # we have to create train and test datasets (with new features generated)

        data = pd.read_csv(const.csv_data) if os.path.exists(
            const.csv_data) else load_data(const.csv_data, to_csv=True)

        # create train and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            # all columns except noise
            data[data.columns.drop('noise')],
            data['noise'],
            test_size=const.test_size,
            random_state=const.random_seed)

        x_train, x_test = generate_features(
            x_train, y_train, x_test, y_test, const.learners_neighbors, to_csv=True)  # [5, 10, 50]

        dataset.load_train_test_from_df(
            x_train.join(y_train), x_test.join(y_test), num_sub_features=num_subset)

    return dataset


def load_data(csv_path: String, to_csv):
    ''' Get real noises from database and save as .csv'''
    # get database connection info
    load_dotenv(find_dotenv())

    results = db.getAll()

    longs = []
    lats = []
    noises = []

    for res in results:
        lats.append(res.latitude)
        longs.append(res.longitude)
        noises.append(res.noise)

    data = pd.DataFrame(
        {'latitude': lats, 'longitude': longs, 'noise': noises})

    if to_csv:
        data.to_csv(csv_path, index=False)
        print('Noises data saved at path {0}'.format(csv_path))

    return data


def get_new_features(x_data, y_data, neighbors_learners):
    features_values = dict()

    for num_neigh, learner in neighbors_learners.items():
        all_distances, all_neighbors_indexes = learner.kneighbors(x_data)

        distance_mean = []
        distance_std = []

        for distances in all_distances:
            distance_mean.append(np.mean(distances))
            distance_std.append(np.std(distances))

        noise_mean = []
        noise_std = []

        for neighbors_indexes in all_neighbors_indexes:
            noises = y_data.iloc[neighbors_indexes]

            noise_mean.append(np.mean(noises))
            noise_std.append(np.std(noises))

        features_values[num_neigh + '_distance_mean'] = distance_mean
        features_values[num_neigh + '_distance_std'] = distance_std
        features_values[num_neigh + '_noise_mean'] = noise_mean
        features_values[num_neigh + '_noise_std'] = noise_std

    return features_values


def generate_features(x_train_data, y_train_data, x_test_data, y_test_data, neighbors_learners, to_csv):
    learners = dict()
    for n in neighbors_learners:
        # create n neigbors
        learners[str(n)] = NearestNeighbors(n_neighbors=n + 1,
                                            algorithm="ball_tree").fit(x_train_data)

    # add column to x_data df
    for train_feature_name, values in get_new_features(x_train_data, y_train_data, learners).items():
        x_train_data[train_feature_name] = values

    for test_feature_name, values in get_new_features(x_test_data, y_train_data, learners).items():
        x_test_data[test_feature_name] = values

    if to_csv:
        x_train_data.join(y_train_data).to_csv(
            const.csv_train_data, index=False)
        print('Train data saved at path {0}'.format(const.csv_train_data))

        x_test_data.join(y_test_data).to_csv(
            const.csv_test_data, index=False)

        print('Test data saved at path {0}'.format(const.csv_test_data))

    return x_train_data, x_test_data


def get_nn_dataset(x_train, x_test, y_train, y_test, scaler=None):
    ''' Return tuple (train dataset, test dataset)'''
    if scaler != None:
        scaled_x_train = scaler.transform(x_train)
        scaled_x_test = scaler.transform(x_test)

        return DatasetNN(
            scaled_x_train, y_train), DatasetNN(scaled_x_test, y_test)

    return DatasetNN(
        x_train, y_train), DatasetNN(x_test, y_test)


def plot_scores(x, y, hue, data, title):
    sns.barplot(x=x, y=y, hue=hue, data=data)

    if hue != None:
        plt.legend(loc='lower right')

    plt.ylabel(f"{y.replace('_', ' ')} value")
    plt.xlabel(x.replace('_', ' '))

    plt.title(title)


def plot_train_test(train_values, test_values, groups, score_type, title):
    fig = plt.figure()
    X = np.arange(0, len(groups)/2, 0.5)
    print(12)

    ax = fig.add_axes([0, 0, 1, 1])

    ax.bar(X + 0.00, train_values, width=0.20)
    ax.bar(X + 0.20, test_values, width=0.20)
    plt.xticks(X + 0.10, groups)

    ax.legend(labels=['Train', 'Test'])
    plt.xlabel('Type')
    plt.ylabel(score_type)

    plt.title(title)


def res_to_csv(data, path):
    if not os.path.exists(path):
        data.to_csv(path)
    else:
        data.to_csv(path, mode='a', header=False)
