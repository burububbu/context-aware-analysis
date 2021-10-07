import models
import utils

import numpy as np
import pandas as pd

import  matplotlib.pyplot as plt
import database.database as db

from dotenv import load_dotenv, find_dotenv


def create_csv():
    load_dotenv(find_dotenv())
    
    results = db.getAll()

    longs = []
    lats = []
    noises = []
 
    for res in results:
        longs.append(res.longitude)
        lats.append(res.latitude)
        noises.append(res.noise)

    data = pd.DataFrame({'longitude': longs, 'latitude':lats, 'noise':noises})
    data.to_csv("data/noises.csv")

def create_subsets(data):
    ''' Create subsets based on clustering'''
    km_threshold = np.inf

    # cluster = models.agglomerative_cluster(data[['latitude', 'longitude']], km_threshold)
    cluster = models.db_scan(data[['latitude', 'longitude']], 3)
    plt.scatter(data['latitude'], data['longitude'], c=cluster.labels_)


    print('{0} clusters found.'.format(len(np.unique(cluster.labels_))))
    
    data['cluster'] = cluster.labels_

    subsets = []

    for cluster_index in np.unique(cluster.labels_):
        subsets.append(data[data['cluster'] == cluster_index])

    print(np.unique(cluster.labels_))
    print(data.shape)

    plt.show()
    
    return(subsets)


# step to do also  for each new data
def features_engineering(data):
    # is this useful? - - - - 
    data = data.reset_index()
    data = data.drop(columns='index')
    # - - - - - - - - - - - -
    # change column order [latitude, long, noise] -> necessary for the haversine formula
    data = data.reindex(columns=['latitude', 'longitude', 'noise'])
    
    # 1. lat and long in radiants
    data['latitude'] = data['latitude'].map(np.radians) 
    data['longitude'] = data['longitude'].map(np.radians) 

    # 2. round noises values
    data['noise'] = data['noise'].apply(lambda value: round(value,5))

    # Plot scatter plot of samples 
    # long is x and lat is y
    plt.scatter(data['longitude'], data['latitude'])
    utils.plot('long (rad)', 'lat (rad)', "Samples", 'samples_scatter')
    
    # 3. create learner for neighbor searches -> add average noise and std in area ( radius -> 10 mt, 50 mt, 100mt)?
    kms_per_radian = 6371.000
    radius = 0.015 / kms_per_radian # 1.5 km of radius

    x_data = data[['latitude', 'longitude']]

    learner_5_nearest = models.create_neigbors_learners(x_data, 5)
    learner_10_nearest = models.create_neigbors_learners(x_data, 10)

    # two examples
    # models.scatter_plot_kneigh(learner_1_nearest, x_data, 10)
    # models.scatter_plot_kneigh(learner_10_nearest, x_data, 10)

    # compute some adding features
    nearest_5_points_noise_mean = []
    nearest_5_points_noise_std = []
    nearest_5_points_distance_mean = []
    nearest_5_points_distance_std = []

    nearest_10_points_noise_mean = []
    nearest_10_points_noise_std = []
    nearest_10_points_distance_mean = []
    nearest_10_points_distance_std = []

    print(x_data.shape)

    for index in range(x_data.shape[0]): # for each row
        coords = x_data.iloc[index:index + 1, :]
        
        # ------------------ five neighbors ---------------
        distances, neighbors_indexes = learner_5_nearest.kneighbors(coords) # tuple (array of arrays, array of arrays)

        # get first query results
        distances = distances[0]
        neighbors_indexes = neighbors_indexes[0]

        nearest_5_points_distance_mean.append(np.mean(distances))
        nearest_5_points_distance_std.append(np.std(distances))

        noises = data.loc[neighbors_indexes]['noise']

        nearest_5_points_noise_mean.append(np.mean(noises))
        nearest_5_points_noise_std.append(np.std(noises))

        # # get first query results
        # distances = distances[0]
        # neighbors_indexes = neighbors_indexes[0]

        # i_nearest_sample = None # array
        # index_nearest_sample = None # dataframe

        # if index not in neighbors_indexes:
        #     index_nearest_sample = 0
        # else:
        #     for i, neigh_index in enumerate(neighbors_indexes):
        #         if index != neigh_index:
        #             i_nearest_sample = i
        #             index_nearest_sample = neigh_index
        
        # nearest_point_distance.append(distances[i_nearest_sample])
        # nearest_point_noise.append(data.iloc[index_nearest_sample]['noise'])

        # ------------------ ten neighbors ---------------
        distances, neighbors_indexes = learner_10_nearest.kneighbors(coords) # tuple (array of arrays, array of arrays)

        # get first query results
        distances = distances[0]
        neighbors_indexes = neighbors_indexes[0]

        nearest_10_points_distance_mean.append(np.mean(distances))
        nearest_10_points_distance_std.append(np.std(distances))

        noises = data.loc[neighbors_indexes]['noise']

        nearest_10_points_noise_mean.append(np.mean(noises))
        nearest_10_points_noise_std.append(np.std(noises))

    x_data['nearest_5_points_distance_mean'] =  nearest_5_points_distance_mean
    x_data['nearest_5_points_distance_std'] =  nearest_5_points_distance_std

    x_data['nearest_5_points_noise_mean'] = nearest_5_points_noise_mean
    x_data['nearest_5_points_noise_std'] =  nearest_5_points_noise_std
    
    x_data['nearest_10_points_distance_mean'] =  nearest_10_points_distance_mean
    x_data['nearest_10_points_distance_std'] =  nearest_10_points_distance_std

    x_data['nearest_10_points_noise_mean'] = nearest_10_points_noise_mean
    x_data['nearest_10_points_noise_std'] =  nearest_10_points_noise_std

    df_to_save = x_data.assign(noise=data['noise'])
    df_to_save.to_csv('./data/noises_features')

    return x_data, data['noise']
