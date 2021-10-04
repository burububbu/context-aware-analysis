
from numpy.core.fromnumeric import shape
from sklearn import neighbors
import models
import numpy as np
import pandas as pd

import  matplotlib.pyplot as plt
import database.database as db

from dotenv import load_dotenv, find_dotenv
from sklearn.neighbors import NearestNeighbors


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
    plot('long (rad)', 'lat (rad)', "Samples", 'samples_scatter')
    
    # 3. create learner for neighbor searches -> add average noise and std in area ( radius -> 10 mt, 50 mt, 100mt)?
    kms_per_radian = 6371.000
    radius = 0.015 / kms_per_radian # 1.5 km of radius

    x_data = data[['latitude', 'longitude']]

    # create learner neighbors
    n_learner = NearestNeighbors(n_neighbors= 5, radius=radius, metric='haversine', algorithm="ball_tree")
    n_learner.fit(x_data)

    # -.-.-.-.-.-.-. an example and relatives plot -.-.-.-.-.-.-.
    # example point
    point_index = 10
    point = x_data.iloc[point_index:point_index + 1, :] # i want it in the form of a dataframe (10th sample)
    
    # First array returned contains the distances to all points which are in the radius.
    # Second array contains their indices.
    distances, neighbors_indexes = n_learner.kneighbors(point) # tuple (array of arrays, array of arrays)
    
    colors = [0] * x_data.shape[0]

    for index in neighbors_indexes[0]:
        colors[index] = 1
   
    colors[point_index] = 2 # index of our point
    
    plt.scatter(data['longitude'], data['latitude'], c=colors)
    plot('long (rad)', 'lat (rad)', "Neighbors of a sample", 'a_sample_neighbors')


    # 2. CLUSTERING
    #   a. DBSCAN is a good idea
    #   b. SEARCH FOR THE ZONE where the point is
    # reverse Geocoding

def plot(x_label, y_label, title, file_name):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.savefig("./plots/{}.png".format(file_name))
    plt.show()
    plt.clf()
