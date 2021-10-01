from sklearn import cluster
import models
import database.database as db
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np

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
    ''' Create subsets base on DBSCAN clustering'''

    dbscan_cluster = models.db_scan(data[['latitude', 'longitude']], 0.2)

    plt.scatter(data['latitude'], data['longitude'], c=dbscan_cluster.labels_)
    
    print('{0} clusters found.'.format(len(np.unique(dbscan_cluster.labels_))))
    
    data['cluster'] = dbscan_cluster.labels_

    subsets = []

    for cluster_index in np.unique(dbscan_cluster.labels_):
        subsets.append(data[data['cluster'] == cluster_index])
    
    plt.show()
    
    return(subsets)



    




# step to do also  for each new data
def extract_features(data):
        pass
    
    # some ideas
    # 1. Use long and lat as they are.
    #   a. round at less decimals NO
    #   b. convert in radians



    # 2. CLUSTERING
    #   a. DBSCAN is a good idea
    #   b. SEARCH FOR THE ZONE where the point is
    # reverse Geocoding


