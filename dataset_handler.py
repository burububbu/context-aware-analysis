# import pandas as pd

# import  matplotlib.pyplot as plt
# import database.database as db

# from dotenv import load_dotenv, find_dotenv

# def create_csv():
#     load_dotenv(find_dotenv())
    
#     results = db.getAll()

#     longs = []
#     lats = []
#     noises = []
 
#     for res in results:
#         longs.append(res.longitude)
#         lats.append(res.latitude)
#         noises.append(res.noise)

#     data = pd.DataFrame({'latitude':lats, 'longitude': longs, 'noise':noises})
#     data.to_csv("data/noises.csv")

# # def create_subsets(data):
#     ''' Create subsets based on clustering'''
#     km_threshold = np.inf

#     # cluster = models.agglomerative_cluster(data[['latitude', 'longitude']], km_threshold)
#     cluster = models.db_scan(data[['latitude', 'longitude']], 3)
#     plt.scatter(data['latitude'], data['longitude'], c=cluster.labels_)


#     print('{0} clusters found.'.format(len(np.unique(cluster.labels_))))
    
#     data['cluster'] = cluster.labels_

#     subsets = []

#     for cluster_index in np.unique(cluster.labels_):
#         subsets.append(data[data['cluster'] == cluster_index])

#     print(np.unique(cluster.labels_))
#     print(data.shape)

#     plt.show()
    
#     return(subsets)
