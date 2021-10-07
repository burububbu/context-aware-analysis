import utils

import  matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.svm import SVR

from sklearn.linear_model import SGDRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler


knn_params = {
    'n_neighbors':[5, 10, 20, 30, 50, 60, 100, 150],
    'weights': ['uniform', 'distance'],
    # 'metric': ['haversine']
}

rf_params = {
    'n_estimators': [100,200, 300]
}

sm_params = {
    'C': [1, 5, 10, 50, 100, 100],
    'kernel': ['poly', 'rbf']
}
   
def db_scan(x_data, km_radius):
    kms_per_radian = 6371.0088
    epsilon = km_radius / kms_per_radian
   
    dbscan_cluster = DBSCAN(eps=epsilon, min_samples=4, algorithm='ball_tree', metric='haversine').fit(x_data)
    
    return dbscan_cluster

def agglomerative_cluster(x_data, km_radius):
    
    kms_per_radian = 6371.0088
    epsilon = km_radius / kms_per_radian

    agg_cluster = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=epsilon).fit(x_data)
    
    return agg_cluster 

def train_knn(x_train, y_train):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)

    model = KNeighborsRegressor()  

    reg = GridSearchCV(estimator = model, param_grid=knn_params)
    
    # reg.fit(x_train.values, y_train)
    reg.fit(x_train, y_train)
    

    print('Best KNN model with accuracy ' + str(reg.best_score_) + '. Params: ' + str(reg.best_params_))

    return model

def train_sgd(x_train, y_train):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)

    model = SGDRegressor()  

    reg = GridSearchCV(estimator = model, param_grid={})
    
    # reg.fit(x_train.values, y_train)
    reg.fit(x_train, y_train)

    print('Best sgd model with accuracy ' + str(reg.best_score_) + '. Params: ' + str(reg.best_params_))



def train_svr(x_train, y_train):
    

    model = SVR()  

    reg = GridSearchCV(estimator = model, param_grid=sm_params)
    
    # reg.fit(x_train.values, y_train)
    reg.fit(x_train, y_train)
    

    print('Best KNN model with accuracy ' + str(reg.best_score_) + '. Params: ' + str(reg.best_params_))

    return model

def train_rf(x_train, y_train):
   
    model = RandomForestRegressor()  

    reg = GridSearchCV(estimator = model, param_grid=rf_params)
    
    # reg.fit(x_train.values, y_train)
    reg.fit(x_train, y_train)
    

    print('Best random forest model with accuracy ' + str(reg.best_score_) + '. Params: ' + str(reg.best_params_))

    return model

def create_neigbors_learners(data, n_neigh):
    # create two learners neighbors
    learner = NearestNeighbors(n_neighbors= n_neigh +1, metric='haversine', algorithm="ball_tree")
    learner.fit(data)

    return learner 

def scatter_plot_kneigh(learner, data, point_index):
    point = data.iloc[point_index:point_index + 1, :] # i want it in the form of a dataframe (10th sample)
    
    # First array returned contains the distances to all points which are in the radius.
    # Second array contains their indices.
    distances, neighbors_indexes = learner.kneighbors(point) # tuple (array of arrays, array of arrays)
    
    colors = [0] * data.shape[0]
    for index in neighbors_indexes[0]:
        colors[index] = 1
    colors[point_index] = 2 # index of our point
    
    plt.scatter(data['longitude'], data['latitude'], c=colors)
    utils.plot('long (rad)', 'lat (rad)', "Neighbors of a sample", 'a_sample_neighbors')
    pass


def plots(data):
    plt.scatter(data['longitude'], data['latitude'])
    # plt.show()


       