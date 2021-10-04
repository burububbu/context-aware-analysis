import  matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.cluster import DBSCAN, AgglomerativeClustering

knn_params = {
    'n_neighbors':[5, 10, 20, 30],
    'weights': ['uniform', 'distance'],
    'metric': ['haversine']
}

rf_params = {
    'n_estimators': range(100, 1000, 100)
}

sm_params = {
    'C': [1, 5, 10, 50, 100],
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
    model = KNeighborsRegressor()    
    reg = GridSearchCV(estimator = model, param_grid=knn_params)
    reg.fit(x_train.values, y_train)
    
    print('Best KNN model with accuracy ' + str(reg.best_score_) + '. Params: ' + str(reg.best_params_))

    return model

def plots(data):
    plt.scatter(data['longitude'], data['latitude'])
    plt.show()
       