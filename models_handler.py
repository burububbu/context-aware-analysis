# handle of training models
import utils

import  matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import SGDRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler

class ModelsHandler():

    def __init__(self, dataset):
        self.models = {}

        self.dataset = dataset

        # init scalers (shared among instances)
        self.standard_scaler = StandardScaler()
        self.standard_scaler.fit(dataset.x_train)

        self.minmax_scaler = MinMaxScaler()
        self.minmax_scaler.fit(dataset.x_train)

    def create_knn(self, params):
        results = []

        model = KNeighborsRegressor()
        
        print('KNN regressor...')
        # test without scaling
        results.append(self._train_model(model, self.dataset.x_train.values, params))
        print('\t{0} score with params {1} and no scaling'.format(results[-1][1], results[-1][0]))
        
        # test with standard scaler
        results.append(self._train_model(model, self.standard_scaler.transform(self.dataset.x_train), params))
        print('\t{0} score with params {1} and standard scaler scaling'.format(results[-1][1], results[-1][0]))

        # test with min max scaler
        results.append(self._train_model(model, self.minmax_scaler.transform(self.dataset.x_train), params))
        print('\t{0} score with params {1} and standard scaler scaling'.format(results[-1][1], results[-1][0]))

        self.models['knn'] =  max(results, key= lambda item: item[1])[2]

    def create_sgd(self, params):
        results = []

        # model = SGDRegressor()
        model = GradientBoostingRegressor()
        
        print('SGD regressor...')
        # test without scaling
        results.append(self._train_model(model, self.dataset.x_train.values, params))
        print('\t{0} score with params {1} and no scaling'.format(results[-1][1], results[-1][0]))
        
        # test with standard scaler
        results.append(self._train_model(model, self.standard_scaler.transform(self.dataset.x_train), params))
        print('\t{0} score with params {1} and standard scaler scaling'.format(results[-1][1], results[-1][0]))

        # test with min max scaler
        results.append(self._train_model(model, self.minmax_scaler.transform(self.dataset.x_train), params))
        print('\t{0} score with params {1} and standard scaler scaling'.format(results[-1][1], results[-1][0]))

        self.models['SGD'] =  max(results, key= lambda item: item[1])[2]

    
    def create_rf(self, params):
        model = RandomForestRegressor()

        reg = GridSearchCV(estimator = model, param_grid=params)
        reg.fit(self.dataset.x_train, self.dataset.y_train)

        print('{0} score with params {1} and no scaling'.format(reg.best_score_, reg.best_params_))

        self.models['random_forest'] = reg.best_estimator_ 

        print(self.models['random_forest'].feature_importances_)
        


    def _train_model(self, model, x_train_data, params):
        ''' Returns the best model '''

        reg = GridSearchCV(estimator = model, param_grid=params)
        reg.fit(x_train_data, self.dataset.y_train)
        
        return reg.best_params_, reg.best_score_, reg.best_estimator_


        

