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
        self.subset_models = {}

        self.dataset = dataset

        # init scalers (shared among instances)
        self.standard_scaler = StandardScaler()
        self.standard_scaler.fit(dataset.x_train)

        self.minmax_scaler = MinMaxScaler()
        self.minmax_scaler.fit(dataset.x_train)

        self.standard_scaler_subset = StandardScaler()
        self.standard_scaler_subset.fit(dataset.x_train_subset)

        self.minmax_scaler_subset = MinMaxScaler()
        self.minmax_scaler_subset.fit(dataset.x_train_subset)

    def create_knn(self, params):
        print('KNN regressor...')
        print('Complete set ...')
        set_results = self._test_models(KNeighborsRegressor(), self.dataset.x_train, params)
        print('Subset...')
        subset_results = self._test_models(KNeighborsRegressor(), self.dataset.x_train_subset, params)

        self.models['knn'] =  max(set_results, key= lambda item: item[1])[2]
        self.subset_models['knn'] = max(subset_results, key= lambda item: item[1])[2]

    def create_sgd(self, params):
        print('SGD regressor...')
        print('Complete set ...')
        set_results = self._test_models(SGDRegressor(), self.dataset.x_train, params)
        print('Subset...')
        subset_results = self._test_models(SGDRegressor(), self.dataset.x_train_subset, params)
        
        self.models['sgd'] =  max(set_results, key= lambda item: item[1])[2]
        self.subset_models['sgd'] = max(subset_results, key= lambda item: item[1])[2]

    
    def create_rf(self, params):
        print('Random forest regressor...')
        print('Complete set...')
        set_results = self._train_model(RandomForestRegressor(), self.dataset.x_train, params)
        print('\t{0} score with params {1} and no scaling'.format(set_results[-1][1], set_results[-1][0]))
        
        print('Subset...')
        set_results = self._train_model(RandomForestRegressor(), self.dataset.x_train_subset, params)
        print('\t{0} score with params {1} and no scaling'.format(set_results[-1][1], set_results[-1][0]))



    def _test_models(self, model, x_data, params):
        standard_data = None
        minmax_data = None

        if len(x_data.columns) == len(self.dataset.x_train.columns):
            print(len(x_data.columns), len(self.dataset.x_train.columns))
            standard_data = self.standard_scaler.transform(x_data)
            minmax_data = self.minmax_scaler.transform(x_data)
        else:
            standard_data = self.standard_scaler_subset.transform(x_data)
            minmax_data = self.minmax_scaler_subset.transform(x_data)
         
        results = []

        # test without scaling
        results.append(self._train_model(model, x_data.values, params))
        print('\t{0} score with params {1} and no scaling'.format(results[-1][1], results[-1][0]))
        
        # test with standard scaler
        results.append(self._train_model(model, standard_data, params))
        print('\t{0} score with params {1} and standard scaler scaling'.format(results[-1][1], results[-1][0]))

        # test with min max scaler
        results.append(self._train_model(model, minmax_data, params))
        print('\t{0} score with params {1} and minmax scaler scaling'.format(results[-1][1], results[-1][0]))

        return results

    def _train_model(self, model, x_train_data, params):
        ''' Returns the best model '''

        reg = GridSearchCV(estimator = model, param_grid=params)
        reg.fit(x_train_data, self.dataset.y_train)
        
        return reg.best_params_, reg.best_score_, reg.best_estimator_


        

