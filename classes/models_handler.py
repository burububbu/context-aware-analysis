from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from classes.nn_classes import DatasetNN

import nn_utils as nn_utils

class ModelsHandler():

    def __init__(self, dataset):
        ''' Init instance properties'''
        self.models = {}
        self.subset_models = {}

        self.dataset = dataset

        # # dataset for neural network
        # self.train_dataset_nn = DatasetNN(dataset.x_train.to_numpy(), dataset.y_train)
        # self.test_dataset_nn = DatasetNN(dataset.x_test.to_numpy(), dataset.y_test)

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
        set_results = self._create_models(KNeighborsRegressor(), self.dataset.x_train, params)
        print('Subset...')
        subset_results = self._create_models(KNeighborsRegressor(), self.dataset.x_train_subset, params)

        self.models['knn'] =  max(set_results, key= lambda item: item[1])[2]
        self.subset_models['knn'] = max(subset_results, key= lambda item: item[1])[2]

    def create_sgd(self, params):
        print('SGD regressor...')
        print('Complete set ...')
        set_results = self._create_models(SGDRegressor(), self.dataset.x_train, params)
        print('Subset...')
        subset_results = self._create_models(SGDRegressor(), self.dataset.x_train_subset, params)
        
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

    def create_nn(self, params):

        # entire set
        print('ENTIRE DATASET...')
        standardized_x_train = self.standard_scaler.transform(self.dataset.x_train)
        standardized_x_test = self.standard_scaler.transform(self.dataset.x_test)

        train_data_nn = DatasetNN(standardized_x_train, self.dataset.y_train)
        test_data_nn =DatasetNN(standardized_x_test, self.dataset.y_test)
        
        nn_utils.create_nn_models(train_data_nn, test_data_nn, params)

        # subset
        print('ENTIRE SUB DATASET...')
        standardized_x_sub_train = self.standard_scaler_subset.transform(self.dataset.x_train_subset)
        standardized_x_sub_test = self.standard_scaler_subset.transform(self.dataset.x_test_subset)

        train_data_nn = DatasetNN(standardized_x_sub_train.to_numpy(), self.dataset.y_train)
        test_data_nn =DatasetNN(standardized_x_sub_test.to_numpy(), self.dataset.y_test)
        
        nn_utils.create_nn_models(train_data_nn, test_data_nn, params)


    def _create_models(self, model, x_data, params):
        results = []

        standard_data = None
        minmax_data = None

        if len(x_data.columns) == len(self.dataset.x_train.columns):
            standard_data = self.standard_scaler.transform(x_data)
            minmax_data = self.minmax_scaler.transform(x_data)
        else:
            standard_data = self.standard_scaler_subset.transform(x_data)
            minmax_data = self.minmax_scaler_subset.transform(x_data)
         
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

