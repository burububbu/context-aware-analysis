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

        # init scalers (shared among instances)
        self.standard_scalers = [
            StandardScaler().fit(dataset.x_train_base),
            StandardScaler().fit(dataset.x_train),
            StandardScaler().fit(dataset.x_train_subset)
            ]

        self.minmax_scalers = [
            MinMaxScaler().fit(dataset.x_train_base),
            MinMaxScaler().fit(dataset.x_train),
            MinMaxScaler().fit(dataset.x_train_subset),
            ]

        self.models = {
            'knn': KNeighborsRegressor,
            'sgd': SGDRegressor,
            'rf': RandomForestRegressor
        }
    
    def create_models(self, model_name, params):
        
        if self.models[model_name] != None:         
            print(f'{model_name} regressor...')
            
            print('\tBase set ...')
            self._train_models(self.models[model_name](), self.dataset.x_train_base, params, 0)

            print('\tComplete set ...')
            self._train_models(self.models[model_name](), self.dataset.x_train, params, 1)
            
            print('\tSubset ...')
            self._train_models(self.models[model_name](), self.dataset.x_train_subset, params, 2)
        else:
            print(f'ERROR: {model_name} doesn''t exist')

    def create_neural_networks(self, params):
        print('Neural network ...')
    
        print('\tBase set...')
        self._train_neural_networks(self.dataset.x_train_base, self.dataset.x_test_base, params, 0)
        
        print('\tComplete set...')
        self._train_neural_networks(self.dataset.x_train, self.dataset.x_test, params, 1)
        
        print('\tSubset...')
        self._train_neural_networks(self.dataset.x_train_subset, self.dataset.x_test_subset, params, 2)

    def _train_models(self, model, x_data, params, scaler_index):
        results = []

        standard_data = self.standard_scalers[scaler_index].transform(x_data)
        minmax_data = self.minmax_scalers[scaler_index].transform(x_data)
         
        # test without scaling
        results.append(self._cross_validating_model(model, x_data.values, params))
        print('\t\t{0} score with params {1} and no scaling'.format(results[-1][1], results[-1][0]))
        
        # test with standard scaler
        results.append(self._cross_validating_model(model, standard_data, params))
        print('\t\t{0} score with params {1} and standard scaler scaling'.format(results[-1][1], results[-1][0]))

        # test with min max scaler
        results.append(self._cross_validating_model(model, minmax_data, params))
        print('\t\t{0} score with params {1} and minmax scaler scaling'.format(results[-1][1], results[-1][0]))

        return results

    def _cross_validating_model(self, model, x_train_data, params):
        ''' Returns the best model '''

        reg = GridSearchCV(estimator = model, param_grid=params)
        reg.fit(x_train_data, self.dataset.y_train)
        
        return reg.best_params_, reg.best_score_, reg.best_estimator_
         
    def _train_neural_networks(self, x_data, x_test, params, index):

        standardized_x_train = self.standard_scalers[index].transform(x_data)
        standardized_x_test = self.standard_scalers[index].transform(x_test)

        train_data_nn = DatasetNN(standardized_x_train, self.dataset.y_train)
        test_data_nn =DatasetNN(standardized_x_test, self.dataset.y_test)
        
        nn_utils.train_neural_networks(train_data_nn, test_data_nn, params)


