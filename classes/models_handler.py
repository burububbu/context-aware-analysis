from math import sqrt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import dataset
from sklearn.metrics import mean_squared_error

from classes.nn_classes import DatasetNN
from utils import utils
from itertools import chain

import utils.nn_utils as nn_utils 
import utils.visualization_utils as visualization_utils

import pandas as pd

class ModelsHandler():

    def __init__(self, dataset):

        ''' Init instance properties'''

        self.dataset = dataset

        # dataframe [model_type, dataset_type, preprocessing_type, params, score]
        self.metrics_names = ['r2_train','r2_test', 'mse_train', 'mse_test', 'rmse_train', 'rmse_test', 'rae_train', 'rae_test']
        self.preprocessing_types = ['no_scaling', 'standard_scaling', 'minmax_scaling']

        self.results_columns = [
            'model_type',
            'dataset_type',
            'preprocessing_type',
            'params',
            *self.metrics_names
            ]
        
        
        # init scalers (shared among instances)
        self.standard_scalers = [
            StandardScaler().fit(dataset.get_set('base', 'train')),
            StandardScaler().fit(dataset.get_set('complete', 'train')),
            StandardScaler().fit(dataset.get_set('sub', 'train'))
            ]

        self.minmax_scalers = [
            MinMaxScaler().fit(dataset.get_set('base', 'train')),
            MinMaxScaler().fit(dataset.get_set('complete', 'train')),
            MinMaxScaler().fit(dataset.get_set('sub', 'train')),
            ]

        self.models = {
            'knn': KNeighborsRegressor,
            'sgd': SGDRegressor,
            'rf': RandomForestRegressor
        }

    def create_models_sets(self, model_name, params):
        ''' Create and train models for each type of set / subset available.
            Sets: Base set (Lat and Long), Complete set, Subset of 5 feat.

            Also saves results for each type of model in a csv file and plor results

        '''
        
        if self.models[model_name] != None:
            
            results = []

            print(f'{model_name} regressor...')
            
            for index_scaler, set_type in enumerate(self.dataset.set_types):
                print(f'\t{set_type} set ...')    
                
                train_data = self.dataset.get_set(set_type, 'train')
                test_data = self.dataset.get_set(set_type, 'test')

                # the form is 
                # {
                #     'no_scaling': estimator,
                #     'standard_scaling': estimator,
                #     'minmax_scaling': estimator,
                # }

                estimators = self._train_models(
                    self.models[model_name](),
                    train_data,
                    params,
                    index_scaler)
                
                metrics = self.calc_statistics(estimators, train_data, test_data)

                # add rows for result dataframe
                for prep_type in self.preprocessing_types:
                    list_metrics = [metrics[prep_type][metric_name] for metric_name in self.metrics_names]
                    results.append([
                        model_name,
                        set_type,
                        prep_type,
                        estimators[prep_type].get_params(),
                        *list_metrics])

            results_df = pd.DataFrame(results, columns= self.results_columns)
            
            # self.plot_results(data, x, y, hue, model_name)

            results_df.to_csv(f'./results/{model_name}_results.csv')
            
            visualization_utils.plot_models_results()

        else:
            print(f'ERROR: {model_name} doesn''t exist')

    def calc_statistics(self, models, x_train, x_test):
        ''' calculate metrics for each estimator passed.
        
            return {preprocessing_type1 :  metrics, preprocessing_type2 :  metrics}
        '''

        all_metrics = {}

        for preprocessing_type in self.preprocessing_types:
            estimator = models[preprocessing_type] # get model with specific type of preprocessing (es. no_scalinf)

            # also preprocess these data
            predicted_train = estimator.predict(x_train.values)
            predicted_test = estimator.predict(x_test.values)

            metrics = {
            'r2_train': estimator.score(x_train.values, self.dataset.y_train),
            'r2_test': estimator.score(x_test.values, self.dataset.y_test),  

            'mse_train': mean_squared_error(self.dataset.y_train, predicted_train),
            'mse_test': mean_squared_error(self.dataset.y_test, predicted_test ),

            'rae_train': utils.rae(self.dataset.y_train, predicted_train),
            'rae_test': utils.rae(self.dataset.y_test, predicted_test ),
            }
        
            metrics['rmse_train'] = sqrt(metrics['mse_train']) 
            metrics['rmse_test'] = sqrt(metrics['mse_test'])  

            all_metrics[preprocessing_type] = metrics

        return all_metrics
     

    def create_neural_networks(self, params):
        print('Neural network ...')
    
        print('\tBase set...')
        self._train_neural_networks(self.dataset.x_train_base, self.dataset.x_test_base, params, 0)
        
        print('\tComplete set...')
        self._train_neural_networks(self.dataset.x_train, self.dataset.x_test, params, 1)
        
        print('\tSubset...')
        self._train_neural_networks(self.dataset.x_train_subset, self.dataset.x_test_subset, params, 2)

    def _train_models(self, model, x_data, params, scaler_index):
        '''
        Train a model three times, for each type of preprocessing available (no scaling, standard scaling, min max scaling).
        
        Return: {
            'no_scaling': estimator,
            'standard_scaling': estimator,
            'minmax_scaling': estimator,
        }
        
        '''
        models = {}

        standard_data = self.standard_scalers[scaler_index].transform(x_data)
        minmax_data = self.minmax_scalers[scaler_index].transform(x_data)
         
        # test without scaling
        best_params, best_score, models['no_scaling'] = self._cross_validating_model(model, x_data.values, params)
        print(f'\t\t{best_score} score with params {best_params} and no scaling')
        
        # test with standard scaler
        best_params, best_score, models['standard_scaling'] = self._cross_validating_model(model, standard_data, params)
        print(f'\t\t{best_score} score with params {best_params} and standard scaler scaling')

        # test with min max scaler
        best_params, best_score, models['minmax_scaling'] = self._cross_validating_model(model, minmax_data, params)
        print(f'\t\t{best_score} score with params {best_params} and minmax scaler scaling')

        return models

    def _cross_validating_model(self, model, x_train_data, params):
        ''' Returns (best params, best r2 score, best estimator) '''

        reg = GridSearchCV(estimator = model, param_grid=params)
        reg.fit(x_train_data, self.dataset.y_train)
        
        return reg.best_params_, reg.best_score_, reg.best_estimator_
         
    def _train_neural_networks(self, x_data, x_test, params, index):

        standardized_x_train = self.standard_scalers[index].transform(x_data)
        standardized_x_test = self.standard_scalers[index].transform(x_test)

        train_data_nn = DatasetNN(standardized_x_train, self.dataset.y_train)
        test_data_nn =DatasetNN(standardized_x_test, self.dataset.y_test)
        
        nn_utils.train_neural_networks(train_data_nn, test_data_nn, params)


