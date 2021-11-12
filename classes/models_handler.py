from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

from math import sqrt

from classes.nn_classes import DatasetNN
from utils import utils
import utils.nn_utils as nn_utils
import pandas as pd

import os

from utils.visualization_utils import plot_loss


class ModelsHandler():

    def __init__(self, dataset):
        ''' Init instance properties'''

        self.dataset = dataset

        # dataframe [model_type, dataset_type, preprocessing_type, params, score]
        self.metrics_names = ['r2_train', 'r2_test', 'mse_train',
                              'mse_test', 'rmse_train', 'rmse_test', 'rae_train', 'rae_test']
        self.preprocessing_types = ['no_scaling',
                                    'standard_scaling', 'minmax_scaling']

        self.results_columns = [
            'model_type',
            'dataset_type',
            'preprocessing_type',
            'params',
            'mean_mse_epochs',
            'last_mse_epochs',
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

        if self.models.get(model_name) != None:

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

                estimators, metrics = self._train_models(
                    model_name,
                    train_data,
                    test_data,
                    params,
                    index_scaler)

                # add rows for result dataframe
                for prep_type in self.preprocessing_types:
                    list_metrics = [metrics[prep_type][metric_name]
                                    for metric_name in self.metrics_names]

                    results.append([
                        model_name,
                        set_type,
                        prep_type,
                        estimators[prep_type].get_params(),
                        *list_metrics])

            results_df = pd.DataFrame(results, columns=self.results_columns)

            # self.plot_results(data, x, y, hue, model_name)

            results_df.to_csv(f'./results/{model_name}_results.csv')

        else:
            print(f'ERROR: {model_name} doesn''t exist')

    def calc_models_statistics(self, estimator, x_train, x_test):
        ''' calculate metrics for each estimator passed.

            return metrics as a dicts
        '''
        predicted_train = estimator.predict(x_train)
        predicted_test = estimator.predict(x_test)

        train_score = estimator.score(x_train, self.dataset.y_train)
        test_score = estimator.score(x_test, self.dataset.y_test)

        return self._calc_statistics(train_score, test_score, predicted_train, predicted_test)

    def _calc_statistics(self, train_score, test_score, predicted_train, predicted_test):
        metrics = {
            'r2_train': train_score,
            'r2_test': test_score,

            'mse_train': mean_squared_error(self.dataset.y_train, predicted_train),
            'mse_test': mean_squared_error(self.dataset.y_test, predicted_test),

            'rae_train': utils.rae(self.dataset.y_train, predicted_train),
            'rae_test': utils.rae(self.dataset.y_test, predicted_test),
        }

        metrics['rmse_train'] = sqrt(metrics['mse_train'])
        metrics['rmse_test'] = sqrt(metrics['mse_test'])

        return metrics

    def create_neural_networks(self, params):
        print('Neural network ...')

        for index_scaler, set_type in enumerate(self.dataset.set_types):
            print(f'\t{set_type} set ...')

            results_df = pd.DataFrame()

            train_data = self.dataset.get_set(set_type, 'train')
            test_data = self.dataset.get_set(set_type, 'test')

            results = self._train_neural_networks(
                train_data, test_data, params, index_scaler, standard_scaling=False, no_scaling=True if set_type == 'base' else False)

            for prep_type in self.preprocessing_types:
                if results.get(prep_type) != None:

                    for result in results[prep_type]:
                        # plot the loss over epochs
                        plot_loss(result[1],  # mean mse
                                  result[0],  # params
                                  'mean_mse', set_type, prep_type)

                    # save in dataframe
                    temp_df = pd.DataFrame(
                        results[prep_type], columns=self.results_columns[3:])

                    temp_df['preprocessing_type'] = prep_type

                results_df = results_df.append(temp_df)

            results_df['dataset_type'] = set_type
            results_df['model_type'] = 'neural_network'

            path = f'./results/neural_network_{set_type}set.csv'

            file_exists = os.path.exists(path)

            # append to the existing file or create a new one
            results_df[self.results_columns].to_csv(
                path, mode='a' if file_exists else 'w', header=False if file_exists else True)

    def _train_models(self, model_name, x_train, x_test, params, scaler_index):
        '''
        Train a model three times, for each type of preprocessing available (no scaling, standard scaling, min max scaling).

        Return: {
            'no_scaling': estimator,
            'standard_scaling': estimator,
            'minmax_scaling': estimator,
        }, {
            'no_scaling': {metrics},
            'standard_scaling': {metrics},
            'minmax_scaling': {metrics},
        }

        '''
        models = {}
        metrics = {}

        standard_x_train = self.standard_scalers[scaler_index].transform(
            x_train)
        minmax_x_train = self.minmax_scalers[scaler_index].transform(x_train)

        standard_x_test = self.standard_scalers[scaler_index].transform(x_test)
        minmax_x_test = self.minmax_scalers[scaler_index].transform(x_test)

        # test without scaling
        best_params, best_score, models['no_scaling'] = self._cross_validating_model(
            self.models[model_name](), x_train.values, params)
        metrics['no_scaling'] = self.calc_models_statistics(
            models['no_scaling'], x_train.values, x_test.values)
        print(f'\t\t{best_score} score with params {best_params} and no scaling')

        # test with standard scaler
        best_params, best_score, models['standard_scaling'] = self._cross_validating_model(
            self.models[model_name](), standard_x_train, params)
        metrics['standard_scaling'] = self.calc_models_statistics(
            models['standard_scaling'], standard_x_train, standard_x_test)
        print(
            f'\t\t{best_score} score with params {best_params} and standard scaler scaling')

        # test with min max scaler
        best_params, best_score, models['minmax_scaling'] = self._cross_validating_model(
            self.models[model_name](), minmax_x_train, params)
        metrics['minmax_scaling'] = self.calc_models_statistics(
            models['minmax_scaling'], minmax_x_train, minmax_x_test)
        print(
            f'\t\t{best_score} score with params {best_params} and minmax scaler scaling')

        return models, metrics

    def _cross_validating_model(self, model, x_train_data, params):
        ''' Returns (best params, best r2 score, best estimator) '''

        reg = GridSearchCV(estimator=model, param_grid=params)
        reg.fit(x_train_data, self.dataset.y_train)

        return reg.best_params_, reg.best_score_, reg.best_estimator_

    def _train_neural_networks(self, x_data, x_test, params, index, standard_scaling=True, min_max_scaling=True, no_scaling=True):
        results = {}

        if no_scaling:
            # testing without min max
            print('\t\tNot scaled data')

            train_data_nn = DatasetNN(x_data.values, self.dataset.y_train)
            test_data_nn = DatasetNN(x_test.values, self.dataset.y_test)

            results['no_scaling'] = nn_utils.train_neural_networks(
                train_data_nn, test_data_nn, params)

        if standard_scaling:
            print('\t\tStandardized data')
            standardized_x_train = self.standard_scalers[index].transform(
                x_data)
            standardized_x_test = self.standard_scalers[index].transform(
                x_test)

            train_data_nn = DatasetNN(
                standardized_x_train, self.dataset.y_train)
            test_data_nn = DatasetNN(standardized_x_test, self.dataset.y_test)

            results['standard_scaling'] = nn_utils.train_neural_networks(
                train_data_nn, test_data_nn, params)

        if min_max_scaling:
            print('\t\tMinmax data')
            minmax_x_train = self.minmax_scalers[index].transform(x_data)
            minmax_x_test = self.minmax_scalers[index].transform(x_test)

            train_data_nn = DatasetNN(minmax_x_train, self.dataset.y_train)
            test_data_nn = DatasetNN(minmax_x_test, self.dataset.y_test)

            results['minmax_scaling'] = nn_utils.train_neural_networks(
                train_data_nn, test_data_nn, params)

        return results
