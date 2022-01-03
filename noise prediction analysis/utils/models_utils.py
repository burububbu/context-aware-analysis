import itertools
import torch

import torch.nn as nn
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader
from math import sqrt

from utils.nn_classes import NeuralNet


def train_models(model, x_train, y_train, x_test, y_test, hyperparams, standard_scaler, minmax_scaler):

    methods = ["no", "standard", "minmax"]

    train_data = [x_train.values, standard_scaler.transform(
        x_train), minmax_scaler.transform(x_train)]

    test_data = [x_test.values, standard_scaler.transform(
        x_test), minmax_scaler.transform(x_test)]

    to_build = []

    for method, x_tr, x_te in zip(methods, train_data, test_data):
        to_build.append(_get_model(model, x_tr, y_train,
                        x_te, y_test, hyperparams))

        to_build[-1]["preprocessing"] = method

    return pd.DataFrame(to_build)


def _get_model(model, x_train, y_train, x_test, y_test, hyperparams):

    reg = GridSearchCV(estimator=model, param_grid=hyperparams)
    reg.fit(x_train, y_train)

    estimator = reg.best_estimator_

    # calc stats
    train_mse = mean_squared_error(
        y_train, estimator.predict(x_train))

    test_mse = mean_squared_error(y_test, estimator.predict(x_test))

    return {
        'r2_train': estimator.score(x_train, y_train),
        'r2_test': estimator.score(x_test, y_test),
        'mse_train': train_mse,
        'mse_test': test_mse,
        'rmse_train': sqrt(train_mse),
        'rmse_test': sqrt(test_mse),
        'params': reg.best_params_
    }


def train_neural_nets(train_dataset, test_dataset, params):
    ''' Returns dataframe  containing metrics for each trained neural net (according to the passed params)'''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # generate all possible combinations
    hyperparams = itertools.product(params['hidden_sizes'], params['nums_layers'],
                                    params['num_epochs'], params['batch_sizes'], params['learning_rates'],
                                    params['gamma'], params['dropout'])

    # num_combinations = reduce(lambda x, y: len(x)*len(y), params.values())
    num_combinations = len(params['hidden_sizes']) * len(params['nums_layers']) * len(
        params['num_epochs']) * len(params['batch_sizes']) * len(params['learning_rates']) * len(params['gamma']) * len(params['dropout'])

    results = []

    print(f'{num_combinations} neural net to train')
    for i, (hidden_size, num_layers, epoch, batch_size, l_rate, gamma, dropout) in enumerate(hyperparams):
        # print the advancement

        # create model, optimizer and specify loss
        model = NeuralNet(
            train_dataset.X.shape[1], hidden_size, num_layers, dropout).to(device)

        optimizer = torch.optim.SGD(
            model.parameters(), lr=l_rate)

        loss = nn.MSELoss()

        # def scheduler for learning rate decay
        def lambda1(epoch): return 1 / (1 + gamma * epoch)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda1)

        # create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        # save mean mse (and las batch mse) for each epoch
        mean_mse_epochs = []
        last_mse_epochs = []

        # train phase
        for _ in range(epoch):
            # do a training step
            mean_mse_epoch, last_mse_epoch = _train(train_dataloader, model, loss,
                                                    optimizer, device)

            # decrease the learning rate after each epoch
            scheduler.step()

            mean_mse_epochs.append(mean_mse_epoch)
            last_mse_epochs.append(last_mse_epoch)

        # test phase (get metrics also for train set)
        train_res = _test(train_dataloader, model, device, "train")
        test_res = _test(test_dataloader, model, device, "test")

        params = {
            'hs': hidden_size,
            'n_layers': num_layers+1,
            'n_epochs': epoch,
            'b_size': batch_size,
            'l_rate': l_rate,
            'gamma': gamma,
            'dropout': dropout
        }

        res = {**train_res, **test_res}
        res['params'] = params
        res['mean_ms_epochs'] = mean_mse_epochs
        res['last_mse_epochs'] = last_mse_epochs

        results.append(res)

        print(
            f'\t{i+1} / {num_combinations} ({round((i+1)/num_combinations*100,3)}%): R2 train: {res["r2_train"]}, R2 test:{res["r2_test"]}')

    return pd.DataFrame(results)


def _train(dataloader, model, loss_fn, optimizer, device):
    ''' Return mean loss over minibatch, loss of the last minibatch'''
    model.train()  # set model model

    num_batches = len(dataloader)
    total_loss = 0
    last_loss = 0

    for data, targets in dataloader:
        optimizer.zero_grad()

        data, targets = data.to(device), targets.to(device)

        # compute prediction error
        prediction = model(data)
        loss = loss_fn(prediction.flatten(), targets)

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        last_loss = loss.item()

    return total_loss/num_batches, last_loss


def _test(dataloader, model, device, type):
    ''' Return dict containing mse, rmse, r2 '''

    model.eval()

    with torch.no_grad():
        x_data = dataloader.dataset.X.to(device)

        preds = model(x_data)  # all data
        preds = preds.detach().cpu().numpy()

        y_data = dataloader.dataset.y.detach().cpu().numpy()

    # calc metrics
    mse = mean_squared_error(y_data, preds)

    return {
        f'mse_{type}': mse,
        f'rmse_{type}': sqrt(mse),
        f'r2_{type}': r2_score(y_data, preds),
    }
