from math import sqrt
from classes.nn_classes import NeuralNet
from sklearn.metrics import r2_score, mean_squared_error

from torch.utils.data import DataLoader

import torch.nn as nn
import torch

import utils.utils as utils
import itertools


def train_neural_networks(train_dataset, test_dataset, params):
    torch.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # generate all possible combinations
    hyperparams = itertools.product(params['hidden_sizes'], params['nums_layers'],
                                    params['num_epochs'], params['batch_sizes'], params['learning_rates'])

    num_combinations = len(params['hidden_sizes']) * len(params['nums_layers']) * len(
        params['num_epochs']) * len(params['batch_sizes']) * len(params['learning_rates'])

    loss = nn.MSELoss()

    results = []

    # hyperparams tuning
    for i, (hidden_size, num_layers, num_epoch, batch_size, learning_rate) in enumerate(hyperparams):
        print(
            f'\t\t{i+1} / {num_combinations} ({round((i+1)/num_combinations*100,3)}%)')
        print(
            f'\t\tTraining model with {hidden_size} hidden size, {num_layers+1 } layers, {num_epoch} epochs, {learning_rate} learning rate and {batch_size} as batch size')

        # create model
        model = NeuralNet(
            train_dataset.X.shape[1], hidden_size, num_layers).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # create Dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        mean_mse_epochs = []
        last_mse_epochs = []

        # train
        for _ in range(num_epoch):
            mean_mse_epoch, last_mse_epoch = _train(train_dataloader, model, loss,
                                                    optimizer, device)

            mean_mse_epochs.append(mean_mse_epoch)
            last_mse_epochs.append(last_mse_epoch)

        tr_m = compute_metrics(train_dataloader, model, device)
        te_m = compute_metrics(test_dataloader, model, device)

        print('\t\t\tTrain loss: {}'.format(tr_m['mse']))
        print('\t\t\tTest loss: {}'.format(te_m['mse']))
        print('\t\t\tR2 train score: {}'.format(tr_m['r2']))
        print('\t\t\tR2 test score: {}'.format(te_m['r2']))

        # construct rows
        params = {
            'hs': hidden_size,
            'n_layers': num_layers+1,
            'num_epochs': num_epoch,
            'batch_size': batch_size,
            'lr': learning_rate
        }

        results.append([
            params,
            mean_mse_epochs, last_mse_epochs,
            tr_m['r2'], te_m['r2'],
            tr_m['mse'], te_m['mse'],
            tr_m['rmse'], te_m['rmse'],
            tr_m['rae'], te_m['rae']])

    return results


def compute_metrics(dataloader, model, device):
    ''' Return dics containing mse, rmse,  r2 and rae'''
    model.eval()

    with torch.no_grad():
        x_data = dataloader.dataset.X.to(device)

        preds = model(x_data)  # all data
        preds = preds.detach().cpu().numpy()

        y_data = dataloader.dataset.y.detach().cpu().numpy()

    mse = mean_squared_error(y_data, preds)

    return {
        'mse': mse,
        'rmse': sqrt(mse),
        'r2': r2_score(y_data, preds),
        'rae': utils.rae(y_data, preds)
    }


def _train(dataloader, model, loss_fn, optimizer, device):
    ''' Return mean loss over minibatch, loss of the last minibatch'''
    model.train()  # set model model

    num_batches = len(dataloader)
    total_loss = 0
    last_loss = 0

    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)

        # compute prediction error
        prediction = model(data)
        loss = loss_fn(prediction.flatten(), targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        last_loss = loss.item()

    return total_loss/num_batches, last_loss


def _test(dataloader, model, loss_fn, device):
    model.eval()

    num_batches = len(dataloader)
    total_loss = 0

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)

            prediction = model(data)
            loss = loss_fn(prediction.flatten(), targets)
            total_loss += loss.item()

    return total_loss/num_batches
