from math import sqrt
from classes.nn_classes import NeuralNet
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

import torch.nn as nn
import torch

import utils.utils as utils
import itertools

def train_neural_networks(train_dataset, test_dataset, params):
    torch.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training Neural Network (using {device} device)...')

    # generate all possible combinations 
    hyperparams = itertools.product(params['hidden_sizes'], params['nums_layers'], params['num_epochs'], params['batch_sizes'], params['learning_rates'])

    loss = nn.MSELoss()

    results = []
    # hyperparams tuning
    for hidden_size, num_layers, num_epoch, batch_size, learning_rate in hyperparams:
        print(f'\t\tTraining model with {hidden_size} hidden size, {num_layers} layers, {num_epoch} epochs, {learning_rate} learning rate and {batch_size} as batch size')

        # create model
        model = NeuralNet(train_dataset.X.shape[1], hidden_size, num_layers).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        # create Dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        mse_train = None

        #train
        for epoch in range(num_epoch):
            mse_train = _train(train_dataloader, model, loss, optimizer, device)
            
            if (epoch+1) % num_epoch == 0:
                print(f'\t\t\tMean train loss (mse): {mse_train}')

        # test
        mse_test = _test(test_dataloader, model, loss, device)
        
        r2_train, rae_train = _compute_statistics(train_dataloader, model, device)
        r2_test, rae_test = _compute_statistics(test_dataloader, model, device)

        rmse_train = sqrt(mse_train)
        rmse_test = sqrt(mse_test)

        print(f'\t\t\tMean train loss: {mse_train}')
        print(f'\t\t\tMean test loss: {mse_test}')
        print(f'\t\t\tR2 train score: {r2_train}')
        print(f'\t\t\tR2 test score: {r2_test}')

        # construct rows
        params = {
            'hs': hidden_size,
            'n_layers': num_layers,
            'num_ephocs': num_epoch,
            'batch_size': batch_size,
            'lr': learning_rate
            }

        results.append([params, r2_train, r2_test, mse_train, mse_test, rmse_train, rmse_test, rae_train, rae_test])

        return results
        
def _compute_statistics(dataloader, model, device):
    model.eval()

    with torch.no_grad():
        x_data = dataloader.dataset.X.to(device)

        preds = model(x_data) # all data
        preds = preds.detach().cpu().numpy()

        y_data = dataloader.dataset.y.detach().cpu().numpy()
        r2 = r2_score(y_data, preds)
        rae = utils.rae(y_data, preds)

    return r2, rae



def _train(dataloader, model, loss_fn, optimizer, device):
    model.train() # set model model
    
    num_batches = len(dataloader)
    total_loss = 0

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

    return total_loss/num_batches

def _test(dataloader, model, loss_fn, device):
    model.eval()
    
    num_batches = len(dataloader)
    total_loss = 0

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)

            prediction = model(data)
            loss =loss_fn(prediction.flatten(), targets)
            total_loss += loss.item()

    return total_loss/num_batches
