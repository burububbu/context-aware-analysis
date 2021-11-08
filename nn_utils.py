from classes.nn_classes import NeuralNet
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

import torch.nn as nn
import torch

import itertools
       
def train_neural_networks(train_dataset, test_dataset, params):
    torch.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f'Training Neural Network (using {device} device)...')

    # generate all possible combinations 
    hyperparams = itertools.product(params['hidden_sizes'], params['num_epochs'], params['batch_sizes'], params['learning_rates'])

    loss = nn.MSELoss()

    # hyperparams tuning
    for hidden_size, num_epoch, batch_size, learning_rate in hyperparams:
        print(f'\t\tTraining model with {hidden_size} hidden size, {num_epoch} epochs and {batch_size} as batch size')

        # create model
        model = NeuralNet(train_dataset.X.shape[1], hidden_size).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        # create Dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        for epoch in range(num_epoch):
            #train
            mean_train_loss, r2_score = _train(train_dataloader, model, loss, optimizer, device)
            
            # if (epoch+1) % 10 == 0:
            #     print(f'\t\t Epoch {epoch + 1}/{num_epoch}')
            #     print(f'\t\t\tMedium train loss (mse): {mean_train_loss}')
            #     print(f'\t\t\tTrain score (r2): {mean_train_loss}') 
            
            if (epoch+1) % num_epoch == 0:
                print(f'\t\t\tMean train loss (mse): {mean_train_loss}')

        # test    
        mean_test_loss, r2_score =_test(test_dataloader, model, loss, device)
        print(f'\t\t\tMean test loss: {mean_test_loss}')
        print(f'\t\t\tMean R2 test score: {r2_score}')


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

    return total_loss/num_batches, None


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
    
    x_data = dataloader.dataset.X.to(device)
    
    preds = model(x_data)
    preds = preds.detach().cpu().numpy()
    
    y_data = dataloader.dataset.y.detach().cpu().numpy()
    r2 = r2_score(y_data, preds)

    return total_loss/num_batches, r2
