
import torch
import torch.nn as nn

from torch.utils.data import Dataset


class DatasetNN(Dataset):
    def __init__(self, x_data, y_data):
        self.X = torch.FloatTensor(x_data)
        self.y = torch.FloatTensor(y_data)

    def __len__(self):
        ''' Amount of samples '''
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]


class NeuralNet(nn.Module):
    '''Neural network model'''

    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super(NeuralNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # first input layer
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        ]

        # add hidden layer
        for _ in range(n_layers):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            new_layer = [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
            layers.extend(new_layer)

        # add output layer
        layers.append(nn.Linear(hidden_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
