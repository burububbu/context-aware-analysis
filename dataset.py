import pandas as pd

from sklearn.model_selection import train_test_split

class Dataset:
    data = None

    x_train = None
    y_train = None
    x_test = None
    t_test = None

    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def split(self, test_size, random_state):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
        self.data[['longitude', 'latitude']],
        self.data['noise'],
        test_size=test_size,
        random_state=random_state)

    def train_to_csv(self, path):
        to_save = self.x_train.join(self.y_train)
        to_save.to_csv(path)


        