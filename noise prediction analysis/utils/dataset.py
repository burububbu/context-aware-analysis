import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


class Dataset:
    def __init__(self):

        self.x_train = None
        self.y_train = None

        self.x_test = None
        self.y_test = None

        # indices of best n features
        self.feat_subset_indices = None

    @property
    def x_train_complete(self):
        return self.x_train

    @property
    def x_test_complete(self):
        return self.x_test

    @property
    def x_train_base(self):
        return self.x_train[['latitude', 'longitude']]

    @property
    def x_test_base(self):
        return self.x_test[['latitude', 'longitude']]

    @property
    def x_train_subset(self):
        return self.x_train.iloc[:, self.feat_subset_indices]

    @property
    def x_test_subset(self):
        return self.x_test.iloc[:, self.feat_subset_indices]

    def load_train_test_from_csv(self, csv_train, csv_test, num_sub_features):
        ''' Load train and test data from specific csv'''
        train_data = pd.read_csv(csv_train)
        test_data = pd.read_csv(csv_test)

        self.load_train_test_from_df(train_data, test_data, num_sub_features)

    def load_train_test_from_df(self, train_data, test_data, num_sub_features):
        ''' Load train and test data from specific csv'''

        self.x_train = train_data[train_data.columns.drop('noise')]
        self.y_train = train_data['noise']

        self.x_test = test_data[test_data.columns.drop('noise')]
        self.y_test = test_data['noise']

        # select n best features for subset
        features_selector = SelectKBest(
            f_regression, k=num_sub_features).fit(self.x_train, self.y_train)

        self.feat_subset_indices = features_selector.get_support(indices=True)
