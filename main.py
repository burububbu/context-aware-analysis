import os

import pandas as pd
import dataset_handler as dh

from sklearn.model_selection import train_test_split

def main():
    csv_data =  "./data/noises.csv"
    
    if not os.path.exists(csv_data):
        dh.create_csv()
    
    data = pd.read_csv(csv_data)

    x_train, x_test, y_train, y_test = train_test_split(
        data[['longitude', 'latitude']],
        data['noise'],
        test_size=0.20,
        random_state=42)

    train_dataset = x_train.assign(noise=y_train)
    
    dh.features_engineering(train_dataset)
    
    # subsets = dh.create_subsets(data)

if __name__ == '__main__':
    main()
