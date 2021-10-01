import models
import os

import numpy as np
import pandas as pd
import dataset_handler as dh

def main():
    csv_data =  "./data/noises.csv"
    
    if not os.path.exists(csv_data):
        dh.create_csv()
    
    data = pd.read_csv(csv_data)

    dh.extract_features(data)

    data['latitude'] = data['latitude'].map(np.radians) 
    data['longitude'] = data['longitude'].map(np.radians) 
    data['noise'] = data['noise'].apply(lambda value: round(value,5))
    
    subsets = dh.create_subsets(data)

    for i,subset in enumerate(subsets):
        print('subset {0} has {1} samples'.format(i, subset.shape[0]))
        
        x_data = subset[['latitude','longitude']]
        y_data = subset['noise']
        
        models.train_knn(x_data, y_data)

if __name__ == '__main__':
    main()
