import models

import numpy as np

class Features_handler():
    neighbors_learners = {}

    def to_radiants(self, data):
        
        data['latitude'] = data['latitude'].map(np.radians) 
        data['longitude'] = data['longitude'].map(np.radians)
        
        return data
    
    def round_values(self, data):
        return data.apply(lambda value: round(value,5))
       
    def init_learners(self, x_train_data, neighbors= []):
        for num_neighbors  in neighbors:
            learner_name = 'nearest_{}'.format(num_neighbors)
            self.neighbors_learners[learner_name] =  models.create_neigbors_learners(x_train_data, num_neighbors)

    def get_neighbors_features(self, x_train, y_train):
        features_values = {}

        for name_learner, learner in self.neighbors_learners.items(): 
            all_distances, all_neighbors_indexes = learner.kneighbors(x_train)
            
            distance_mean = []
            distance_std = []
            noise_mean = []
            noise_std = []
            
            for distances in all_distances:
                distance_mean.append(np.mean(distances))
                distance_std.append(np.std(distances))
                
            for neighbors_indexes in all_neighbors_indexes:
                noises = y_train.iloc[neighbors_indexes]

                noise_mean.append(np.mean(noises))
                noise_std.append(np.std(noises))

            features_values[name_learner + '_distance_mean'] = distance_mean
            features_values[name_learner + '_distance_std'] =  distance_std
            features_values[name_learner + '_noise_mean'] = noise_mean
            features_values[name_learner + '_noise_std'] = noise_std
        
        return features_values
            





