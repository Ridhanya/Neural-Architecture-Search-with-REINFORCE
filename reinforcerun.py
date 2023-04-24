import pandas as pd
import pickle
import numpy as np
from reinforceNAS import *
from reinforcevariables import *



# read the data
train_data = pd.read_csv('DATA/train.csv')
val_data = pd.read_csv('DATA/val.csv')


# split it into X and y values
x = np.array(train_data.drop(['label','filename','patient_id'], axis=1, inplace=False)).astype('float32')
#y = pd.get_dummies(data['label']).values
y = (train_data['label']).values

#validation dataset
x_val = np.array(val_data.drop(['label','filename','patient_id'], axis=1, inplace=False)).astype('float32')
y_val = (val_data['label']).values

# let the search begin
full_data, only_models = reinforcesearch(np.shape(x[0]),x,y,x_val,y_val)


#log data
with open(nas_data_log, 'wb') as f:
    pickle.dump(only_models, f)

#log data
with open(nas_full_data_log, 'wb') as g:
    pickle.dump(full_data, g)

# # get top n architectures (the n is defined in constants)
# get_top_n_architectures(TOP_N)

# #plot
# get_nas_accuracy_plot()