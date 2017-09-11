import pandas as pd
import numpy as np
from pandas import DataFrame
from functions import array_to_one_hot
from session_run import session_run


train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

## we random shuffle the train data
train = train.sample(frac=1).reset_index(drop=True)

feature_name = train.columns[1:]
label_name = train.columns[0]

train_feature = train[feature_name].as_matrix()
test_feature = test[feature_name].as_matrix()

labels = sorted(set(train[label_name].tolist()))

train_label_one_hot = array_to_one_hot(train[label_name].tolist(),labels)
test_label_one_hot = array_to_one_hot(test[label_name].tolist(),labels)

train_label_array = np.array(train[label_name])
test_label_array = np.array(test[label_name])

##############################

# Parameters

train_num = train_feature.shape[0] ## train_size
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

# Network Parameters
layers_structure = [1600,800]

run = session_run(train_feature, train_label_one_hot, train_label_array, \
                test_feature, test_label_one_hot, test_label_array, labels,\
                layers_structure, train_num, learning_rate, training_epochs,\
                batch_size, display_step)

test_accuracy = run['test_accuracy']
train_accuracy = run['train_accuracy']
train_pred_label_array = run['train_pred_label_array']
test_pred_label_array = run['test_pred_label_array']
train_pred_prob_matrix = run['train_pred_prob_matrix']
test_pred_prob_matrix = run['test_pred_prob_matrix']
test_loss = run['test_loss']
train_loss = run['train_loss']
        
