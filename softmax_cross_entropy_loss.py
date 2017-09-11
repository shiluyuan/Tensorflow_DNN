from math import log
import numpy as np

def single_cross_entropy_loss(true_prob, pred_prob):
    cross_entropy_loss = 0
    
    for i in range(len(true_prob)):
        if true_prob[i] == 0:
            cross_entropy_loss += 0
        else:
            if pred_prob[i] == 0:
                cross_entropy_loss += true_prob[i] * log(1e-10)
            else:
                cross_entropy_loss += true_prob[i] * log(pred_prob[i])
    
    return -cross_entropy_loss

def all_cross_entropy_loss(true_prob_matrix, pred_prob_matrix):
    all_cross_entropy_loss = 0
    for i in range(len(true_prob_matrix)):
        single_loss = single_cross_entropy_loss(true_prob_matrix[i], pred_prob_matrix[i])
        all_cross_entropy_loss += single_loss
    return all_cross_entropy_loss

def get_lable_array(lable):
    # input lable is list
    new_lable =  []
    lables =  sorted(set(lable))
    
    for i in lable:
        single_lable = [0] * len(lables)
        index = lables.index(i)
        single_lable[index] = 1
        new_lable.append(single_lable)
    return np.array(new_lable)
