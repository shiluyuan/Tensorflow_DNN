import numpy as np
    
def array_to_one_hot(all_lables,lables):
    # input lable is list
    new_lable =  [] 
    for i in all_lables:
        single_lable = [0] * len(lables)
        index = lables.index(i)
        single_lable[index] = 1
        new_lable.append(single_lable)
    return np.array(new_lable)


def next_batch(round_num,sample_num, data):
    # Return a total of `num` samples from the array `data`. 
    return data[sample_num*round_num:sample_num*(round_num+1)]


def fake_label_to_real_label_array(fake_label, labels):
    real_label_array = []
    for i in fake_label:
        real_label_array.append(labels[i])
    return np.array(real_label_array)