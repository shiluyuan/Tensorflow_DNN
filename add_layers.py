import tensorflow as tf

def add_single_layer(input_layer,in_size,out_size,activation_function=None):
    weights = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.5))
    biases = tf.Variable(tf.zeros([out_size])+0.1)
    wx_plus_b = tf.add(tf.matmul(input_layer,weights),biases)
    if activation_function is None:
        output_layer = wx_plus_b
    else:
        output_layer = activation_function(wx_plus_b) 
    return output_layer



def add_layers(layers_structure,feature_layer,n_features,n_classes):
    # layers is like [44,33,22]
    layer_num = len(layers_structure)
    layer_name = list(range(layer_num))
    layer_dict = {}    
    # set the first layer
    layer_dict[layer_name[0]] = \
    add_single_layer(feature_layer,n_features,layers_structure[0],activation_function=tf.nn.relu)
    # set the middle layer
    for i in range(1,(layer_num)):
        layer_dict[layer_name[i]] = \
        add_single_layer(layer_dict[layer_name[i-1]],layers_structure[i-1],layers_structure[i],activation_function=tf.nn.relu)
    # set the last layer
    out_layer = \
    add_single_layer(layer_dict[layer_name[-1]],layers_structure[-1],n_classes,activation_function=None)
    
    return out_layer

       