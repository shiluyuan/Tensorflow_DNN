import tensorflow as tf
import numpy as np
from add_layers import add_layers
from functions import next_batch
from functions import fake_label_to_real_label_array
from softmax_cross_entropy_loss import all_cross_entropy_loss

def session_run(train_feature, train_label_one_hot, train_label_array, \
                test_feature, test_label_one_hot, test_label_array, labels,\
                layers_structure, train_num, learning_rate, training_epochs,\
                batch_size, display_step):
    
    n_features = len(train_feature[1]) # origninal feature number
    n_classes = len(train_label_one_hot[1])
    
    x = tf.placeholder(tf.float32,[None,n_features])  
    y = tf.placeholder(tf.float32,[None,n_classes])          
    ## set the network structure
    pred = add_layers(layers_structure,x,n_features, n_classes)   
    
    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    ## final run
    with tf.Session() as sess:
        sess.run(init)   
        # Training cycle
        for k in range(training_epochs):
            total_batch = int(train_num/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x = next_batch(i,batch_size,train_feature)
                batch_y = next_batch(i,batch_size,train_label_one_hot)
                # Run optimization op (backprop) and cost op (to get loss value)
                sess.run(optimizer, feed_dict={x: batch_x,y: batch_y})
            # Display accuracy
            if k % display_step == 0:                   
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            
                test_accuracy = accuracy.eval({x: test_feature, y: test_label_one_hot})
                train_accuracy = accuracy.eval({x: train_feature, y: train_label_one_hot})
                print("period %d" %k, "train:" ,train_accuracy,"test:" ,test_accuracy)
        ######
        soft_y = tf.nn.softmax(pred)
        test_pred_prob_matrix = soft_y.eval(feed_dict={x: test_feature})
        train_pred_prob_matrix = soft_y.eval(feed_dict={x: train_feature})
        ## get predict label 
        test_pred_index = sess.run(tf.argmax(pred, 1), feed_dict={x: test_feature})
        train_pred_index = sess.run(tf.argmax(pred, 1), feed_dict={x: train_feature})
        
        test_pred_label_array = fake_label_to_real_label_array(np.array(test_pred_index.tolist()), labels) #list form
        train_pred_label_array = fake_label_to_real_label_array(np.array(train_pred_index.tolist()), labels) #list form 
        ## calculate loss
        ## for test
        test_true_prob_matrix = test_label_one_hot
        test_loss = all_cross_entropy_loss(test_true_prob_matrix, test_pred_prob_matrix) / len(test_feature)
            
        ## for train
        train_true_prob_matrix = train_label_one_hot
        train_loss = all_cross_entropy_loss(train_true_prob_matrix, train_pred_prob_matrix) / len(train_feature)

    return_dict = {"test_accuracy":test_accuracy,
                   "train_accuracy":train_accuracy,
                   "train_pred_label_array":train_pred_label_array,
                   "test_pred_label_array":test_pred_label_array,
                   "train_pred_prob_matrix":train_pred_prob_matrix,
                   "test_pred_prob_matrix":test_pred_prob_matrix,
                   "test_loss":test_loss,
                   "train_loss":train_loss}
        
    return return_dict
        
        
        
        
        
        