from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf

class lstm:
    def __init__(self,input_size):
        self.xx = tf.Variable(tf.truncated_normal([input_size, num_nodes * 4], -0.1, 0.1))
        self.mm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes * 4], -0.1, 0.1))
        self.bb = tf.Variable(tf.zeros([1, num_nodes * 4]))


                          
    def lstm_cell(self,i,o,state):
        global dropout
        #i=tf.nn.dropout(i,keep_prob=dropout)
        matmuls = tf.matmul(i, self.xx)+ tf.matmul(o, self.mm) + self.bb        
        input_gate  = tf.sigmoid(matmuls[:, 0 * num_nodes : 1 * num_nodes])
        forget_gate = tf.sigmoid(matmuls[:, 1 * num_nodes : 2 * num_nodes])
        update      =            matmuls[:, 2 * num_nodes : 3 * num_nodes]
        output_gate = tf.sigmoid(matmuls[:, 3 * num_nodes : 4 * num_nodes])
        state       = forget_gate * state + input_gate * tf.tanh(update)
        output=output_gate * tf.tanh(state)
        return output, state
