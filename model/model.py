from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from model_flow import *
from attention_lstm import *
from batch_generator import *
from utils import *

num_steps = 20000
summary_frequency = 1000

with tf.Session(graph=graph) as session:
  global dropout
  tf.global_variables_initializer().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches,output_batches = train_batches.next()
    feed_dict = dict()
    dropout=0.5
    
    for i in range(num_unrollings):
        #Feeding input from reverse according to https://arxiv.org/abs/1409.3215
        feed_dict[train_inputs[i]]=batches[i]
        feed_dict[decoder_inputs[i]]=output_batches[i]

        
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    #reset_state.run()

    if step % (summary_frequency ) == 0:
        dropout=1
        print('-'*80)
        print('Step '+str(step))
        print('Loss '+str(l))
        
        labels=np.concatenate(list(output_batches)[:])
#        print(characters(labels))
#       print(characters(predictions))
        print('Batch Accuracy: %.2f' % float(accuracy(labels,predictions)*100))
        
        num_validation = valid_size // num_unrollings
        reset_sample_state.run()
        sum_acc=0
        for _ in range(num_validation):
            valid,valid_output=valid_batches.next()
            valid_feed_dict=dict()
            for i in range(num_unrollings):
                valid_feed_dict[sample_input[i]]=valid[i]
            sample_pred=session.run(sample_outputs,feed_dict=valid_feed_dict)
            labels=np.concatenate(list(valid_output)[:],axis=0)
            pred=np.concatenate(list(sample_pred)[:],axis=0)
            sum_acc=sum_acc+accuracy(labels,pred)
        val_acc=sum_acc/num_validation
        print('Validation Accuracy: %0.2f'%(val_acc*100))
        print('Input Test String '+str(batches2string(valid)))
        print('Output Prediction'+str(batches2string(sample_pred)))
        print('Actual'+str(batches2string(valid_output)))
