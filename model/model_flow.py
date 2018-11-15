from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
from attention_lstm import *
from batch_generator import *
from utils import *

num_nodes =256
vocabulary_size = 10
#dropout=1.0


class Model:

	graph = tf.Graph()
	with graph.as_default():
	    #Variables and placeholders
	    #
	    #Weights for the hidden states accross time
	    attn_weights=tf.Variable(tf.truncated_normal([num_nodes], -0.1, 0.1))
	    #Weights for the context(hidden state) at time t-1
	    prev_hidden_weights=tf.Variable(tf.truncated_normal([num_nodes], -0.1, 0.1))
	    #LSTM for encoder and decoder
	    encoder_lstm=lstm(vocabulary_size)
	    #feed decoder Y(t-1) and attention context
	    decoder_lstm=lstm(num_nodes+vocabulary_size)

	    #State saving across unrollings
	    saved_state=tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
	    saved_output=tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False) 
	    state=saved_state
	    output=saved_output
	    
	    
	    reset_state = tf.group(
		output.assign(tf.zeros([batch_size, num_nodes])),
		state.assign(tf.zeros([batch_size, num_nodes])),
		)
	    
	    # Classifier weights and biases.
	    w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
	    b = tf.Variable(tf.zeros([vocabulary_size]))
	    
	    #Define our train input, decoder output variables
	    train_inputs=[]
	    decoder_inputs=[]
	    outputs=[]
	    
	    for i in range(num_unrollings):
		train_inputs.append(tf.placeholder(tf.float32,shape=[batch_size,vocabulary_size]))
		decoder_inputs.append(tf.placeholder(tf.float32,shape=[batch_size,vocabulary_size]))
		
		
	    #    
	    #Encoder, Decoder and attention functions
	    #
	    
	    def encoder(train_input,output,state):
		'''
		Args:       
		
		train_input : array of size num_unrolling, each array element is a Tensor of dimension batch_size,
		vocabulary size.
		
		Returns:
		
		output : Output of LSTM aka Hidden State
		state : Cell state of the LSTM
		
		'''
		i = len(train_inputs) - 1
		outputs=[]
		while i >= 0:
		    output, state = encoder_lstm.lstm_cell(train_input[i],output,state)
		    outputs.append(output)
		    i=i-1
		#Return the all the outputs because they will be required by the attention mechanism
		return outputs,output,state
	    
	    def soft_attention(hidden_states,prev_hidden_state,batch_size):
		'''
		
		Implements soft attention mechanism over an array of encoder hidden 
		states given previous decoder hidden states
		
		Returns a context by attending over the hidden states accross time
		
		Used by the decoder at each timestep during decoding
		
		'''
		#Prev hidden weights
		prev_hidden_state_times_w=tf.multiply(prev_hidden_state,prev_hidden_weights)
		for h in range(num_unrollings):
		    hidden_states[h]=tf.multiply(hidden_states[h],attn_weights)+prev_hidden_state_times_w 
		unrol_states=tf.reshape(tf.concat(hidden_states,1),(batch_size,num_unrollings,num_nodes))
		eij=tf.tanh(unrol_states)
		#Softmax across the unrolling dimension
		softmax=tf.nn.softmax(eij,dim=1)
		context=tf.reduce_sum(tf.multiply(softmax,unrol_states),axis=1) #Sum across axis time
		return context
		
	    
	    def training_decoder(decoder_input,hidden_states,output,state):
		outputs=[]
		#Predict the first character using the EOS Tag. We use EOS tag as the start tag
		context=soft_attention(hidden_states,output,batch_size)
		inp_concat=tf.concat([decoder_input[-1],context],axis=1)
		output, state = decoder_lstm.lstm_cell(inp_concat,output,state)
		outputs.append(output)
		#Now predict the next outputs using the training labels itself. Using y(n-1) to predict y(n)
		for i in decoder_input[0:-1]:
		    context=soft_attention(hidden_states,output,batch_size)
		    inp_concat=tf.concat([i,context],axis=1)
		    output,state=decoder_lstm.lstm_cell(inp_concat,output,state)
		    outputs.append(output)
		    
		return outputs,output,state
	    
	    
	    def inference_decoder(go_char,hidden_states,decode_steps,output,state):
		outputs=[]
		#First input to decoder is the the Go Character
		context=soft_attention(hidden_states,output,1)
		inp_concat=tf.concat([go_char,context],axis=1)
		output,state=decoder_lstm.lstm_cell(inp_concat,output,state)
		outputs.append(output)
		for i in range(decode_steps-1):
		    #Feed the previous output as the next decoder input
		    decoder_input=tf.nn.softmax(tf.nn.xw_plus_b(output, w, b))
		    context=soft_attention(hidden_states,output,1)
		    inp_concat=tf.concat([decoder_input,context],axis=1)
		    output,state=decoder_lstm.lstm_cell(inp_concat,output,state)
		    outputs.append(output)
		return outputs,output,state
	    

		
	    #
	    #Model Definition
	    #
	    hidden_states,output,state=encoder(train_inputs,output,state)
	    outputs,output,state=training_decoder(decoder_inputs,hidden_states,output,state)
	    


	    with tf.control_dependencies([saved_state.assign(state),
		                        saved_output.assign(output),
		                            ]):
		logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		                            labels=tf.concat(decoder_inputs, 0), logits=logits))
		
	    #Loss function and optimizer
	    global_step = tf.Variable(0)
	    learning_rate = tf.train.exponential_decay(
		                    10.0, global_step, 5000, 0.1, staircase=True)
	    optimizer = tf.train.AdamOptimizer()
	    gradients, v = zip(*optimizer.compute_gradients(loss))
	    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
	    optimizer = optimizer.apply_gradients(
		            zip(gradients, v), global_step=global_step)
	    
	    # Predictions.
	    train_prediction = tf.nn.softmax(logits)
	    
	    #  
	    #Sample Prediction
	    #
	    sample_input=[]
	    sample_outputs=[]

	    for i in range(num_unrollings):
		sample_input.append(tf.placeholder(tf.float32,shape=[1,vocabulary_size]))
		
	    sample_saved_state=tf.Variable(tf.zeros([1, num_nodes]), trainable=False)
	    sample_saved_output=tf.Variable(tf.zeros([1, num_nodes]), trainable=False)
	    

	    
	    sample_output=sample_saved_output
	    sample_state=sample_saved_state

	    
	    
	    reset_sample_state = tf.group(
		sample_output.assign(tf.zeros([1, num_nodes])),
		sample_state.assign(tf.zeros([1, num_nodes])),

		)
	    

	    hidden_states,sample_output,sample_state=encoder(sample_input,sample_output,sample_state)
	    sample_decoder_outputs,sample_output,sample_state=inference_decoder(sample_input[-1],hidden_states,num_unrollings,sample_output,sample_state)

	    with tf.control_dependencies([sample_saved_output.assign(sample_output),
		                        sample_saved_state.assign(sample_state),
		                       ]):
		for d in sample_decoder_outputs:
		        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(d, w, b))
		        sample_outputs.append(sample_prediction)


