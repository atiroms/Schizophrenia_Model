######################################################################
# Description ########################################################
######################################################################
'''
Python code for neural networks used for meta-RL.
'''


######################################################################
# Libraries ##########################################################
######################################################################

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
#from functions.helper import *


######################################################################
# Functions ##########################################################
######################################################################
# Used to initialize weights for policy and value output layers

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


######################################################################
# LSTM-RNN network ###################################################
######################################################################

class LSTM_RNN_Network():
    def __init__(self,param,n_actions,scope,trainer):
        self.param=param
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.prev_rewards = tf.placeholder(shape=[None,1],dtype=tf.float32)
            self.prev_actions = tf.placeholder(shape=[None],dtype=tf.int32)
            self.timestep = tf.placeholder(shape=[None,1],dtype=tf.float32)
            self.prev_actions_onehot = tf.one_hot(self.prev_actions,n_actions,dtype=tf.float32)
            
            # Input to LSTM-RNN. timestep is fed
            hidden = tf.concat([self.prev_rewards,self.prev_actions_onehot,self.timestep],1)
            
            # LSTM cells
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(int(self.param.n_cells_lstm),state_is_tuple=True, name='LSTM_Cells')
            
            # Initial all-zero state of LSTM cells
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]

            # Placeholder of lstm cell states input
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)

            rnn_in = tf.expand_dims(hidden, [0])
            # returns length of reward array along the first axis (usually zero?)
            step_size = tf.shape(self.prev_rewards)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, int(self.param.n_cells_lstm)])
            
            self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions,n_actions,dtype=tf.float32)
                        
            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out,n_actions,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(rnn_out,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            
            # Only the agent network need ops for loss functions and gradient updating.
            if scope != 'master':
                self.value_target = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                #self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                #self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
                #self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7)*self.advantages)
                #self.loss = 0.5 *self.value_loss + self.policy_loss - self.entropy * 0.05
                self.loss_policy = - tf.reduce_sum(tf.log(self.responsible_outputs + 1e-10)*self.advantages) # advantage as a constant
                # advantage as a variable. this expression is equivalent to Wang 2018 method
                self.loss_value = 0.5 * tf.reduce_sum(tf.square(self.value_target - tf.reshape(self.value,[-1])))
                self.loss_entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10))
                self.loss_total = self.loss_policy + self.param.cost_statevalue_estimate * self.loss_value - self.param.cost_entropy * self.loss_entropy

                #Get gradients from local network using local losses
                vars_local = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss_total,vars_local)
                # return square root of the sum of squares of l2 norms of the input tensors
                self.norms_var = tf.global_norm(vars_local)
                # return a list of tensors clipped using global norms
                grads,self.norms_grad = tf.clip_by_global_norm(self.gradients,50.0)
                
                # Apply local gradients to master network
                vars_master = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master')
                self.apply_grads = trainer.apply_gradients(zip(grads,vars_master))