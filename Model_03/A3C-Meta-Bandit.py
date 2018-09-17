# Description
# python code modified from awjuliani/meta-RL implementation of meta reinforcement learning


# Parameters

#n_agents = 1       # number of agents that acts in parallel
n_agents = 2
n_actions = 2       # choice of actions
n_cells_lstm = 48   # number of cells in LSTM network

#gamma = .8
gamma = .9          # discount rate for advantage estimation and reward discounting
#learning_rate = 1e-3   # awjuliani/meta-RL (Adam optimzer)
learning_rate = 0.0007  # Wang Nat Neurosci 2018 (RMSProp optimizer)
cost_statevalue_estimate = 0.05
cost_entropy = 0.05

#load_model = True  # load trained model
load_model = False  # train model from scratch
load_model_path = "./saved_data/20180917_011631"

train = True        # enable training using the slow RL
#train = False      # disable training using the slow RL


# Libraries

import os
import threading
#import multiprocessing
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import datetime
from functions.helper import *


# Directory Organization

saved_data_path="./saved_data/"+"{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
model_path=saved_data_path+"/model"
pics_path=saved_data_path+"/pics"
summary_path=saved_data_path+"/summary"
if not os.path.exists(saved_data_path):
    os.makedirs(saved_data_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(pics_path):
    os.makedirs(pics_path)
if not os.path.exists(summary_path):
    os.makedirs(summary_path)


# Environment of Dependent Bandit

class Dependent_Bandit():
    def __init__(self,difficulty):
        self.num_actions = 2
        self.difficulty = difficulty
        self.reset()
        
    def set_restless_prob(self):
        self.bandit = np.array([self.restless_list[self.timestep],1 - self.restless_list[self.timestep]])
        
    def reset(self):
        self.timestep = 0
        if self.difficulty == 'restless': 
            variance = np.random.uniform(0,.5)
            self.restless_list = np.cumsum(np.random.uniform(-variance,variance,(150,1)))
            self.restless_list = (self.restless_list - np.min(self.restless_list)) / (np.max(self.restless_list - np.min(self.restless_list))) 
            self.set_restless_prob()
        if self.difficulty == 'easy': bandit_prob = np.random.choice([0.9,0.1])
        if self.difficulty == 'medium': bandit_prob = np.random.choice([0.75,0.25])
        if self.difficulty == 'hard': bandit_prob = np.random.choice([0.6,0.4])
        if self.difficulty == 'uniform': bandit_prob = np.random.uniform()
        if self.difficulty != 'independent' and self.difficulty != 'restless':
            self.bandit = np.array([bandit_prob,1 - bandit_prob])
        else:
            self.bandit = np.random.uniform(size=2)
        
    def pullArm(self,action):
        #Get a random number.
        if self.difficulty == 'restless': self.set_restless_prob()
        self.timestep += 1
        bandit = self.bandit[action]
        result = np.random.uniform()
        if result < bandit:
            #return a positive reward.
            reward = 1
        else:
            #return a negative reward.
            reward = 0
        if self.timestep > 99: 
            done = True
        else: done = False
        return reward,done,self.timestep


# Actor-Critic Network

class AC_Network():
    def __init__(self,n_actions,scope,trainer):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.prev_rewards = tf.placeholder(shape=[None,1],dtype=tf.float32)
            self.prev_actions = tf.placeholder(shape=[None],dtype=tf.int32)
            self.timestep = tf.placeholder(shape=[None,1],dtype=tf.float32)
            self.prev_actions_onehot = tf.one_hot(self.prev_actions,n_actions,dtype=tf.float32)

            hidden = tf.concat([self.prev_rewards,self.prev_actions_onehot,self.timestep],1)
            
            #Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_cells_lstm,state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.prev_rewards)[:1]     # returns shape of the tensor
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, n_cells_lstm])
            
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
            
            #Only the agent network need ops for loss functions and gradient updating.
            if scope != 'master':
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                #self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                #self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
                #self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7)*self.advantages)
                #self.loss = 0.5 *self.value_loss + self.policy_loss - self.entropy * 0.05
                self.policy_loss = - tf.reduce_sum(tf.log(self.responsible_outputs + 1e-10)*self.advantages) # advantage as a constant 
                self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1]))) # advantage as a variable
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10))
                self.loss = self.policy_loss + cost_statevalue_estimate * self.value_loss - cost_entropy * self.entropy

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars) # returns square root of the sum of squares of l2 norms of the input tensors
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,50.0) # returns a list of tensors clipped using global norms
                
                #Apply local gradients to master network
                master_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master')
                self.apply_grads = trainer.apply_gradients(zip(grads,master_vars))


# Agent

class Agent():
    def __init__(self,game,id,n_actions,trainer,model_path,global_episodes):
        self.name = "agent_" + str(id)
        self.id = id        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        #self.episode_rewards = []
        #self.episode_lengths = []
        #self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(summary_path+"/agent_"+str(self.id))   # store summaries for each agent

        #Create the local copy of the network and the tensorflow op to copy master paramters to local network
        self.local_AC = AC_Network(n_actions,self.name,trainer)
        self.update_local_ops = update_target_graph('master',self.name)        
        self.env = game
        
    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        actions = rollout[:,0]
        rewards = rollout[:,1]
        timesteps = rollout[:,2]
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        values = rollout[:,4]
        
        self.pr = prev_rewards
        self.pa = prev_actions
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the master network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {
            self.local_AC.target_v:discounted_rewards,
            self.local_AC.prev_rewards:np.vstack(prev_rewards),
            self.local_AC.prev_actions:prev_actions,
            self.local_AC.actions:actions,
            self.local_AC.timestep:np.vstack(timesteps),
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:rnn_state[0],
            self.local_AC.state_in[1]:rnn_state[1]}
        t_l,v_l,p_l,e_l,g_n,v_n,_ = sess.run([
            self.local_AC.loss,
            self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return t_l / len(rollout), v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n
        
    def work(self,gamma,sess,coord,saver,train):
        episode_count_global = sess.run(self.global_episodes) # refer to global episode count over all agents
        episode_count_local = 0
        agent_steps = 0
        print("Starting " + self.name + "                    ")
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                episode_count_global = sess.run(self.global_episodes)   # refer to global episode count over all agents
                sess.run(self.increment)                                # add to global global_episodes
                if self.name == 'agent_0':
                    print("Running global episode " + str(episode_count_global) + ", " + self.name + " local episode " + str(episode_count_local)+ "          ", end="\r")
                sess.run(self.update_local_ops) # copy master graph to local
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = [0,0]
                episode_steps = 0   # counter of steps within an episode
                d = False
                r = 0
                a = 0
                t = 0
                self.env.reset()
                rnn_state = self.local_AC.state_init
                
                while d == False: # d is "done" flag returned from the environment
                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state_new = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                        feed_dict={
                        self.local_AC.prev_rewards:[[r]],
                        self.local_AC.timestep:[[t]],
                        self.local_AC.prev_actions:[a],
                        self.local_AC.state_in[0]:rnn_state[0],
                        self.local_AC.state_in[1]:rnn_state[1]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    
                    rnn_state = rnn_state_new
                    r,d,t = self.env.pullArm(a)                        
                    episode_buffer.append([a,r,t,d,v[0,0]])
                    episode_values.append(v[0,0])
                    episode_frames.append(set_image_bandit(episode_reward,self.env.bandit,a,t))
                    episode_reward[a] += r
                    agent_steps += 1
                    episode_steps += 1
                
                #self.episode_rewards.append(np.sum(episode_reward))
                #self.episode_lengths.append(episode_steps)
                #self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0 and train == True:
                    t_l,v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)

                # Save performance coefficients of the episode
                summary = tf.Summary()
                summary.value.add(tag="Performance/Reward", simple_value=float(np.sum(episode_reward)))
                summary.value.add(tag="Performance/Step Length", simple_value=int(episode_steps))
                summary.value.add(tag="Performance/Mean State-Action Value", simple_value=float(np.mean(episode_values)))
                if train=True:
                    summary.value.add(tag="Loss/Total Loss", simple_value=float(t_l))
                    summary.value.add(tag="Loss/Value Loss", simple_value=float(v_l))
                    summary.value.add(tag="Loss/Policy Loss", simple_value=float(p_l))
                    summary.value.add(tag="Loss/Entropy", simple_value=float(e_l))
                    summary.value.add(tag="Loss/Gradient L2Norm", simple_value=float(g_n))
                    summary.value.add(tag="Loss/Varriable L2Norm", simple_value=float(v_n))
                self.summary_writer.add_summary(summary, episode_count_local)
                self.summary_writer.flush()
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count_local % 50 == 0 and episode_count_local != 0:
                    # save means of coefficients over the last 50 episodes
                    #mean_reward = np.mean(self.episode_rewards[-50:])
                    #mean_length = np.mean(self.episode_lengths[-50:])
                    #mean_value = np.mean(self.episode_mean_values[-50:])
                    # this syntax is used for saving only every 50 episodes instead of saving all
                    #summary = tf.Summary()
                    #summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    #summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    #summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    #if train == True:
                    #    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    #    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    #    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    #    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    #    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    #self.summary_writer.add_summary(summary, episode_count_local)
                    #self.summary_writer.flush()

                    # save master model + one local learning, baed on local episode_count, only in agent_0
                    if episode_count_local % 500 == 0 and self.name == 'agent_0' and train == True:
                        saver.save(sess,self.model_path+'/model-'+str(episode_count_local)+'.ckpt')
                        print("Saved model parameters                    ")

                    # save gif image of fast learning only in agent_0
                    if episode_count_local % 100 == 0 and self.name == 'agent_0':
                        self.images = np.array(episode_frames)
                        make_gif(self.images,pics_path+'/image'+str(episode_count_local)+'.gif',
                            #duration=len(self.images)*0.1,true_image=True,salience=False)
                            duration=len(self.images)*0.1,true_image=True)

                #if self.name == 'agent_0':    # add to global global_episodes only if agent_0
                #    sess.run(self.increment)
                episode_count_local += 1        # add to local episode_count in all agents


# Main code

tf.reset_default_graph()

# Setup agents for multiple threading
with tf.device("/cpu:0"): 
#with tf.device("/device:GPU:0"): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)  # counter of episodes in agent_0 defined outside Agent class
    #trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
    trainer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    master_network = AC_Network(n_actions,'master',None) # Generate master network
    #n_agents = multiprocessing.cpu_count() # Set agents to number of available CPU threads
    
    agents = []
    # Create Agent classes
    for i in range(n_agents):
        agents.append(Agent(Dependent_Bandit('uniform'),i,n_actions,trainer,model_path,global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

# Run agents
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(load_model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    agent_threads = []
    for agent in agents:
        agent_work = lambda: agent.work(gamma,sess,coord,saver,train)
        thread = threading.Thread(target=(agent_work))
        thread.start()
        agent_threads.append(thread)
    coord.join(agent_threads)

# End of code