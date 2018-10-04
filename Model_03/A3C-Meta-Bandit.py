###############
# DESCRIPTION #
###############

# Python code for meta reinforcement learning
# For a single run,
#   run=Run()
#   run.run()
# For batch runs,
#   batch=BatchRun()
#   batch.run()


##############
# PARAMETERS #
##############

param_basic={
    'param_set': 'Wang2018',
    #'param_set' : 'awjuliani',
    #'param_set' : 'Wang2018_fast',
    #'param_set' : 'Wang2018_statevalue',

    'xpu' : '/cpu:0',                    # processing device allocation
    #'xpu' : '/gpu:0',

    'train' : True,
    'load_model' : False,
    'path_load' : './saved_data/20180917_011631',

    'path_save_master' : './saved_data',

    'n_agents' : 1,                       # number of agents that acts in parallel

    'agent': 'A2C',
    'environment' : 'Two_Armed_Bandit',

    #'episode_stop' : 100000,
    'episode_stop' : 100,

    'interval_ckpt': 1000,              # interval to save network parameters in tf default format
    #'interval_pic': 100,
    'interval_pic': 0,                  # interval to save task pictures
    'interval_var': 10                  # interval to save trainable network variables in original format
}
param_default={    # Wang 2018 parameters
    'n_cells_lstm' : 48,                  # number of cells in LSTM-RNN network
    'bootstrap_value' : 0.0,
    'bandit_difficulty' : 'uniform',      # select "independent" for independent bandit
    'gamma' : .9,                         # 0.9 in Wang Nat Neurosci 2018, discount rate for advantage estimation and reward discounting
    'optimizer' : 'RMSProp',              # "RMSProp" in Wang 2018, "Adam" in awjuliani/meta-RL
    'learning_rate' : 0.0007,             # Wang Nat Neurosci 2018
    'cost_statevalue_estimate' : 0.05,    # 0.05 in Wang 2018, 0.5 in awjuliani/meta-RL
    'cost_entropy' : 0.05,                # 0.05 in Wang 2018 and awjuliani/meta-RL
    'dummy_counter' : 0                   # dummy counter used for batch calculation
}
param_awjuliani={   # awjuliani/metaRL parameters
    'gamma' : .8,                         # 0.8 in awjuliani/meta-RL
    'optimizer' : 'Adam',
    'learning_rate' : 1e-3,               # awjuliani/meta-RL
    'cost_statevalue_estimate' : 0.5,
}
param_Wang2018_fast={
    'learning_rate' : 0.007
}
param_Wang2018_satatevalue={
    'cost_statevalue_estimate' : 0.5
}

param_batch=[
    {'name':'dummy_counter', 'n':5, 'type':'parametric', 'method':'grid', 'min':0,'max':4}
    #{'name':'learning_rate', 'n':5, 'type':'parametric', 'method':'random', 'min':0.0001, 'max':0.001},
    #{'name':'optimizer', 'n':2, 'type':'list','list':['RMSProp','Adam']},
    #{'name':'gamma','n':3,'type':'parametric','method':'grid','min':0.7,'max':0.9}
]


#############
# LIBRARIES #
#############

import os
import threading
#import multiprocessing
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import datetime
import time
import pandas as pd
import json
import moviepy.editor as mpy
from PIL import Image
from PIL import ImageDraw 
from PIL import ImageFont
from functions.helper import *


###################################
# ENVIRONMENT OF TWO-ARMED BANDIT #
###################################

class Two_Armed_Bandit():
    def __init__(self,difficulty):
        self.n_actions = 2
        self.difficulty = difficulty
        #self.reset()
        
    def set_restless_prob(self):    # sample from random walk list
        self.bandit = np.array([self.restless_list[self.timestep],1 - self.restless_list[self.timestep]])  
        
    def reset(self):
        self.timestep = 0
        if self.difficulty == "restless":       # bandit probability random-walks within an episode
            variance = np.random.uniform(0,.5)  # degree of random walk
            self.restless_list = np.cumsum(np.random.uniform(-variance,variance,(150,1)))   # calculation of random walk
            self.restless_list = (self.restless_list - np.min(self.restless_list)) / (np.max(self.restless_list - np.min(self.restless_list))) 
            self.set_restless_prob()
        if self.difficulty == "easy": bandit_prob = np.random.choice([0.9,0.1])
        if self.difficulty == "medium": bandit_prob = np.random.choice([0.75,0.25])
        if self.difficulty == "hard": bandit_prob = np.random.choice([0.6,0.4])
        if self.difficulty == "uniform": bandit_prob = np.random.uniform()
        if self.difficulty == "independent": self.bandit = np.random.uniform(size=2)

        if self.difficulty != "restless" and self.difficulty != "independent":
            self.bandit = np.array([bandit_prob,1 - bandit_prob])

        return self.bandit
        
    def step(self,action):
        #Get a random number.
        if self.difficulty == "restless": self.set_restless_prob()  # sample from random walk list
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
        else: 
            done = False
        return reward,done,self.timestep
    
    def make_gif(self,buffer,path,count):     
        font = ImageFont.truetype("./resources/FreeSans.ttf", 24)
        images=[]
        r_cumulative=[0,0]
        for i in range(len(buffer)):
            r_cumulative[int(buffer[i,1])]+=buffer[i,2]
            bandit_image = Image.open('./resources/bandit.png')
            draw = ImageDraw.Draw(bandit_image)
            draw.text((40, 10),str(float("{0:.2f}".format(self.bandit[0]))),(0,0,0),font=font)
            draw.text((130, 10),str(float("{0:.2f}".format(self.bandit[1]))),(0,0,0),font=font)
            draw.text((60, 370),'Trial: ' + str(int(buffer[i,0])),(0,0,0),font=font)
            bandit_image = np.array(bandit_image)
            bandit_image[115:115+math.floor(r_cumulative[0]*2.5),20:75,:] = [0,255.0,0] 
            bandit_image[115:115+math.floor(r_cumulative[1]*2.5),120:175,:] = [0,255.0,0]
            bandit_image[101:107,10+(int(buffer[i,1])*95):10+(int(buffer[i,1])*95)+80,:] = [80.0,80.0,225.0]
            images.append(bandit_image)
        images=np.array(images)
        filename=path+'/'+str(count)+'.gif'
        duration=len(images)*0.1
        def make_frame(t):
            try:
                x = images[int(len(images)/duration*t)]
            except:
                x = images[-1]
            return x.astype(np.uint8)
        clip = mpy.VideoClip(make_frame, duration=duration)
        clip.write_gif(filename, fps = len(images) / duration, verbose=False, progress_bar=False)


####################
# LSTM-RNN NETWORK #
####################

class LSTM_RNN_Network():
    def __init__(self,param,n_actions,scope,trainer):
        self.param=param
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.prev_rewards = tf.placeholder(shape=[None,1],dtype=tf.float32)
            self.prev_actions = tf.placeholder(shape=[None],dtype=tf.int32)
            self.timestep = tf.placeholder(shape=[None,1],dtype=tf.float32)
            self.prev_actions_onehot = tf.one_hot(self.prev_actions,n_actions,dtype=tf.float32)

            hidden = tf.concat([self.prev_rewards,self.prev_actions_onehot,self.timestep],1)
            
            #Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.param.n_cells_lstm,state_is_tuple=True, name='LSTM_Cells')
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
            rnn_out = tf.reshape(lstm_outputs, [-1, self.param.n_cells_lstm])
            
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
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                #self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                #self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
                #self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7)*self.advantages)
                #self.loss = 0.5 *self.value_loss + self.policy_loss - self.entropy * 0.05
                self.policy_loss = - tf.reduce_sum(tf.log(self.responsible_outputs + 1e-10)*self.advantages) # advantage as a constant
                # advantage as a variable. this expression is equivalent to Wang 2018 method
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10))
                self.loss = self.policy_loss + self.param.cost_statevalue_estimate * self.value_loss - self.param.cost_entropy * self.entropy

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                # return square root of the sum of squares of l2 norms of the input tensors
                self.var_norms = tf.global_norm(local_vars)
                # return a list of tensors clipped using global norms
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,50.0)
                
                # Apply local gradients to master network
                master_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master')
                self.apply_grads = trainer.apply_gradients(zip(grads,master_vars))
                

#############
# A2C AGENT #
#############

class A2C_Agent():
    def __init__(self,id,param,environment,trainer,saver,global_episodes):
        self.id = id
        self.name = "agent_" + str(id)
        self.param=param
        self.env = environment
        self.n_actions=self.env.n_actions
        self.trainer = trainer
        self.saver=saver
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        # store summaries for each agent
        self.summary_writer = tf.summary.FileWriter(self.param.path_save+"/summary/"+self.name)

        # Create the local copy of the network and the tensorflow op to copy master paramters to local network
        self.local_AC = LSTM_RNN_Network(self.param,self.n_actions,self.name,trainer)
        self.update_local_ops = update_target_graph('master',self.name)        
        
        
    def train(self,episode_buffer,sess):
        timesteps = episode_buffer[:,0]
        actions = episode_buffer[:,1]
        rewards = episode_buffer[:,2]
        values = episode_buffer[:,3]
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        
        self.pr = prev_rewards
        self.pa = prev_actions
        # Here we take the rewards and values from the episode_buffer, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [self.param.bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,self.param.gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [self.param.bootstrap_value])
        advantages = rewards + self.param.gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,self.param.gamma)

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
        t_l /= episode_buffer.shape[0]
        v_l /= episode_buffer.shape[0]
        p_l /= episode_buffer.shape[0]
        e_l /= episode_buffer.shape[0] 

        return t_l, v_l, p_l, e_l, g_n, v_n
        
    def work(self,sess,coord):
        episode_count_global = sess.run(self.global_episodes)           # refer to global episode counter over all agents
        episode_count_local = 0
        agent_steps = 0
        #print("Starting " + self.name + "                    ")
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():                              # iterate over episodes
                episode_count_global = sess.run(self.global_episodes)   # refer to global episode counter over all agents
                sess.run(self.increment)                                # add to global episode counter
                print("Running global episode: " + str(episode_count_global) + ", " + self.name + " local episode: " + str(episode_count_local)+ "          ", end="\r")
                t_start = time.time()
                sess.run(self.update_local_ops)                         # copy master graph to local
                episode_buffer = []
                episode_values = []
                #episode_frames = []
                episode_reward = [0,0]
                episode_steps = 0                                       # counter of steps within an episode
                d = False
                r = 0
                a = 0
                t = 0
                bandit = self.env.reset()                               # returns np.array of bandit probabilities
                rnn_state = self.local_AC.state_init
                
                # act
                while d == False:                                       # d is "done" flag returned from the environment
                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state_new = sess.run(
                        [self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                        feed_dict={
                            self.local_AC.prev_rewards:[[r]],
                            self.local_AC.timestep:[[t]],
                            self.local_AC.prev_actions:[a],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    
                    rnn_state = rnn_state_new
                    r,d,t = self.env.step(a)                        
                    #episode_buffer.append([a,r,t,d,v[0,0]])
                    episode_buffer.append([t,a,r,v[0,0]])
                    episode_values.append(v[0,0])
                    #episode_frames.append(set_image_bandit(episode_reward,bandit,a,t))
                    episode_reward[a] += r
                    agent_steps += 1
                    episode_steps += 1

                episode_buffer=np.array(episode_buffer)
                
                # train the network using the experience buffer at the end of the episode.
                #if len(episode_buffer) != 0 and train == True:
                if self.param.train == True:
                    t_l,v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess)

                # Save activity in /activity/activity.h5 file
                df_episode = pd.DataFrame(episode_buffer)
                df_episode.columns = ['timestep', 'action', 'reward', 'value']
                df_episode.insert(loc=1, column='arm0_prob', value=bandit[0])
                df_episode.insert(loc=2, column='arm1_prob', value=bandit[1])
                df_episode.insert(loc=0, column='agent', value=self.id)
                df_episode.insert(loc=0, column='episode_count', value=episode_count_global)
                df_episode.ix[:,['episode_count','agent','timestep','action']]=df_episode.ix[:,['episode_count','agent','timestep','action']].astype('int64')

                hdf=pd.HDFStore(self.param.path_save+'/activity/activity.h5')
                hdf.put('activity',df_episode,format='table',append=True,data_columns=True)
                hdf.close()
                    
                # save model parameters as tensorflow saver
                if self.param.interval_ckpt>0:
                    if episode_count_global % self.param.interval_ckpt == 0 and self.param.train == True:
                        self.saver.save(sess,self.param.path_save+'/model/'+str(episode_count_global)+'.ckpt')
                        #print('Saved model parameters at global episode ' + str(episode_count_global) + '.                 ')

                # save model trainable variables in hdf5 format
                if self.param.interval_var>0:
                    if episode_count_global % self.param.interval_var == 0 and self.param.train == True:
                        master_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master')
                        val = sess.run(master_vars)                      
                        all_vars=np.empty(shape=[0,])
                        for v in val:
                            all_vars=np.concatenate((all_vars,v.ravel()),axis=0)
                        all_vars=pd.DataFrame(all_vars,columns=['value'])
                        all_vars.insert(loc=0, column='variable', value=range(all_vars.shape[0]))
                        all_vars.insert(loc=0, column='episode_count',value=episode_count_global)
                        #all_vars=all_vars.reshape((1,-1))
                        #all_vars=pd.DataFrame(all_vars,columns=('var'+str(i) for i in range(all_vars.shape[1])))
                        #print(type(all_vars))
                        hdf=pd.HDFStore(self.param.path_save+'/model/variable.h5')
                        hdf.put('variable',all_vars,format='table' ,append=True,data_columns=True)
                        hdf.close()

                # save gif image of fast learning
                if self.param.interval_pic>0:
                    if episode_count_global % self.param.interval_pic == 0:
                        self.env.make_gif(episode_buffer, self.param.path_save + '/pic', episode_count_global)

                # Save episode summary in /summary folder
                summary_episode = tf.Summary()
                summary_episode.value.add(tag="Performance/Reward", simple_value=float(np.sum(episode_reward)))
                summary_episode.value.add(tag="Performance/Mean State-Action Value", simple_value=float(np.mean(episode_values)))
                summary_episode.value.add(tag="Simulation/Calculation Time", simple_value=float(time.time()-t_start))
                summary_episode.value.add(tag="Environment/Step Length", simple_value=int(episode_steps))
                summary_episode.value.add(tag="Environment/Arm0 Probability", simple_value=float(bandit[0]))
                summary_episode.value.add(tag="Environment/Arm1 Probability", simple_value=float(bandit[1]))
                if self.param.train == True:
                    summary_episode.value.add(tag="Loss/Total Loss", simple_value=float(t_l))
                    summary_episode.value.add(tag="Loss/Value Loss", simple_value=float(v_l))
                    summary_episode.value.add(tag="Loss/Policy Loss", simple_value=float(p_l))
                    summary_episode.value.add(tag="Loss/Policy Entropy", simple_value=float(e_l))
                    summary_episode.value.add(tag="Loss/Gradient L2Norm", simple_value=float(g_n))
                    summary_episode.value.add(tag="Loss/Variable L2Norm", simple_value=float(v_n))
                self.summary_writer.add_summary(summary_episode, episode_count_global)
                self.summary_writer.flush()

                if episode_count_global == self.param.episode_stop:
                    print('Reached maximum episode count: '+ str(episode_count_global) + '.                           ')
                    break

                episode_count_local += 1        # add to local counter in all agents


####################
# PARAMETERS CLASS #
####################

class Parameters():
    def __init__(self,param_basic):
        self.add_item(param_basic)
        if self.param_set == 'Wang2018':
            self.add_item(param_default)
        elif self.param_set == 'awjuliani':
            self.add_item(param_default)
            self.add_item(param_awjuliani)
        elif self.param_set == 'Wang2018_fast':     # 10 times larger learning rate
            self.add_item(param_default)
            self.add_item(param_Wang2018_fast)
        elif self.param_set == 'Wang2018_statevalue':
            self.add_item(param_default)
            self.add_item(param_Wang2018_satatevalue)
        else:
            raise ValueError('Undefined parameter set name: ' + self.param_set + '.')
    
    def add_item(self,dictionary):
        for key,value in dictionary.items():
            setattr(self,key,value)


#############
# MAIN CODE #
#############

class Run():
    def __init__(self,param_basic=param_basic,param_change=None):
        self.param=Parameters(param_basic)
        if param_change is not None:
            self.param.add_item(param_change)

        # Timestamping directory name
        datetime_start="{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        path_save=self.param.path_save_master+'/'+datetime_start
        self.param.add_item({'datetime_start':datetime_start, 'path_save':path_save})

        if not os.path.exists(path_save):
            os.makedirs(path_save)
        for subdir in ['model','pic','summary','activity']:
            if not os.path.exists(path_save + '/' + subdir):
                os.makedirs(path_save + '/' + subdir)

        with open(path_save+'/parameters.json', 'w') as fp:
            json.dump(self.param.__dict__, fp, indent=1)

    def run(self):
        print('Starting calculation: '+ self.param.datetime_start + '.')
        tf.reset_default_graph()
        # Setup agents for multiple threading
        with tf.device(self.param.xpu):
            # counter of total episodes defined outside A2C_Agent class
            self.global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
            if self.param.optimizer == "Adam":
                self.trainer = tf.train.AdamOptimizer(learning_rate=self.param.learning_rate)
            elif self.param.optimizer == "RMSProp":
                self.trainer = tf.train.RMSPropOptimizer(learning_rate=self.param.learning_rate)
            self.master_network = LSTM_RNN_Network(self.param,
                                                Two_Armed_Bandit(self.param.bandit_difficulty).n_actions,
                                                'master',None) # Generate master network
            #n_agents = multiprocessing.cpu_count() # Set agents to number of available CPU threads
            self.saver = tf.train.Saver(max_to_keep=5)
            self.agents = []
            # Create A2C_Agent classes
            for i in range(self.param.n_agents):
                self.agents.append(A2C_Agent(i,self.param,Two_Armed_Bandit(self.param.bandit_difficulty),
                                            self.trainer,self.saver,self.global_episodes))
            
        # Run agents
        #config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config=tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            if self.param.load_model == True:
                print('Loading model: '+ self.param.path_load + '.')
                ckpt = tf.train.get_checkpoint_state(self.param.path_load)
                self.saver.restore(sess,ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            if self.param.xpu=='/gpu:0' and self.param.n_agents==1:
                self.agents[0].work(self.param,sess,coord,self.saver)
            elif self.param.xpu=='/gpu:0' and self.param.n_agents>1:
                raise ValueError('Multi-threading not allowed with GPU.')
            else:
                agent_threads = []
                for agent in self.agents:
                    agent_work = lambda: agent.work(sess,coord)
                    thread = threading.Thread(target=(agent_work))
                    thread.start()
                    agent_threads.append(thread)
                coord.join(agent_threads)
        print('Finished calculation: '+ self.param.datetime_start + '.')


#############
# BATCH RUN #
#############

class BatchRun():
    def __init__(self,param_batch=param_batch,param_basic=param_basic):
        self.n_param=len(param_batch)
        # Directory organization for saving
        datetime_start="{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        self.path_save_batch=param_basic['path_save_master'] + '/' + datetime_start
        if not os.path.exists(self.path_save_batch):
            os.makedirs(self.path_save_batch)

        # Batch table preparation
        batch_current_id=np.zeros((self.n_param,),dtype=np.int16) # table of ids of iteration for each parameter
        self.batch_table=pd.DataFrame()
        batch_count=0
        flag_break=0        # 1: batch current id update successfull, 2: end of recursion
        while flag_break < 2:
            for i in range(self.n_param):
                param=param_batch[i]
                if param['type']=='list':
                    self.batch_table.loc[batch_count,param['name']] = param['list'][batch_current_id[i]]
                elif param['type']=='parametric':
                    if param['method']=='grid':
                        self.batch_table.loc[batch_count,param['name']] = param['min']+(param['max']-param['min'])*batch_current_id[i]/(param['n']-1)
                    elif param['method']=='random':
                        self.batch_table.loc[batch_count,param['name']] = np.random.uniform(low=param['min'],high=param['max'])
                else:
                    raise ValueError('Incorrect batch parameter type.')
            
            param_id_level=self.n_param-1
            flag_break=0
            while flag_break < 1:
                batch_current_id[param_id_level] += 1
                if batch_current_id[param_id_level] < param_batch[param_id_level]['n']:
                    # break updating id when within limit
                    flag_break = 1
                else:
                    # reset current level to 0
                    batch_current_id[param_id_level] = 0
                    # move to the upper level
                    param_id_level -= 1
                    if param_id_level < 0:
                        # break creating list when reached end
                        flag_break = 2

            batch_count += 1

        self.batch_table.loc[:,'datetime_start']=np.NaN
        self.batch_table.loc[:,'done']=False
        self.save_batch_table()

        with open(self.path_save_batch+'/parameters_batch.json', 'w') as fp:
            json.dump(param_batch, fp, indent=1)

    def run(self):
        for i in range(len(self.batch_table)):
            print('Starting batch: ' + str(i + 1) + '/' + str(len(self.batch_table)))
            param_dict=self.batch_table.loc[i,self.batch_table.columns.difference(['datetime_start','done'])].to_dict()
            param_dict['path_save_master']=self.path_save_batch
            run=Run(param_basic=param_basic,param_change=param_dict)
            datetime_start=run.param.datetime_start
            self.batch_table.loc[i,'datetime_start']=datetime_start
            self.save_batch_table()
            run.run()
            self.batch_table.loc[i,'done']=True
            self.save_batch_table()
        print('Finished batch calculation.')
        
    def save_batch_table(self):
        hdf=pd.HDFStore(self.path_save_batch+'/batch_table.h5')
        hdf.put('batch_table',self.batch_table,format='table',append=False,data_columns=True)
        hdf.close()


print('End of file.')