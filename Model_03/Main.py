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
    #'param_set':  'exp1',
    #'param_set' : 'awjuliani',
    #'param_set' : 'Wang2018_fast',
    #'param_set' : 'Wang2018_statevalue',

    'xpu' : '/cpu:0',                    # processing device allocation
    #'xpu' : '/gpu:0',

    'train' : True,
    'load_model' : False,
    'path_load' : './saved_data/20180917_011631',

    'path_save_master' : ['/media/atiroms/MORITA_HDD3/Machine_Learning/Schizophrenia_Model/saved_data',
                          'C:/Users/atiro/Documents/Machine_Learning/Schizophrenia_Model/saved_data'],
    #'path_save_master' : 'C:/Users/atiro/Documents/Machine_Learning/Schizophrenia_Model/saved_data',

    'n_agents' : 1,                       # number of agents that act in parallel

    'agent': 'A2C',

    'episode_stop' : 50000,
    #'episode_stop' : 200000,
    #'episode_stop' : 100,

    'interval_summary':1,               # interval to save simulation summary in original format
    #'interval_summary':100,
    'interval_ckpt': 1000,              # interval to save network parameters in tf default format
    #'interval_pic': 100,
    'interval_pic': 0,                  # interval to save task pictures
    'interval_activity':1,              # interval to save all activity of an episode
    #'interval_activity':0,
    'interval_var': 10,                 # interval to save trainable network variables in original format
    #'interval_var': 0,
    'interval_persist':1000,             # interval of persistent saving
    #'interval_persist':100,
    'interval_gc':100                   # interval of garbage collection
}
param_default={    # Wang 2018 parameters
    'n_cells_lstm' : 48,                  # number of cells in LSTM-RNN network
    'bootstrap_value' : 0.0,
    'environment' : 'Two_Armed_Bandit',
    'config_environment' : 'uniform',             # select "independent" for independent bandit
    'gamma' : .9,                         # 0.9 in Wang Nat Neurosci 2018, discount rate for advantage estimation and reward discounting
    'optimizer' : 'RMSProp',              # "RMSProp" in Wang 2018, "Adam" in awjuliani/meta-RL
    'learning_rate' : 0.0007,             # Wang Nat Neurosci 2018
    'cost_statevalue_estimate' : 0.05,    # 0.05 in Wang 2018, 0.5 in awjuliani/meta-RL
    'cost_entropy' : 0.05,                # 0.05 in Wang 2018 and awjuliani/meta-RL
    'dummy_counter' : 0                   # dummy counter used for batch calculation
}
param_exp1={
    'environment' : 'Dual_Assignment_with_Hold',
    'gamma' : 0.75
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
    #{'name': 'learning_rate', 'n':11, 'type':'parametric','method':'grid','min':0.0002,'max':0.0052}
    {'name': 'learning_rate', 'n':10, 'type':'parametric','method':'grid','min':0.0057,'max':0.0102},
    {'name':'dummy_counter', 'n':2, 'type':'parametric', 'method':'grid', 'min':0,'max':1}
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
import datetime
import time
import pandas as pd
import json
from Agent import *
from Network import *
from Environment import *


####################
# PARAMETERS CLASS #
####################

class Parameters():
    def __init__(self,param_basic):
        self.add_item(param_basic)
        if self.param_set == 'Wang2018':
            self.add_item(param_default)
        elif self.param_set == 'exp1':
            self.add_item(param_default)
            self.add_item(param_exp1)
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

        for i in range(len(self.param.path_save_master)):
            if os.path.exists(self.param.path_save_master[i]):
                path_save=self.param.path_save_master[i]+'/'+datetime_start
                break
            elif i==len(self.param.path_save_master)-1:
                raise ValueError('Save folder does not exist.')

        #path_save=self.param.path_save_master+'/'+datetime_start
        self.param.add_item({'datetime_start':datetime_start, 'path_save':path_save})

        if not os.path.exists(path_save):
            os.makedirs(path_save)
        for subdir in ['model','pic','summary','activity']:
            if not os.path.exists(path_save + '/' + subdir):
                os.makedirs(path_save + '/' + subdir)

        with open(path_save+'/parameters.json', 'w') as fp:
            json.dump(self.param.__dict__, fp, indent=1)

    def run(self):
        print('Data saved in: '+ self.param.path_save + '.')
        tf.reset_default_graph()
        # Setup agents for multiple threading
        with tf.device(self.param.xpu):
            # counter of total episodes defined outside A2C_Agent class
            self.episode_global = tf.Variable(0,dtype=tf.int32,name='episode_global',trainable=False)
            if self.param.optimizer == "Adam":
                self.trainer = tf.train.AdamOptimizer(learning_rate=self.param.learning_rate)
            elif self.param.optimizer == "RMSProp":
                self.trainer = tf.train.RMSPropOptimizer(learning_rate=self.param.learning_rate)
            if self.param.environment == 'Two_Armed_Bandit':
                agent_alias=Two_Armed_Bandit
            elif self.param.environment == 'Dual_Assignment_with_Hold':
                agent_alias=Dual_Assignment_with_Hold
            self.master_network = LSTM_RNN_Network(self.param,
                                                agent_alias(self.param.config_environment).n_actions,
                                                'master',None) # Generate master network
            #n_agents = multiprocessing.cpu_count() # Set agents to number of available CPU threads
            self.saver = tf.train.Saver(max_to_keep=5)
            self.agents = []
            # Create A2C_Agent classes
            for i in range(self.param.n_agents):
                self.agents.append(A2C_Agent(i,self.param,agent_alias(self.param.config_environment),
                                             self.trainer,self.saver,self.episode_global))

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
        print('Finished ID: '+ self.param.datetime_start + '.')


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