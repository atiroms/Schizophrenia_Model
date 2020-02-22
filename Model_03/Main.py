######################################################################
# Description ########################################################
######################################################################
'''
Python code for meta reinforcement learning
For a single simulation,
  sim=Sim()
  sim.run()
For batch simulations,
  batch=Batch()
  batch.run()
'''


######################################################################
# Parameters #########################################################
######################################################################

#set_param_sim='param_sim.json'
set_param_sim='param_sim_long.json'
#set_param_sim='param_test.json'
set_param_mod='param_wang2018.json'
#set_param_mod='param_wang2018_parallel.json'

#dir_restart='20200219_223846'
#dir_restart='20200221_234851'
dir_restart='20200222_002120'
#dir_restart=None

param_batch=[
    #{'name': 'learning_rate', 'n':11, 'type':'parametric','method':'grid','min':0.0002,'max':0.0052}
    #{'name': 'learning_rate', 'n':10, 'type':'parametric','method':'grid','min':0.0057,'max':0.0102},
    #{'name': 'learning_rate', 'n':100, 'type':'parametric','method':'grid','min':0.0001,'max':0.0100},
    #{'name': 'learning_rate', 'n':2, 'type':'parametric','method':'grid','min':0.0001,'max':0.0100},
    {'name':'dummy_counter', 'n':3, 'type':'parametric', 'method':'grid', 'min':0,'max':2}
    #{'name':'learning_rate', 'n':5, 'type':'parametric', 'method':'random', 'min':0.0001, 'max':0.001},
    #{'name':'optimizer', 'n':2, 'type':'list','list':['RMSProp','Adam']}
    #{'name':'gamma','n':3,'type':'parametric','method':'grid','min':0.7,'max':0.9}
    #{'name': 'n_cells_lstm', 'n':20, 'type':'parametric','method':'grid','min':5,'max':100}
    #{'name': 'learning_rate', 'n':19, 'type':'parametric','method':'grid','min':0.0001,'max':0.0019},
    #{'name': 'learning_rate', 'n':17, 'type':'parametric','method':'grid','min':0.002,'max':0.01}
    #{'name': 'episode_stop', 'n':5, 'type':'parametric','method':'grid','min':50000,'max':0.01}
]


######################################################################
# Libraries ##########################################################
######################################################################

import os
list_path_code=[
    'D:/atiroms/GitHub/Schizophrenia_Model/Model_03',
    'C:/Users/atiro/GitHub/Schizophrenia_Model/Model_03',
    '/home/atiroms/GitHub/Schizophrenia_Model/Model_03'
]
for i in range(len(list_path_code)):
    if os.path.exists(list_path_code[i]):
        path_code=list_path_code[i]
        os.chdir(path_code)
        break
    elif i==len(list_path_code)-1:
        raise ValueError('Code folder does not exist in the list.')
list_path_save=[
    "/media/veracrypt1/Machine_Learning/Schizophrenia_Model/saved_data",
    "/media/atiroms/MORITA_HDD3/Machine_Learning/Schizophrenia_Model/saved_data",
    "C:/Users/atiro/Documents/Machine_Learning/Schizophrenia_Model/saved_data",
    "D:/Machine_Learning/Schizophrenia_Model/saved_data",
    "F:/Machine_Learning/Schizophrenia_Model/saved_data"
]
for i in range(len(list_path_save)):
    if os.path.exists(list_path_save[i]):
        path_save=list_path_save[i]
        break
    elif i==len(list_path_save)-1:
        raise ValueError('Save folder does not exist in the list.')

import threading
#import multiprocessing
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import time
import pandas as pd
import json
import Agent
import Network
import Environment


######################################################################
# Parameters class for parameter exchange between classes ############
######################################################################

class Parameters():
    def __init__(self,set_param,path_code=path_code):
        self.set=None
        self.add_json(set_param,path_code)

    def add_dict(self,dict_param):
        for key,value in dict_param.items():
            if key!="//":
                setattr(self,key,value)

    def add_json(self,set_param,path_code=path_code):
        with open(os.path.join(path_code,"parameters",set_param)) as f:
            dict_param=json.load(f)
            self.add_dict(dict_param)
        if self.set is None:
            self.set=list()
        self.set.append(set_param)


######################################################################
# Single run of simulation ###########################################
######################################################################

class Sim():
    #def __init__(self,param_basic=param_basic,param_change=None):
    def __init__(self,set_param_sim=set_param_sim,set_param_mod=set_param_mod,
                 set_param_overwrite=None,
                 path_code=path_code,path_save=path_save):

        # Timestamping directory name
        datetime_start="{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

        # Setup parameters in Parmeters object
        self.param=Parameters(set_param_sim,path_code)
        self.param.add_json(set_param_mod,path_code)
        self.param.add_dict({'datetime_start':datetime_start, 'path_save':os.path.join(path_save,datetime_start)})
        if set_param_overwrite is not None:
            self.param.add_dict(set_param_overwrite)

        # Make directories for saving
        if not os.path.exists(self.param.path_save):
            os.makedirs(self.param.path_save)
        for subdir in ['model','pic','summary','activity']:
            if not os.path.exists(os.path.join(self.param.path_save,subdir)):
                os.makedirs(os.path.join(self.param.path_save,subdir))
        
        # Save parameters
        with open(os.path.join(self.param.path_save,'parameters.json'), 'w') as fp:
            json.dump(self.param.__dict__, fp, indent=1)

    def run(self):
        print('Running: '+ self.param.datetime_start + '.')
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
                agent_alias=Environment.Two_Armed_Bandit
            elif self.param.environment == 'Dual_Assignment_with_Hold':
                agent_alias=Environment.Dual_Assignment_with_Hold
            self.master_network = Network.LSTM_RNN_Network(self.param,
                                                agent_alias(self.param.config_environment).n_actions,
                                                'master',None) # Generate master network
            #n_agents = multiprocessing.cpu_count() # Set agents to number of available CPU threads
            self.saver = tf.train.Saver(max_to_keep=5)
            self.agents = []
            # Create A2C_Agent classes
            for i in range(self.param.n_agents):
                self.agents.append(Agent.A2C_Agent(i,self.param,agent_alias(self.param.config_environment),
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
            #if self.param.xpu=='/gpu:0' and self.param.n_agents==1:
            if self.param.n_agents==1:
                self.agents[0].work(sess,coord)
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
        print('Done single run: '+ self.param.datetime_start + '.')


######################################################################
# Batch run of simulations ###########################################
######################################################################

class Batch():
    def __init__(self,param_batch=param_batch,path_save=path_save,
                 dir_restart=dir_restart):
        if dir_restart is None:
            self.prep(param_batch=param_batch,path_save=path_save)
        else:
            self.prep_restart(dir_restart=dir_restart,path_save=path_save)

    def prep_restart(self,dir_restart,path_save):
        self.path_save_batch=os.path.join(path_save,dir_restart)
        if os.path.exists(os.path.join(self.path_save_batch,"batch_table.h5")):
            with pd.HDFStore(os.path.join(self.path_save_batch,"batch_table.h5")) as hdf:
                self.batch_table = pd.DataFrame(hdf['batch_table'])
            self.batch_table.loc[:,'run']=False
            list_idx_rerun=self.batch_table.loc[self.batch_table['done']==False,:].index.values.tolist()
            print('Unfinished runs: '+str(len(list_idx_rerun))+'.')
            for i in list_idx_rerun:
                sr_append=self.batch_table.loc[i,:]
                sr_append['datetime_start']=np.NaN
                sr_append['run']=True
                sr_append['done']=False
                self.batch_table=self.batch_table.append(sr_append)
            self.batch_table=self.batch_table.reset_index(drop=True)
            self.save_batch_table()
        else:
            print('dir_restart not found: '+dir_restart)

    def prep(self,param_batch=param_batch,path_save=path_save):
        self.n_param=len(param_batch)
        # Timestamping directory name
        datetime_start="{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        self.path_save_batch=os.path.join(path_save,datetime_start)
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
        self.batch_table.loc[:,'run']=True
        self.batch_table.loc[:,'done']=False
        self.save_batch_table()

        with open(self.path_save_batch+'/parameters_batch.json', 'w') as fp:
            json.dump(param_batch, fp, indent=1)

        print('Done batch setup.')

    def run(self):
        batch_table_run=self.batch_table.loc[self.batch_table['run']==True,:]
        #for i in range(len(self.batch_table)):
        list_idx_run=batch_table_run.index.values.tolist()
        for i in range(len(list_idx_run)):
            idx=list_idx_run[i]
            print('Batch simulation: ' + str(i + 1) + '/' + str(len(list_idx_run)),'.')
            param_overwrite=self.batch_table.loc[idx,self.batch_table.columns.difference(['datetime_start','run','done'])].to_dict()
            param_overwrite['path_save_batch']=self.path_save_batch
            sim=Sim(path_save=self.path_save_batch,set_param_overwrite=param_overwrite)
            self.batch_table.loc[idx,'datetime_start']=sim.param.datetime_start
            self.save_batch_table()
            sim.run()
            self.batch_table.loc[idx,'done']=True
            self.save_batch_table()
        print('Done batch simulation.')

    def save_batch_table(self):
        hdf=pd.HDFStore(self.path_save_batch+'/batch_table.h5')
        hdf.put('batch_table',self.batch_table,format='table',append=False,data_columns=True)
        hdf.close()

print('End of file.')