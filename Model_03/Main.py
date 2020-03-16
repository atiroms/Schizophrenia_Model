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

set_param_sim='param_sim.json'
#set_param_sim='param_sim_gpu.json'
#set_param_sim='param_sim_pic.json'
#set_param_sim='param_sim_long.json'
#set_param_sim='param_test.json'

#set_param_mod='param_wang2018.json'
set_param_mod='param_wang2018_small.json'
#set_param_mod='param_wang2018_parallel.json'

#dir_restart='20200219_223846'
#dir_restart='20200221_234851'
#dir_restart='20200222_002120'
#dir_restart='20200222_233321'
#dir_restart='20200224_234232'
#dir_restart='20200226_161100'
#dir_restart='20200228_130159'
#dir_restart='20200229_214730'
#dir_restart='20200302_062250'
#dir_restart='20200311_234904'
#dir_restart='20200311_235109'
#dir_restart='20200311_235702'
dir_restart=None

#dir_load='20200222_002120/20200222_122717'
dir_load='20200314_202346'
#dir_load=None

param_batch=[
    #{'name':'dummy_counter', 'n':3, 'type':'parametric', 'method':'grid', 'min':0,'max':2}
    #{'name':'optimizer', 'n':2, 'type':'list','list':['RMSProp','Adam']}
    #{'name':'gamma','n':3,'type':'parametric','method':'grid','min':0.7,'max':0.9}
    #{'name': 'n_cells_lstm', 'n':20, 'type':'parametric','method':'grid','min':5,'max':100}
    #{'name': 'learning_rate', 'n':19, 'type':'parametric','method':'grid','min':0.0001,'max':0.0019},
    #{'name': 'learning_rate', 'n':17, 'type':'parametric','method':'grid','min':0.002,'max':0.01}
    #{'name': 'learning_rate', 'n':18, 'type':'parametric','method':'grid','min':0.015,'max':0.100}
    #{'name': 'n_cells_lstm', 'n':3, 'type':'parametric','method':'grid','min':36,'max':60}
    #{'name': 'n_cells_lstm', 'n':13, 'type':'parametric','method':'grid','min':12,'max':60}
    #{'name': 'n_cells_lstm', 'n':2, 'type':'parametric','method':'grid','min':4,'max':8}
    #{'name': 'n_cells_lstm', 'n':11, 'type':'parametric','method':'grid','min':1,'max':11}
    #{'name': 'n_cells_lstm', 'n':24, 'type':'parametric','method':'grid','min':2,'max':48}
    #{'name': 'n_cells_lstm', 'n':10, 'type':'parametric','method':'grid','min':110,'max':200}
    {'name': 'n_cells_lstm', 'n':12, 'type':'parametric','method':'grid','min':2,'max':24}
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
import random
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
                 path_code=path_code,path_save=path_save,
                 path_save_batch=None,
                 dir_load=dir_load):

        # Timestamping directory name
        datetime_start="{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        if path_save_batch is not None:
            path_save_run=os.path.join(path_save_batch,datetime_start)
        else:
            path_save_run=os.path.join(path_save,datetime_start)
        self.path_save=path_save

        # Setup parameters in Parmeters object
        self.param=Parameters(set_param_sim,path_code)
        self.param.add_json(set_param_mod,path_code)
        self.param.add_dict({'datetime_start':datetime_start, 'path_save':path_save_run})
        if set_param_overwrite is not None:
            self.param.add_dict(set_param_overwrite)
        if dir_load is not None:
            '''
            with open(os.path.join(path_save,dir_load,"parameters.json")) as f:
                dict_param=json.load(f)
            episode_done=dict_param['episode_stop']
            episode_stop=episode_done+self.param.episode_stop
            print('episode_stop='+str(episode_done)+'+'+str(self.param.episode_stop))
            self.param.add_dict({'load_model':1,'dir_load':dir_load,'episode_stop':episode_stop})
            '''
            self.param.add_dict({'load_model':1,'dir_load':dir_load})
        else:
            self.param.add_dict({'load_model':0})


        # Make directories for saving
        if not os.path.exists(self.param.path_save):
            os.makedirs(self.param.path_save)
        for subdir in ['model','pic','summary','activity']:
            if not os.path.exists(os.path.join(self.param.path_save,subdir)):
                os.makedirs(os.path.join(self.param.path_save,subdir))
        
        # Save parameters
        with open(os.path.join(self.param.path_save,'parameters.json'), 'w') as fp:
            json.dump(self.param.__dict__, fp, indent=1)

    def replace_uncontigious(self,ary_src,ary_rep,idx_row,idx_col):
        ary_dst=ary_src
        if len(idx_row)!=ary_rep.shape[0]:
            print('idx_row and ary_rep 0th dim do not match: '+str(len(idx_row))+', '+str(ary_rep.shape[0]))
        if len(idx_col)!=ary_rep.shape[1]:
            print('idx_col and ary_rep 1st dim do not match: '+str(len(idx_col))+', '+str(ary_rep.shape[1]))
        for i in range(len(idx_row)):
            for j in range(len(idx_col)):
                ary_dst[idx_row[i],idx_col[j]]=ary_rep[i,j]
        return(ary_dst)

    def load_graph(self,list_ary_dst,env_alias):
        # Reshape saved graph variables to fit into newly initialized graph.
        # Enabled for different LSTM cell numbers between saved and new graph,
        # Which TF default loading does not support.

        # Load source graph specs
        with open(os.path.join(self.path_save,self.param.dir_load,'parameters.json')) as f:
            dict_param=json.load(f)
        n_cells_src=int(dict_param['n_cells_lstm'])
        n_actions=env_alias(self.param.config_environment).n_actions

        # Load source graph variables
        with pd.HDFStore(os.path.join(self.path_save,self.param.dir_load,'model/variable.h5')) as hdf:
            df_var_src = pd.DataFrame(hdf['variable'])
        df_var_src=df_var_src.loc[df_var_src['episode']==max(df_var_src['episode']),:]

        # Reshape source graph variables into arrays
        ary_var_src=np.asarray(df_var_src['value'],order='c').astype('float32')
        ary_kernel_src,ary_bias_src,ary_fc0_src,ary_fc1_src=np.split(ary_var_src,
                                                     [(n_actions+2+n_cells_src)*4*n_cells_src,
                                                      (n_actions+3+n_cells_src)*4*n_cells_src,
                                                      ((n_actions+3+n_cells_src)*4+n_actions)*n_cells_src])
        ary_kernel_src=ary_kernel_src.reshape([n_actions+2+n_cells_src,4*n_cells_src])
        ary_bias_src=ary_bias_src.reshape([4*n_cells_src])
        ary_fc0_src=ary_fc0_src.reshape([n_cells_src,n_actions])
        ary_fc1_src=ary_fc1_src.reshape([n_cells_src,1])

        # Load variables from initialized destination graph
        ary_kernel_dst,ary_bias_dst,ary_fc0_dst,ary_fc1_dst = list_ary_dst

        # Overwrite destination graph variable arrays, after deletion if necessary
        n_cells_dst=int(self.param.n_cells_lstm)
        if n_cells_dst==n_cells_src:
            ary_kernel_dst=ary_kernel_src
            ary_bias_dst=ary_bias_src
            ary_fc0_dst=ary_fc0_src
            ary_fc1_dst=ary_fc1_src
            print("Preserved "+str(int(n_cells_src))+" LSTM cells.")
        elif n_cells_dst<n_cells_src:
            # idx_del=np.arange(n_cells_dst,n_cells_src)
            idx_del=random.sample(np.arange(n_cells_src).tolist(),n_cells_src-n_cells_dst)
            idx_del=np.sort(np.asarray(idx_del,dtype='int64'))
            idx_del_4=[]
            for i in range(4):
                idx_del_4=np.concatenate([idx_del_4,idx_del+n_cells_src*i])
            idx_del_4=idx_del_4.astype('int64')
            ary_kernel_dst=np.delete(ary_kernel_src,idx_del+n_actions+2,0)
            ary_kernel_dst=np.delete(ary_kernel_dst,idx_del_4,1)
            ary_bias_dst=np.delete(ary_bias_src,idx_del_4,0)
            ary_fc0_dst=np.delete(ary_fc0_src,idx_del,0)
            ary_fc1_dst=np.delete(ary_fc1_src,idx_del,0)
            print("Deleted "+str(int(n_cells_src-n_cells_dst))+" LSTM cells.")
        elif n_cells_dst>n_cells_src:
            #idx_ow=np.arange(n_cells_src)
            idx_ow=random.sample(np.arange(n_cells_dst).tolist(),n_cells_src)
            idx_ow=np.sort(np.asarray(idx_ow,dtype='int64'))
            idx_ow_4=[]
            for i in range(4):
                idx_ow_4=np.concatenate([idx_ow_4,idx_ow+n_cells_dst*i])
            idx_ow_4=idx_ow_4.astype('int64')
            ary_kernel_dst=self.replace_uncontigious(ary_kernel_dst,ary_kernel_src,
                                                      np.concatenate([np.arange(n_actions+2),
                                                                      idx_ow+n_actions+2]),
                                                      idx_ow_4)
            ary_bias_dst[idx_ow_4]=ary_bias_src
            ary_fc0_dst=self.replace_uncontigious(ary_fc0_dst,ary_fc0_src,
                                                   idx_ow,np.arange(n_actions))
            ary_fc1_dst=self.replace_uncontigious(ary_fc1_dst,ary_fc1_src,
                                                   idx_ow,[0])
            print("Added "+str(int(n_cells_dst-n_cells_src))+" LSTM cells.")
        
        return([ary_kernel_dst,ary_bias_dst,ary_fc0_dst,ary_fc1_dst])

    def run(self):
        print('Simulating: '+ self.param.datetime_start + '.')
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
                env_alias=Environment.Two_Armed_Bandit
            elif self.param.environment == 'Dual_Assignment_with_Hold':
                env_alias=Environment.Dual_Assignment_with_Hold
            # Generate master network
            self.master_network = Network.LSTM_RNN(self.param,
                                                           env_alias(self.param.config_environment).n_actions,
                                                           'master',None) 
            #n_agents = multiprocessing.cpu_count() # Set agents to number of available CPU threads
            self.saver = tf.train.Saver(max_to_keep=5)
            self.agents = []
            # Create A2C_Agent classes (local network is defined within agent definition)
            for i in range(self.param.n_agents):
                self.agents.append(Agent.A2C_Agent(i,self.param,env_alias(self.param.config_environment),
                                                   self.trainer,self.saver,self.episode_global))

        # Run agents
        #config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config=tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if self.param.load_model == True:
                # Load variables from initialized destination graph
                vars_master = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'master')
                list_ary_dst=sess.run(vars_master)
                # New original graph loading
                list_ary_dst=self.load_graph(list_ary_dst,env_alias)
                # Assign destination arrays to TF tensors
                sess.run([vars_master[0].assign(list_ary_dst[0]),
                          vars_master[1].assign(list_ary_dst[1]),
                          vars_master[2].assign(list_ary_dst[2]),
                          vars_master[3].assign(list_ary_dst[3])])

                # TensorFlow default graph data loading
                # Only for the same sized graph
                '''
                path_load=os.path.join(self.path_save,self.param.dir_load)
                ckpt = tf.train.get_checkpoint_state(os.path.join(path_load,'model'))
                self.saver.restore(sess,ckpt.model_checkpoint_path)
                '''
                print('Loaded parameters from '+ self.param.dir_load + '.')

            coord = tf.train.Coordinator()
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
        print('Done single simulation: '+ self.param.datetime_start + '.')


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
                self.batch_table=self.batch_table.drop(i)
                sr_append['datetime_start']=np.NaN
                sr_append['run']=True
                sr_append['done']=False
                self.batch_table=self.batch_table.append(sr_append)

            self.batch_table=self.batch_table.reset_index(drop=True)
            self.save_batch_table()
            print('Done batch setup in restarting mode.')
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
            param_overwrite=self.batch_table.loc[idx,self.batch_table.columns.difference(['datetime_start','run','done'])].to_dict()
            print('Batch simulation: ' + str(i + 1) + '/' + str(len(list_idx_run))+' '+str(param_overwrite))
            param_overwrite['path_save_batch']=self.path_save_batch
            sim=Sim(path_save_batch=self.path_save_batch,set_param_overwrite=param_overwrite)
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