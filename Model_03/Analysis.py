######################################################################
# Description ########################################################
######################################################################
'''
Python code to analyze saved data files generated by meta-RL program.

Data to be extracted:
activity/activity.h5
    for each timestep within an episode at intervals specified in interval_activity,
    actions, rewards, values, etc...
model/checkpoint
    for each episode at intervals specified in interval_ckpt,
    model parameters in tensorflow default format
model/variable.h5
    for each episode at intervals specified in interval_var,
    all the trainable variables of network
pic/
    for each episode at intervals specified in interval_pic,
    gif images (movie) of all task actions within episode
summary/summary.h5:
    for each episode at intervals specified in interval_summary,
    total reward, task arm probabilities, learning losses, etc...
'''

######################################################################
# Parameters #########################################################
######################################################################
import os
list_path_data=[
    "/media/veracrypt1/Machine_Learning/Schizophrenia_Model/saved_data",
    "/media/atiroms/MORITA_HDD3/Machine_Learning/Schizophrenia_Model/saved_data",
    "C:/Users/atiro/Documents/Machine_Learning/Schizophrenia_Model/saved_data",
    "D:/Machine_Learning/Schizophrenia_Model/saved_data",
    "F:/Machine_Learning/Schizophrenia_Model/saved_data"
]
for i in range(len(list_path_data)):
    if os.path.exists(list_path_data[i]):
        path_data=list_path_data[i]
        break
    elif i==len(list_path_data)-1:
        raise ValueError('Data folder does not exist in the list.')

#dir_data = '20200216_191229'
#dir_data = '20200216_204436'
#dir_data = '20200216_233234' # n_lstm_cell 4, 15, ... 48
#dir_data = '20200217_103834'
dir_data='20200218_212228' # n_lstm_cell 5, 10, ... 100
#dir_data='20200219_223846' # learning_rate 0.0001, 0.0002, ... 0.0019 (0.0019 failed)
#dir_data='20200220_230830' # learning_rate 0.0020, 0.0025, ... 0.0100

######################################################################
# Libraries ##########################################################
######################################################################

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from time import sleep
import datetime
#import tensorflow as tf
#import plotly as py
#import cufflinks as cf
#import glob


######################################################################
# Batch data analysis ################################################
######################################################################

class BatchAnalysis():
    def __init__(self, path_data=path_data,dir_data=dir_data,subset={}):
        self.path=os.path.join(path_data,dir_data)
        self.path_analysis=os.path.join(self.path,"analysis")
        if not os.path.exists(self.path_analysis):
            os.makedirs(self.path_analysis)
        self.subset=subset

    def batch_load_reward(self):
        # Read batch_table
        with pd.HDFStore(os.path.join(self.path,'batch_table.h5')) as hdf:
            df_batch = pd.DataFrame(hdf['batch_table'])
        #df_batch = df_batch.iloc[0:10,:]
        self.df_batch=df_batch

        # Subset batch table by keys and values specified in 'subset'
        if len(self.subset)>0:
            for key in list(self.subset.keys()):
                df_batch_subset=df_batch.loc[df_batch[key]==self.subset[key]]
        else:
            df_batch_subset=df_batch
        column_batchlabel=df_batch_subset.columns.tolist()
        for column in ['datetime_start','done']:
            column_batchlabel.remove(column)

        self.n_batch=len(df_batch_subset)
        label_batch=['']*self.n_batch
        for column in column_batchlabel:
            if label_batch[0]=='':
                label_batch=[str(b) for b in df_batch_subset[column].tolist()]
                title_batch=column
            else:
                label_batch=[a+'_'+str(b) for a,b in zip(label_batch,df_batch_subset[column].tolist())]
                title_batch=title_batch+'_'+column
        self.label_batch=label_batch
        self.title_batch=title_batch

        # Read subdirectory using subset of batch table
        print('Loading data.')
        sleep(1)
        for i in tqdm(range(self.n_batch)):
            #print('\rLoading ' + str(i+1) + '/' + str(self.n_batch) + '                 ',end='')
            subdir=df_batch_subset['datetime_start'].iloc[i]
            path=self.path + '/' + subdir
            with pd.HDFStore(path+'/summary/summary.h5') as hdf:
                summary = pd.DataFrame(hdf['summary'])
            
            summary=summary[['episode','reward']].rename(columns={'reward':str(i)})
            if i == 0:
                output=summary
            else:
                output=pd.merge(output,summary,how='outer', on='episode')
        print('Finished loading data.')
        #self.df_ave=MovAveEpisode(dataframe=self.summaries).output
        return(output)

    def block_ave_reward(self,df_reward,window=100):
        self.win_ave=window
        print('Calculating block averages over ' + str(window) + ' episodes.')
        sleep(1)
        output=pd.DataFrame(columns=['episode_start','episode_stop']+df_reward.columns.tolist())
        for i in tqdm(range(math.floor(len(df_reward)/window))):
            output=output.append(pd.concat([pd.Series([i*window,(i+1)*window-1],index=['episode_start','episode_stop']),
                                            df_reward.iloc[i*window:(i+1)*window,:].mean()]),ignore_index=True)
        print('Finished calculating block averages.')
        return(output)

    def mov_ave_reward(self,df_reward,window=100):
        self.win_ave=window
        print('Calculating moving averages over ' + str(window) + ' episodes.')
        sleep(1)
        output=pd.DataFrame(columns=['episode_start','episode_stop']+df_reward.columns.tolist())
        for i in tqdm(range(len(df_reward)-window+1)):
            output=output.append(pd.concat([pd.Series([i,i+window-1],index=['episode_start','episode_stop']),
                                            df_reward.iloc[i:(i+window),:].mean()]),ignore_index=True)
            #self.output=self.output.append(self.input.iloc[(self.interval*i):(self.interval*(i+1)),:].mean(),ignore_index=True)
        print('Finished calculating moving averages.')
        return(output)

    def state_reward(self,df_reward,threshold=[65,67.5]):
        self.thresh_reward=threshold
        print('Calculating disease states.')
        sleep(1)
        col_batch=df_reward.drop(['episode','episode_start','episode_stop'],axis=1).columns.tolist()
        # State at each episode, 0: unlearned, 1: learned 2: psychotic 3: remitted
        df_state=pd.DataFrame(0,columns=col_batch,index=df_reward.index).astype(int)
        # State history, -1: never learned, 0: has learned, >1: N psychotic episodes
        df_count=pd.DataFrame(-1,columns=col_batch,index=df_reward.index).astype(int)
        #sr_state=pd.Series(-1,index=col_batch).astype(int)
        for i in tqdm(range(1,len(df_reward))):
            for col in col_batch:
                if df_reward.loc[i,col]<threshold[0]:                           # Currently below threshold
                    if df_state.loc[i-1,col]==0 or df_state.loc[i-1,col]==2:    # Stayed below threshold
                        df_state.loc[i,col]=df_state.loc[i-1,col]                  # No change
                        df_count.loc[i,col]=df_count.loc[i-1,col]
                    else:                                                       # Fell below threshold
                        df_state.loc[i,col]=2                                      # Fall psychotic                   
                        df_count.loc[i,col]=df_count.loc[i-1,col]+1                # Count up psychosis
                elif df_reward.loc[i,col]>=threshold[1]:                        # Currently above threshold
                    if df_state.loc[i-1,col]==0 or df_state.loc[i-1,col]==2:    # Climbed above threshold
                        if df_count.loc[i-1,col]==-1:                                   # Learned for the first time
                            df_state.loc[i,col]=1
                            df_count.loc[i,col]=0
                        else:                                                   # Remitted
                            df_state.loc[i,col]=3
                            df_count.loc[i,col]=df_count.loc[i-1,col]
                    else:                                                       # Stayed above threshold
                        df_state.loc[i,col]=df_state.loc[i-1,col]                  # No change
                        df_count.loc[i,col]=df_count.loc[i-1,col]
                else:                                                           # Currently gray zone
                    df_state.loc[i,col]=df_state.loc[i-1,col]                      # No change
                    df_count.loc[i,col]=df_count.loc[i-1,col]
        df_state=pd.concat([df_reward[['episode_start','episode_stop','episode']],df_state],axis=1)
        df_count=pd.concat([df_reward[['episode_start','episode_stop','episode']],df_count],axis=1)
        print('Finished calculating disease states')
        return([df_state,df_count])

    def heatmap_reward(self,df_reward):
        #self.path=os.path.join(path_data,dir_data)
        #self.df_ave=df_ave
        df_plot=df_reward.drop(['episode','episode_start','episode_stop'],axis=1).T
        df_plot.columns=df_reward['episode_start'].tolist()
        df_plot.index=self.label_batch
        self.df_plot=df_plot
        fig=plt.figure(figsize=(6,4),dpi=100)
        ax=fig.add_subplot(1,1,1)
        heatmap=ax.pcolor(df_reward['episode_start'].tolist(),np.arange(self.n_batch+1),df_plot,cmap=cm.rainbow)
        #ax.set_xticks(np.arange(df_plot.shape[1]), minor=False)
        ax.set_yticks(np.arange(df_plot.shape[0]) + 0.5, minor=False)
        ax.invert_yaxis()
        #ax.set_xticklabels([str(int(i)) for i in df_reward['episode_start'].tolist()], minor=False)
        ax.set_yticklabels(self.label_batch, minor=False)
        ax.set_title("Average reward over "+str(self.win_ave)+" episodes")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Batch ("+self.title_batch+")")
        cbar=fig.colorbar(heatmap,ax=ax)
        cbar.set_label('Average reward')
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_analysis,
                                 "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())+'_heatmap.png'))
        plt.show()

    def plot_reward(self,df_reward):
        #self.path=os.path.join(path_data,dir_data)
        #self.df_ave=df_ave
        fig=plt.figure(figsize=(6,6),dpi=100)
        ax=fig.add_subplot(1,1,1)
        for i in range(self.n_batch):
            ax.plot(df_reward['episode'],df_reward.drop(['episode_start','episode_stop','episode'],axis=1).iloc[:,i],
                    color=cm.rainbow(i/self.n_batch))
        ax.set_title("Average reward over "+str(self.win_ave)+" episodes")
        ax.set_xlabel("Task episode")
        ax.set_ylabel("Reward")
        ax.legend(title=self.title_batch,labels=self.label_batch,
                  bbox_to_anchor=(1.05,1),loc='upper left')
        #ax.plot(np.arange(0,x_test.shape[0],1),y_test)
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_analysis,
                                 "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())+'_reward.png'))
        plt.show()

    def plot_state(self,df_state):
        fig=plt.figure(figsize=(6,6),dpi=100)
        ax=fig.add_subplot(1,1,1)
        for i in range(self.n_batch):
            ax.plot(df_state['episode'],df_state.drop(['episode_start','episode_stop','episode'],axis=1).iloc[:,i],
                    color=cm.rainbow(i/self.n_batch))
        ax.set_yticks([0,1,2,3], minor=False)
        ax.set_yticklabels(['unlearned','learned','psychotic','remitted'], minor=False)
        ax.set_title("Disease state transision")
        ax.set_xlabel("Task episode")
        ax.set_ylabel("State")
        ax.legend(title=self.title_batch,labels=self.label_batch,
                  bbox_to_anchor=(1.05,1),loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_analysis,
                                 "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())+'_state.png'))
        plt.show()

    def plot_count(self,df_count):
        fig=plt.figure(figsize=(6,6),dpi=100)
        ax=fig.add_subplot(1,1,1)
        for i in range(self.n_batch):
            ax.plot(df_count['episode'],df_count.drop(['episode_start','episode_stop','episode'],axis=1).iloc[:,i],
                    color=cm.rainbow(i/self.n_batch))
        ax.set_title("Count of psychotic episodes")
        ax.set_xlabel("Task episode")
        ax.set_ylabel("Count of psychotic episodes")
        ax.legend(title=self.title_batch,labels=self.label_batch,
                  bbox_to_anchor=(1.05,1),loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_analysis,
                                 "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())+'_count.png'))
        plt.show()

    def pipe_state(self,window=1000):
        df_batchreward=self.batch_load_reward()
        df_batchrewardave=self.mov_ave_reward(df_batchreward,window=window)
        data_state=self.state_reward(df_batchrewardave)
        self.plot_reward(df_batchrewardave)
        self.plot_state(data_state[0])
        self.plot_count(data_state[1])

    def pipe_mov(self,window=1000):
        df_batchreward=self.batch_load_reward()
        df_batchrewardave=self.mov_ave_reward(df_batchreward,window=window)
        self.plot_reward(df_batchrewardave)

    def pipe_block(self,window=100):
        df_batchreward=self.batch_load_reward()
        df_batchrewardave=self.block_ave_reward(df_batchreward,window=window)
        self.plot_reward(df_batchrewardave)

    def pipe_hm_mov(self,window=100):
        df_batchreward=self.batch_load_reward()
        df_batchrewardave=self.mov_ave_reward(df_batchreward,window=window)
        self.heatmap_reward(df_batchrewardave)

    def pipe_hm_block(self,window=100):
        df_batchreward=self.batch_load_reward()
        df_batchrewardave=self.block_ave_reward(df_batchreward,window=window)
        self.heatmap_reward(df_batchrewardave)


######################################################################
# Average summary data to smooth for vizualization ###################
######################################################################
'''
class MovAveEpisode():
    def __init__(self,dataframe,window=100,column=[]):
        self.input=dataframe
        if len(column)>0:
            column_selected=['episode']+column
            self.input=self.input[column_selected]
        self.window=window
        self.length=len(self.input)-self.window+1
        #self.length=math.floor(len(self.input)/self.window)
        self.calc_movave()

    def calc_movave(self):
        print('Calculating moving averages over ' + str(self.window) + ' episodes.')
        self.output=pd.DataFrame(columns=self.input.columns)
        for i in range(self.length):
            #print('\rCalculating ' + str(i) + '/' + str(self.length) + '                 ', end='')
            self.output=self.output.append(self.input.iloc[i:(i+self.window),:].mean(),ignore_index=True)
            #self.output=self.output.append(self.input.iloc[(self.interval*i):(self.interval*(i+1)),:].mean(),ignore_index=True)
        print('Finished calculating moving averages.')
'''

######################################################################
# HDF5 data loading for each type of data ############################
######################################################################
'''
class Load_Activity():
    def __init__(self,path_data=None):
        if path_data is None:
            path=Confirm_Datafolder().path_output
        print('Starting hdf5 file loading.')
        self.path=path + '/activity'
        hdf = pd.HDFStore(self.path+'/activity.h5')
        self.output = pd.DataFrame(hdf['activity'])
        hdf.close()
        print('Finished hdf5 file loading.')

class Load_Variable():
    def __init__(self,path_data=None):
        if path_data is None:
            path=Confirm_Datafolder().path_output
        print('Starting hdf5 file loading.')
        self.path=path + '/model'
        hdf = pd.HDFStore(self.path+'/variable.h5')
        self.output = pd.DataFrame(hdf['variable'])
        hdf.close()
        print('Finished hdf5 file loading.')
'''


######################################################################
# Data folder configuration ##########################################
######################################################################
'''
class Confirm_Datafolder():
    def __init__(self,path_data=path_data,path_data_master=path_data_master):
        for i in range(len(path_data_master)):
            if os.path.exists(path_data_master[i]):
                path_data=path_data_master[i]+'/'+path_data
                break
            elif i==len(path_data_master)-1:
                raise ValueError('Save folder does not exist.')
        self.path_output=path_data
'''


######################################################################
# Visualization ######################################################
######################################################################
'''
class RewardAverageGraphBatch():
    def __init__(self,paths_data=paths_data):
        for p in paths_data:
            print('Calculating ' + p + '.')
            df=Load_Summary(path_data=p).output
            ave=Average_Episode(dataframe=df,extent=100).output
            _=Visualize(dataframe=ave,path_data=p)
        print('Finished batch calculation.')
'''
'''
class VisAve():   
    def __init__(self,df_ave,path_data=path_data,dir_data=dir_data):
        self.path=os.path.join(path_data,dir_data)
        self.df_ave=df_ave
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.plot(self.df_ave['episode'],self.df_ave.drop('episode',axis=1))
        #ax.plot(np.arange(0,x_test.shape[0],1),y_test)
        plt.show()

        #fig = self.df_ave.iplot(
        #    kind="scatter", asFigure=True,x='episode', title='Reward - Episode',
        #    #xTitle='Episode', yTitle='Reward', colors=['blue'])
        #    xTitle='Episode', yTitle='Reward')
        ##fig = df.iplot(kind="scatter",  asFigure=True,x='Simulation/Global Episode Count', y='Perf/Reward')
        #py.offline.plot(fig, filename=self.path + '/Reward.html')
'''
'''
class Vis():
    def __init__(self,dataframe,path_data=path_data,key='reward'):
        self.df=dataframe
        self.path_data=path_data
        self.key=key
        #cf.set_config_file(offline=True, theme="white", offline_show_link=False)
        #cf.go_offline()
        #df.plot(x='Simulation/Global Episode Count', y='Performance/Reward')
        #plt.show()
        #df.iplot(kind="scatter", mode='markers', x='Simulation/Global Episode Count', y='Performance/Reward')

        fig = self.df.iplot(
            kind="scatter", asFigure=True,x='episode', y=key,
            title='Reward - Episode', xTitle='Episode', yTitle='Reward',
            colors=['blue'])
        #fig = df.iplot(kind="scatter",  asFigure=True,x='Simulation/Global Episode Count', y='Perf/Reward')
        py.offline.plot(fig, filename=self.path_data + '/Reward.html')
        print('Generated graph.')
'''

######################################################################
# Data Extraction and saving #########################################
######################################################################
'''
class Extract_Checkpoint():
    def __init__(self,path_data=path_data):
        # Collect summary files
        self.path_data=path_data + '/summary'
        self.paths_data = glob.glob(os.path.join(self.path_data, '*', 'event*'))

        # Extract data from summary files
        print('Starting data extraction.')
        for p in self.paths_data:
            count=0
            for e in tf.train.summary_iterator(p):
                print('Extracting episode ' + str(int(e.step)), end='/r')
                if count==1:
                #if count==0:
                    colnames=['Simulation/Global Episode Count']+[v.tag for v in e.summary.value]
                    self.output=pd.DataFrame(columns=colnames)

                if count>0:
                #if count>-1:
                    data=[e.step]+[v.simple_value for v in e.summary.value]
                    self.output.loc[count]=data
                count+=1

        print('/n')
        print('Finished data extraction. ' + str(count) + ' timepoints.')
        print('Saving extracted data.')

        # Save summary files in hdf5 format
        # '/' cannot be used as column names when stored in hdf5, so column names are stored separately
        colnames=pd.Series(self.output.columns.values,index=('col'+str(i) for i in range(self.output.shape[1])))
        self.output.columns=list('col'+str(i) for i in range(self.output.shape[1]))
        hdf=pd.HDFStore(self.path_data+'/summary.h5')
        hdf.put('summary',self.output,format='table',append=False,data_columns=True)
        hdf.put('colnames',colnames)
        hdf.close()
        self.output.columns=colnames.tolist()
        print('Finished saving data.')
'''

print('End of file.')