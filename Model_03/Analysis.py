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
#dir_data = '20200216_233234'
dir_data = '20200217_103834'
#list_dir_data = ['20200216_003928']

######################################################################
# Libraries ##########################################################
######################################################################

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
#import tensorflow as tf
#import matplotlib.pyplot as plt
#import plotly as py
#import cufflinks as cf
#import glob


######################################################################
# Batch data analysis ################################################
######################################################################

class BatchAnalysis():
    def __init__(self, path_data=path_data,dir_data=dir_data,subset={}):
        self.path=os.path.join(path_data,dir_data)
        self.subset=subset

    def batch_load_reward(self):
        # Read batch_table
        with pd.HDFStore(os.path.join(self.path,'batch_table.h5')) as hdf:
            batch_table = pd.DataFrame(hdf['batch_table'])
        #batch_table = batch_table.iloc[0:10,:]

        # Subset batch table by keys and values specified in 'subset'
        if len(self.subset)>0:
            for key in list(self.subset.keys()):
                batch_table_subset=batch_table.loc[batch_table[key]==self.subset[key]]
        else:
            batch_table_subset=batch_table

        # Read subdirectory using subset of batch table
        for i in range(len(batch_table_subset)):
            print('\rLoading ' + str(i+1) + '/' + str(len(batch_table_subset)) + '                 ',end='')
            subdir=batch_table_subset['datetime_start'].iloc[i]
            path=self.path + '/' + subdir
            with pd.HDFStore(path+'/summary/summary.h5') as hdf:
                summary = pd.DataFrame(hdf['summary'])
            
            summary=summary[['episode','reward']].rename(columns={'reward':str(i)})
            if i == 0:
                output=summary
            else:
                output=pd.merge(output,summary,how='outer', on='episode')
        print('')
        #self.df_ave=MovAveEpisode(dataframe=self.summaries).output
        return(output)

    def block_ave_reward(self,df_reward,window=100):
        print('Calculating block averages over ' + str(window) + ' episodes.')
        output=pd.DataFrame(columns=df_reward.columns)
        for i in range(math.floor(len(df_reward)/window)):
            output=output.append(df_reward.iloc[i*window:(i+1)*window,:].mean(),ignore_index=True)
        print('Finished calculating block averages.')
        return(output)

    def mov_ave_reward(self,df_reward,window=100):
        print('Calculating moving averages over ' + str(window) + ' episodes.')
        output=pd.DataFrame(columns=df_reward.columns)
        for i in range(len(df_reward)-window+1):
            #print('\rCalculating ' + str(i) + '/' + str(self.length) + '                 ', end='')
            output=output.append(df_reward.iloc[i:(i+window),:].mean(),ignore_index=True)
            #self.output=self.output.append(self.input.iloc[(self.interval*i):(self.interval*(i+1)),:].mean(),ignore_index=True)
        print('Finished calculating moving averages.')
        return(output)

    def vis_reward(self,df_reward):
        #self.path=os.path.join(path_data,dir_data)
        #self.df_ave=df_ave
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.plot(df_reward['episode'],df_reward.drop('episode',axis=1))
        #ax.plot(np.arange(0,x_test.shape[0],1),y_test)
        plt.show()

    def pipe_mov(self):
        df_batchreward=self.batch_load_reward()
        df_batchrewardave=self.mov_ave_reward(df_batchreward)
        self.vis_reward(df_batchrewardave)

    def pipe_block(self):
        df_batchreward=self.batch_load_reward()
        df_batchrewardave=self.block_ave_reward(df_batchreward)
        self.vis_reward(df_batchrewardave)


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