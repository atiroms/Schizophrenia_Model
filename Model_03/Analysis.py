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

dir_data='20200321_014712'

#list_dir_data=['20200218_212228','20200303_183303']
list_dir_data=['20200321_014554','20200321_014642','20200321_014712']


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
import scipy.stats as stats
import json


######################################################################
# Combine multiple batches ###########################################
######################################################################
class BatchCombine():
    def __init__(self, path_data=path_data,list_dir_data=list_dir_data):
        self.path_data=path_data
        self.df_batch=pd.DataFrame()
        for dir_data in list_dir_data:
            path_load_batch=os.path.join(path_data,dir_data)
            if os.path.exists(path_load_batch):
                # Read batch_table
                with pd.HDFStore(os.path.join(path_load_batch,'batch_table.h5')) as hdf:
                    df_batch_append = pd.DataFrame(hdf['batch_table'])
                #df_batch_append=df_batch_append.loc[df_batch['done']==True,:]
                self.df_batch=self.df_batch.append(df_batch_append)
            else:
                print("Batch dir does not exist: "+dir_data+".")
        
        self.df_batch=self.df_batch.reset_index(drop=True)
        self.df_batch=self.df_batch.drop('run',axis=1)
        print('Detected '+str(len(self.df_batch))+' runs.')

    def combine(self):
        # Timestamping directory name
        datetime_start="{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        self.path_save_batch=os.path.join(self.path_data,datetime_start)
        if not os.path.exists(self.path_save_batch):
            os.makedirs(self.path_save_batch)
        hdf=pd.HDFStore(self.path_save_batch+'/batch_table.h5')
        hdf.put('batch_table',self.df_batch,format='table',append=False,data_columns=True)
        hdf.close()
        print('Saved new batch table.')
        print('Please copy subdirectories manually.')


######################################################################
# Batch data analysis ################################################
######################################################################

class BatchAnalysis():
    def __init__(self, path_data=path_data,dir_data=dir_data,subset={}):
        self.path_data=path_data
        self.path_load_batch=os.path.join(path_data,dir_data)
        self.path_save_analysis=os.path.join(self.path_load_batch,"analysis")
        if not os.path.exists(self.path_save_analysis):
            os.makedirs(self.path_save_analysis)
        self.subset=subset

    def examine_batch(self,path_data=path_data,dir_data=dir_data):
        # Read batch_table
        with pd.HDFStore(os.path.join(self.path_load_batch,'batch_table.h5')) as hdf:
            df_batch = pd.DataFrame(hdf['batch_table'])
        print(df_batch)

    def single_plot(self,key='reward',window=1000,padding=10):
        with pd.HDFStore(self.path_load_batch+'/summary/summary.h5') as hdf:
            summary = pd.DataFrame(hdf['summary'])
        df_reward=summary[['episode',key]]

        df_reward=self.ave_reward(df_reward,window=window,padding=padding)

        print('Preparing line plot.')
        #self.path=os.path.join(path_data,dir_data)
        #self.df_ave=df_ave
        fig=plt.figure(figsize=(5,2),dpi=100)
        ax=fig.add_subplot(1,1,1)
        ax.plot(df_reward['episode'],df_reward[key])
        ax.set_title("Average reward, window: "+str(window)+", padding: "+str(padding))
        ax.set_xlabel("Task episode")
        ax.set_ylabel("Reward")
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_save_analysis,
                                 "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())+'_reward.png'))
        plt.show()

    def batch_load(self,key='reward',len_precalc=5000,list_key_batch=None):
        # Read batch_table
        with pd.HDFStore(os.path.join(self.path_load_batch,'batch_table.h5')) as hdf:
            df_batch = pd.DataFrame(hdf['batch_table'])
        
        df_batch=df_batch.loc[df_batch['done']==True,:]
        self.df_batch=df_batch

        # Subset batch table by keys and values specified in 'subset'
        if len(self.subset)>0:
            for key_subset in list(self.subset.keys()):
                df_batch_subset=df_batch.loc[df_batch[key_subset]==self.subset[key_subset]]
        else:
            df_batch_subset=df_batch

        self.n_batch=len(df_batch_subset)

        # Automatically create list of batch keys if not explicitly defined
        if list_key_batch is None:
            list_key_batch=df_batch_subset.columns.tolist()
            for key_batch in ['datetime_start','run','done']:
                if key_batch in list_key_batch:
                    list_key_batch.remove(key_batch)

        # Add or replace df_batch_subset using values from parmeters.json
        for idx_subdir in range(self.n_batch):
            subdir=df_batch_subset.loc[idx_subdir,'datetime_start']
            with open(os.path.join(self.path_load_batch,subdir,'parameters.json')) as f:
                dict_param=json.load(f)
            for key_batch in list_key_batch:
                df_batch_subset.loc[idx_subdir,key_batch]=dict_param[key_batch]

        # Create batch label and title for later plotting
        title_batch=None
        for key_batch in list_key_batch:
            if type(df_batch_subset[key_batch].tolist()[0])==str:
                label_batch_key=df_batch_subset[key_batch].tolist()
            else:
                if max(df_batch_subset[key_batch].tolist())>1:
                    regex='{0:.0f}'
                else:
                    regex='{:.4f}'
                label_batch_key=[regex.format(b) for b in df_batch_subset[key_batch].tolist()]
            if title_batch is None:
                label_batch=label_batch_key
                title_batch=key_batch
            else:
                label_batch=[a+':'+b for a,b in zip(label_batch,label_batch_key)]
                title_batch=title_batch+':'+key_batch
        self.label_batch=label_batch
        self.title_batch=title_batch

        # Read subdirectory using subset of batch table
        print('Loading data.')
        sleep(1)
        for i in tqdm(range(self.n_batch)):
            #print('\rLoading ' + str(i+1) + '/' + str(self.n_batch) + '                 ',end='')
            subdir=df_batch_subset['datetime_start'].iloc[i]
            path=self.path_load_batch + '/' + subdir

            # Load summary data
            with pd.HDFStore(path+'/summary/summary.h5') as hdf:
                summary = pd.DataFrame(hdf['summary'])
            summary=summary[['episode',key]].rename(columns={key:str(i)})

            # When the data is from pre-learned model, concatenate the last specified episodes before the data
            with open(os.path.join(path,'parameters.json')) as f:
                dict_param=json.load(f)
            if dict_param['load_model']:    
            #if 'dir_load' in dict_param.keys():
                dir_load=dict_param['dir_load']
                path_precalc=os.path.join(self.path_data,dir_load)
                with pd.HDFStore(path_precalc+'/summary/summary.h5') as hdf:
                    summary_precalc = pd.DataFrame(hdf['summary'])
                summary_precalc=summary_precalc[['episode',key]].rename(columns={key:str(i)})
                summary_precalc=summary_precalc.iloc[-len_precalc:,:]
                diff_precalc=max(summary_precalc['episode'])+1
                summary_precalc['episode']=summary_precalc['episode']-diff_precalc
                summary=pd.concat([summary_precalc,summary])
                summary=summary.reset_index(drop=True)

            # Horizontally concatenate the loaded data
            if i == 0:
                output=summary
            else:
                output=pd.merge(output,summary,how='outer', on='episode')
        #output['episode']=output['episode']-output.loc[0,'episode']
        print('Finished loading data.')
        #self.df_ave=MovAveEpisode(dataframe=self.summaries).output
        return(output)

    def reward_prob(self,window=1000,padding=10,prob_comp=1/3):
        self.win_ave=window
        self.pad_ave=padding
        df_reward=self.batch_load(key='reward')
        df_arm=self.batch_load(key='prob_arm0')
        len_out=int((len(df_reward)-window)/padding+1)
        list_column=df_reward.columns.tolist()
        list_column.remove('episode')
        df_ave=pd.DataFrame(columns=['episode_start','episode_stop']+df_reward.columns.tolist())
        df_ave_difficult=pd.DataFrame(columns=['episode_start','episode_stop']+df_reward.columns.tolist())
        df_ave_easy=pd.DataFrame(columns=['episode_start','episode_stop']+df_reward.columns.tolist())
        df_ave_diff=pd.DataFrame(columns=['episode_start','episode_stop']+df_reward.columns.tolist())
        df_ave_ratio=pd.DataFrame(columns=['episode_start','episode_stop']+df_reward.columns.tolist())
        df_slope=pd.DataFrame(columns=['episode_start','episode_stop']+df_reward.columns.tolist())

        print('Calculating averages and slopes.')
        sleep(1)
        for i in tqdm(range(len_out)):
            ave_append=[df_reward['episode'].iloc[i*padding],
                        df_reward['episode'].iloc[i*padding+window-1],
                        df_reward['episode'].iloc[i*padding:(i*padding+window)].mean()]
            ave_difficult_append=ave_append
            ave_easy_append=ave_append
            ave_diff_append=ave_append
            ave_ratio_append=ave_append
            slope_append=ave_append

            for column in list_column:
                reward_window=df_reward[column].iloc[i*padding:(i*padding+window)].tolist()
                arm0_window=df_arm[column].iloc[i*padding:(i*padding+window)].tolist()
                armdiff_window=[abs(arm0-0.5)*2 for arm0 in arm0_window]
                ave=np.mean(reward_window)

                reward_window=np.array(reward_window)
                armdiff_window=np.array(armdiff_window)
                ave_difficult=reward_window[armdiff_window<prob_comp].mean()
                ave_easy=reward_window[armdiff_window>(1-prob_comp)].mean()
                ave_diff=ave_easy-ave_difficult
                ave_ratio=ave_easy/ave_difficult
                slope, intercept, r_value, p_avlue, std_err=stats.linregress(armdiff_window,reward_window)

                ave_append=ave_append+[ave]
                ave_difficult_append=ave_difficult_append+[ave_difficult]
                ave_easy_append=ave_easy_append+[ave_easy]
                ave_diff_append=ave_diff_append+[ave_diff]
                ave_ratio_append=ave_ratio_append+[ave_ratio]
                slope_append=slope_append+[slope]

            df_ave=df_ave.append(pd.Series(ave_append,index=['episode_start','episode_stop','episode']+list_column),ignore_index=True)
            df_ave_difficult=df_ave_difficult.append(pd.Series(ave_difficult_append,index=['episode_start','episode_stop','episode']+list_column),ignore_index=True)
            df_ave_easy=df_ave_easy.append(pd.Series(ave_easy_append,index=['episode_start','episode_stop','episode']+list_column),ignore_index=True)
            df_ave_diff=df_ave_diff.append(pd.Series(ave_diff_append,index=['episode_start','episode_stop','episode']+list_column),ignore_index=True)
            df_ave_ratio=df_ave_ratio.append(pd.Series(ave_ratio_append,index=['episode_start','episode_stop','episode']+list_column),ignore_index=True)
            df_slope=df_slope.append(pd.Series(slope_append,index=['episode_start','episode_stop','episode']+list_column),ignore_index=True)
        print('Finished calculating averages and slopes of rewards.')

        self.plot_reward(df_ave)
        sleep(1)
        self.plot_reward(df_ave_difficult)
        sleep(1)
        self.plot_reward(df_ave_easy)
        sleep(1)
        self.plot_reward(df_ave_diff)
        sleep(1)
        self.plot_reward(df_ave_ratio)
        sleep(1)
        self.plot_reward(df_slope)
        return([df_ave,df_ave_difficult,df_ave_easy,df_ave_diff,df_ave_ratio,df_slope])

    def ave_reward(self,df_reward,window=100,padding=10):
        self.win_ave=window
        self.pad_ave=padding
        # len = win + (n-1) * pad    >>     n = (len - win)/pad + 1
        len_out=int((len(df_reward)-window)/padding+1)
        print('Averaging reward, window: '+str(window)+', padding: '+str(padding)+', output: '+str(len_out)+'.')
        sleep(1)
        output=pd.DataFrame(columns=['episode_start','episode_stop']+df_reward.columns.tolist())
        for i in tqdm(range(len_out)):
            output=output.append(pd.concat([pd.Series([i*padding,i*padding+window-1],index=['episode_start','episode_stop']),
                                            df_reward.iloc[i*padding:(i*padding+window),:].mean()]),ignore_index=True)
            #self.output=self.output.append(self.input.iloc[(self.interval*i):(self.interval*(i+1)),:].mean(),ignore_index=True)
        print('Finished averaging reward.')
        return(output)

    def state_reward(self,df_reward,learned=False,threshold=[65,67.5]):
        self.thresh_reward=threshold
        print('Calculating disease states.')
        sleep(1)
        col_batch=df_reward.drop(['episode','episode_start','episode_stop'],axis=1).columns.tolist()
        if learned:
            state_init=[1,0,0]
        else:
            state_init=[0,-1,0]
        # State at each episode, 0: unlearned, 1: learned 2: psychotic 3: remitted
        df_state=pd.DataFrame(state_init[0],columns=col_batch,index=df_reward.index).astype(int)
        # State history, -1: never learned, 0: has learned, N>0: N psychotic episodes
        df_count=pd.DataFrame(state_init[1],columns=col_batch,index=df_reward.index).astype(int)
        # Cumulative psychosis
        df_cumul=pd.DataFrame(state_init[2],columns=col_batch,index=df_reward.index)
        #sr_state=pd.Series(-1,index=col_batch).astype(int)
        for i in tqdm(range(1,len(df_reward))):
            for col in col_batch:
                df_cumul.loc[i,col]=df_cumul.loc[i-1,col]
                if df_reward.loc[i,col]<threshold[0]:                           # Currently below threshold
                    if df_state.loc[i-1,col]==0 or df_state.loc[i-1,col]==2:    # Stayed below threshold
                        df_state.loc[i,col]=df_state.loc[i-1,col]                  # No change
                        df_count.loc[i,col]=df_count.loc[i-1,col]
                    else:                                                       # Fell below threshold
                        df_state.loc[i,col]=2                                      # Fall psychotic                   
                        df_count.loc[i,col]=df_count.loc[i-1,col]+1                # Count up psychosis
                    if df_state.loc[i,col]==2:                                   # Currently psychotic
                        df_cumul.loc[i,col]=df_cumul.loc[i-1,col]+(threshold[0]-df_reward.loc[i,col])*self.pad_ave
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
        df_cumul=pd.concat([df_reward[['episode_start','episode_stop','episode']],df_cumul],axis=1)
        print('Finished calculating disease states')
        return([df_state,df_count,df_cumul])

    def heatmap_reward(self,df_reward):
        print('Preparing heatmap plot.')
        df_plot=df_reward.drop(['episode','episode_start','episode_stop'],axis=1).T
        df_plot.columns=df_reward['episode_start'].tolist()
        df_plot.index=self.label_batch
        self.df_plot=df_plot
        fig=plt.figure(figsize=(6,0.90+0.125*len(df_plot)),dpi=100)
        ax=fig.add_subplot(1,1,1)
        #heatmap=ax.pcolor(df_reward['episode_start'].tolist(),np.arange(self.n_batch+1),df_plot,cmap=cm.rainbow)
        heatmap=ax.pcolor(df_reward['episode'].tolist(),np.arange(self.n_batch+1),df_plot,cmap=cm.rainbow_r)
        #ax.set_xticks(np.arange(df_plot.shape[1]), minor=False)
        ax.set_yticks(np.arange(df_plot.shape[0]) + 0.5, minor=False)
        ax.invert_yaxis()
        #ax.set_xticklabels([str(int(i)) for i in df_reward['episode_start'].tolist()], minor=False)
        ax.set_yticklabels(self.label_batch, minor=False)
        ax.set_title("Average reward, window: "+str(self.win_ave)+", padding: "+str(self.pad_ave))
        ax.set_xlabel("Task episode")
        ax.set_ylabel(self.title_batch)
        cbar=fig.colorbar(heatmap,ax=ax)
        cbar.set_label('Average reward')
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_save_analysis,
                                 "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())+'_heatmap.png'))
        plt.show()

    def plot_reward(self,df_reward,select_col=None):
        print('Preparing line plot.')
        #self.path=os.path.join(path_data,dir_data)
        #self.df_ave=df_ave
        fig=plt.figure(figsize=(6,5),dpi=100)
        ax=fig.add_subplot(1,1,1)
        if select_col is None:
            select_col=[i for i in range(self.n_batch)]
        for i in range(len(select_col)):
            ax.plot(df_reward['episode'],df_reward.drop(['episode_start','episode_stop','episode'],axis=1).iloc[:,select_col[i]],
                    color=cm.rainbow(i/len(select_col)))
        ax.invert_yaxis()
        ax.set_title("Average reward, window: "+str(self.win_ave)+", padding: "+str(self.pad_ave))
        ax.set_xlabel("Task episode")
        ax.set_ylabel("Reward")
        ax.legend(title=self.title_batch,labels=[self.label_batch[i] for i in select_col],
                  bbox_to_anchor=(1.05,1),loc='upper left')
        #ax.plot(np.arange(0,x_test.shape[0],1),y_test)
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_save_analysis,
                                 "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())+'_reward.png'))
        plt.show()

    def plot_state(self,df_state):
        fig=plt.figure(figsize=(6,5),dpi=100)
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
                  bbox_to_anchor=(1.05,1),loc='upper left',fontsize=4)
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_save_analysis,
                                 "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())+'_state.png'))
        plt.show()

    def plot_count(self,df_count):
        fig=plt.figure(figsize=(6,5),dpi=100)
        ax=fig.add_subplot(1,1,1)
        for i in range(self.n_batch):
            ax.plot(df_count['episode'],df_count.drop(['episode_start','episode_stop','episode'],axis=1).iloc[:,i],
                    color=cm.rainbow(i/self.n_batch))
        ax.set_title("Count of psychotic episodes")
        ax.set_xlabel("Task episode")
        ax.set_ylabel("Count of psychotic episodes")
        ax.legend(title=self.title_batch,labels=self.label_batch,
                  bbox_to_anchor=(1.05,1),loc='upper left',fontsize=4)
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_save_analysis,
                                 "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())+'_cumul.png'))
        plt.show()

    def plot_cumulative(self,df_cumul):
        fig=plt.figure(figsize=(6,5),dpi=100)
        ax=fig.add_subplot(1,1,1)
        for i in range(self.n_batch):
            ax.plot(df_cumul['episode'],df_cumul.drop(['episode_start','episode_stop','episode'],axis=1).iloc[:,i],
                    color=cm.rainbow(i/self.n_batch))
        ax.set_title("Cumulative psychosis duration x severity")
        ax.set_xlabel("Task episode")
        ax.set_ylabel("Duration x Severity")
        ax.legend(title=self.title_batch,labels=self.label_batch,
                  bbox_to_anchor=(1.05,1),loc='upper left',fontsize=4)
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_save_analysis,
                                 "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())+'_count.png'))
        plt.show()

    def pipe_state(self,window=1000,padding=10,learned=True,threshold=[65,67.5],
                   select_col=None,list_key_batch=['learning_rate','n_cells_lstm']):
        df_batchreward=self.batch_load(list_key_batch=list_key_batch)
        df_batchrewardave=self.ave_reward(df_batchreward,window=window,padding=padding)
        data_state=self.state_reward(df_batchrewardave,learned=learned,threshold=threshold)
        self.heatmap_reward(df_batchrewardave)
        self.plot_reward(df_batchrewardave,select_col=select_col)
        self.plot_state(data_state[0])
        self.plot_count(data_state[1])
        self.plot_cumulative(data_state[2])
        return(data_state)
        
    def pipe_mov(self,window=1000,padding=10,select_col=[0,1,3,10,17]):
        df_batchreward=self.batch_load()
        df_batchrewardave=self.ave_reward(df_batchreward,window=window)
        self.plot_reward(df_batchrewardave,select_col=select_col)

    def pipe_block(self,window=100,padding=100,select_col=[0,1,3,10,17]):
        df_batchreward=self.batch_load()
        df_batchrewardave=self.ave_reward(df_batchreward,window=window,padding=padding)
        self.plot_reward(df_batchrewardave,select_col=select_col)

    def pipe_hm_mov(self,window=1000,padding=10):
        df_batchreward=self.batch_load()
        df_batchrewardave=self.ave_reward(df_batchreward,window=window,padding=padding)
        self.heatmap_reward(df_batchrewardave)

    def pipe_hm_block(self,window=100,padding=100):
        df_batchreward=self.batch_load()
        df_batchrewardave=self.ave_reward(df_batchreward,window=window,padding=padding)
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