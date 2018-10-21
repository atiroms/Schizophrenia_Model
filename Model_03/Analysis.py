###############
# DESCRIPTION #
###############

# Python code to analyze saved data files generated by meta-RL program.


##############
# PARAMETERS #
##############

path_data_master=['/media/atiroms/MORITA_HDD3/Machine_Learning/Schizophrenia_Model/saved_data',
                  'C:/Users/atiro/Documents/Machine_Learning/Schizophrenia_Model/saved_data']

#path_data = '20180914_000352'   # summary saved every 50 episodes
#path_data = '20180918_211807'
#path_data = '20180920_130605'
#path_data = '20180921_011111'
#path_data = '20180923_114142'
#path_data = '20180924_175630'
#path_data = '20180924_235841'
#path_data = '20180926_002716'
#path_data = '20180928_233909'
#path_data = '20180929_001701'
#path_data = '20181002_004726'
#path_data = '20181002_010133'
#path_data = '20181010_054352'
#path_data = '20181015_171743'
#path_data = '20181018_160331'
#path_data = '20181021_164116'
#path_data = '20181021_164922'
path_data = '20181021_190448'

paths_data=[
    '20180918_211807',
    '20180921_011111',
    '20180923_114142',
    '20180924_175630',
    '20180924_235841',
    '20180926_002716',
    '20180928_233909',
]


#############
# LIBRARIES #
#############

import numpy as np
import pandas as pd
import tensorflow as tf
#import matplotlib.pyplot as plt
import plotly as py
import cufflinks as cf
import glob
import os
import math


############################
# DATA FOLDER CONFIRMATION #
############################

for i in range(len(path_data_master)):
    if os.path.exists(path_data_master[i]):
        path_data=path_data_master[i]+'/'+path_data
        break
    elif i==len(path_data_master)-1:
        raise ValueError('Save folder does not exist.')


##############################
# DATA EXTRACTION AND SAVING #
##############################

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


#####################
# HDF5 DATA LOADING #
#####################

class Load_Summary():
    def __init__(self,path_data=path_data):
        print('Starting hdf5 file loading.')
        self.path_data=path_data + '/summary'
        hdf = pd.HDFStore(self.path_data+'/summary.h5')
        self.output = pd.DataFrame(hdf['summary'])
        hdf.close()
        print('Finished hdf5 file loading.')

class Load_Summary_Old():
    def __init__(self,path_data=path_data):
        print('Starting hdf5 file loading.')
        self.path_data=path_data + '/summary'
        hdf = pd.HDFStore(self.path_data+'/summary.h5')
        self.output = pd.DataFrame(hdf['summary'])
        self.output.columns=hdf['colnames'].tolist()
        hdf.close()
        print('Finished hdf5 file loading.')

class Load_Activity():
    def __init__(self,path_data=path_data):
        print('Starting hdf5 file loading.')
        self.path_data=path_data + '/activity'
        hdf = pd.HDFStore(self.path_data+'/activity.h5')
        self.output = pd.DataFrame(hdf['activity'])
        hdf.close()
        print('Finished hdf5 file loading.')


class Load_Variable():
    def __init__(self,path_data=path_data):
        print('Starting hdf5 file loading.')
        self.path_data=path_data + '/model'
        hdf = pd.HDFStore(self.path_data+'/variable.h5')
        self.output = pd.DataFrame(hdf['variable'])
        hdf.close()
        print('Finished hdf5 file loading.')



#################
# VISUALIZATION #
#################

class Visualize():
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


#########################
# AVERAGE OVER EPISODES #
#########################

class AveEpisode():
    def __init__(self,dataframe,interval):
        self.input=dataframe
        self.interval=interval
        self.length=math.floor(len(self.input)/self.interval)
        self.calc_average()

    def calc_average(self):
        print('Starting calculation of averages.')
        print('Calculating averages over ' + str(self.interval) + ' episodes.')
        self.output=pd.DataFrame(columns=self.input.columns)
        for i in range(self.length):
            self.output=self.output.append(self.input.iloc[(self.interval*i):(self.interval*(i+1)),:].mean(),ignore_index=True)
        print('Finished calculating averages.')


##############################
# REWARD AVERAGE GRAPH BATCH #
##############################

class RewardAverageGraphBatch():
    def __init__(self,paths_data=paths_data):
        for p in paths_data:
            print('Calculating ' + p + '.')
            df=Load_Summary(path_data=p).output
            ave=AveEpisode(dataframe=df,interval=100).output
            vis=Visualize(dataframe=ave,path_data=p)
        print('Finished batch calculation.')


print('End of file.')