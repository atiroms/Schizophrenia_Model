testdata = [[1,2,3],[4,5,6]]

testdata

import numpy as np

testdata = np.array(testdata)

testdata.shape[0]

import pandas as pd

df=pd.DataFrame(testdata)
df.columns=['a','b','c']

df.insert(loc=0,column='d',value=7)

df['d','a']=df['d','a'].astype('int64')

df.ix[:,['d','b']]=df.ix[:,['d','b']].astype('float64')

df.info()


df.iloc[1,:]

df.iloc[1,:].to_dict()


for i,row in df.iterrows():
    datadict=row.to_dict()
    print(datadict)

df=pd.DataFrame()

df


for i in range(len(df)):
    dictionary=df.iloc[i,:].to_dict()
    print(dictionary)