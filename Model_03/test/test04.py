import numpy as np

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])

c=np.concatenate(a)

d=np.concatenate((c,b))

c=a.ravel()

d=b.ravel()

c

d

e=np.concatenate((c))

e=np.concatenate((c,d))

f=np.empty(shape=[0,])
f.shape
c.shape
e=np.concatenate((f,c))
e

a.b=0
a=1
a.b=1
a=None
a

a.b=1


for i in [1,2,3]:
    print(i)


a='test'

b={
    a+'1':2
}



df=pd.DataFrame(columns=['a','b'],index=range(5))
df
df.iloc[0,1]=1



import numpy as np
import pandas as pd

df=pd.DataFrame(columns=['col1','col2'])

a=[1.5,2.5]

b=[int(1.5),2.5]

df.loc[len(df)]=a
df.loc[len(df)]=b

print(df)

df=pd.DataFrame(columns=['col1','col2'],index=range(10))

print(df)

df.loc[:,['col1','col2']]=[1,2]

df.loc[:,['col3','col4']]=[1,2

df.assign(col3=3,col4=4)

df.dtypes

df.loc[:,'col1'].astype('int64')

df.loc[:,'col1']=df.loc[:,'col1'].astype('int64')

df

hdf=pd.HDFStore('test.h5')
hdf.put('test',df,format='table',append=True,data_columns=True)
hdf.close()