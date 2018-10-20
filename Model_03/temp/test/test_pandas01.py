


import pandas as pd
import numpy as np

df=pd.DataFrame(np.random.randn(5,3), columns=['A','B','C'])

hdf=pd.HDFStore('storage.h5')
hdf.put('d1',df,format='table',data_columns=True)


hdf['d1']

hdf.append('d1',pd.DataFrame(np.random.rand(5,3), 
           columns=('A','B','C')), 
           format='table', data_columns=True)


hdf.close()

hdf=pd.read_hdf('storage.h5','d1')

hdf=pd.read_hdf('storage.h5')

hdf

df