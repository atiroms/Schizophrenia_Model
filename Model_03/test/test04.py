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

