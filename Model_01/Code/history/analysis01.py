data_path='/home/atiroms/Documents/SiS/data/20171123235347711948/mainlog.log'

import numpy as np
import re
import matplotlib.pyplot as plt
data_file=open(data_path, 'r')
data_content=data_file.read()

t_re=re.compile('t:')
r_re=re.compile('r:')
a_re=re.compile('a:')
t2_re=re.compile('t2:')
piloss_re=re.compile('pi_loss:')
vloss_re=re.compile('v_loss:')
t3_re=re.compile('local_step:')
lr_re=re.compile('lr:')
R_re=re.compile('R:')
digit_re=re.compile('([+-]?[0-9]+\.?[0-9]*)')

t1list=[]
rlist=[]
alist=[]

t2list=[]
pilosslist=[]
vlosslist=[]

t3list=[]
lrlist=[]
Rlist=[]

idx=0
while idx>=0:
    t1search=t_re.search(data_content, idx)
    if t1search:
        idx=t1search.end()
        digitsearch=digit_re.search(data_content, idx)
        t1list.append(int(digitsearch.group()))
        idx=digitsearch.end()

        rsearch=r_re.search(data_content, idx)
        idx=rsearch.end()
        digitsearch=digit_re.search(data_content, idx)
        rlist.append(float(digitsearch.group()))
        idx=digitsearch.end()

        asearch=a_re.search(data_content, idx)
        idx=asearch.end()
        digitsearch=digit_re.search(data_content, idx)
        alist.append(float(digitsearch.group()))
        idx=digitsearch.end()
    else:
        break


idx=0
while idx>=0:
    t2search=t2_re.search(data_content, idx)
    if t2search:
        idx=t2search.end()
        digitsearch=digit_re.search(data_content, idx)
        t2list.append(int(digitsearch.group()))
        idx=digitsearch.end()

        pilosssearch=piloss_re.search(data_content, idx)
        idx=pilosssearch.end()
        digitsearch=digit_re.search(data_content, idx)
        pilosslist.append(float(digitsearch.group()))
        idx=digitsearch.end()

        vlosssearch=vloss_re.search(data_content, idx)
        idx=vlosssearch.end()
        digitsearch=digit_re.search(data_content, idx)
        vlosslist.append(float(digitsearch.group()))
        idx=digitsearch.end()
    else:
        break

idx=0
while idx>=0:
    t3search=t3_re.search(data_content, idx)
    if t3search:
        idx=t3search.end()
        digitsearch=digit_re.search(data_content, idx)
        t3list.append(int(digitsearch.group()))
        idx=digitsearch.end()

        lrsearch=lr_re.search(data_content, idx)
        idx=lrsearch.end()
        digitsearch=digit_re.search(data_content, idx)
        lrlist.append(float(digitsearch.group()))
        idx=digitsearch.end()

        Rsearch=R_re.search(data_content, idx)
        idx=Rsearch.end()
        digitsearch=digit_re.search(data_content, idx)
        Rlist.append(float(digitsearch.group()))
        idx=digitsearch.end()
    else:
        break

#print(t1list)
#print(t2list)
#print(t3list)
#print(rlist)
#print(alist)
#print(lrlist)
#print(Rlist)
plt.subplot(2, 2, 1)
#plt.plot(t1list, alist)
plt.plot(t1list, rlist)
plt.legend()
plt.xlabel('t')
plt.subplot(2, 2, 2)
plt.plot(t2list, pilosslist)
plt.plot(t2list, vlosslist)
plt.legend()
plt.xlabel('t')
plt.subplot(2, 2, 3)
plt.plot(t3list, lrlist)
plt.legend()
plt.xlabel('t')
plt.subplot(2, 2, 4)
plt.plot(t3list, Rlist)
plt.legend()
plt.xlabel('t')
plt.show()
