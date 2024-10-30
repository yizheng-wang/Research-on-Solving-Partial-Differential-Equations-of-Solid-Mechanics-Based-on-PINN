# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 22:38:04 2022

@author: yludragon
"""
import numpy as np
import sys

def plot_inner_hete(N,mode=0):
    if mode==0:
        x=np.zeros([N*N,2])
        x[:,0]=np.repeat(np.linspace(-2.48,2.48,N,endpoint=True),N)
        x[:,1]=np.tile(np.linspace(-2.48,2.48,N,endpoint=True),N)     
    # elif mode==1:
    #     x=np.zeros([N,2])
    #     x[:,0]=np.linspace(0.0,2.0,N)
    #     x[:,1]=0
          
    LR=np.linalg.norm(x,axis=1)
    L1 = np.where(LR>= 1*1.02)[0]
    L2 = np.where(LR<=1*0.98)[0]
    x1 = x[L1]
    x2 = x[L2]
    x = np.concatenate([x1,x2],axis=0)
    return x

y = plot_inner_hete(200)
N_node = y.shape[0]
y=np.concatenate([y,np.zeros([N_node,1])],axis=1)
y=y.tolist()
z = [tuple(coo) for coo in y ]
z = tuple(z)
f=open('coo.txt','w')
print(z,file=f)
f.close()



