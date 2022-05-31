# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 09:50:06 2021

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np 
import matplotlib as mpl
plt.rcParams['figure.figsize'] = (8, 6.4)
mpl.rcParams['font.sans-serif']=['SimHei'] # 为了显示中文matplotlib
plt.rcParams['font.family'] = ['sans-serif'] # 用来正常显示负号
mpl.rcParams['figure.dpi'] = 1000
y0 = np.load('y0.npy')
mise_abaqusy0 = np.load('mise_abaqusy0.npy')
pred_mise_r = np.load('pre_mise_4000.npy')
plt.plot(y0, mise_abaqusy0, marker='*', ls = ':', ms=5, label = r'Abaqus(参照解)', lw = 2)    
plt.plot(y0, pred_mise_r, color = 'red', marker='o', ms=4, label = 'DEM', lw = 2)
plt.xlabel('x坐标') 
plt.ylabel('Mise')
plt.xlim(10, 20)
#plt.title('mise : y=0,x>0, number points : 596' )
plt.legend(fontsize=20)
plt.show()