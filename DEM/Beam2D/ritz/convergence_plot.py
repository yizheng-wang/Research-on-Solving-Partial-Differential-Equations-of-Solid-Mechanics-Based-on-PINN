#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 10:36:17 2021

@author: sg
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei'] # 为了显示中文matplotlib

mpl.rcParams['figure.dpi'] = 1000

pred_num = np.load('pred_num.npy')

points_num = np.load('points_num.npy')

abaqus_y = np.load('abaqus_y.npy')

#  画曲线评估
plt.plot(points_num[4:19], abaqus_y[4:19], label = 'Abaqus参照解')
plt.plot(points_num[4:19], pred_num[4:19], marker='o', color = 'red',label = 'DEM', )
plt.ylabel('y方向位移')
plt.xlabel('配点数')
#plt.title('y方向最大位移')
plt.legend()
plt.show()
