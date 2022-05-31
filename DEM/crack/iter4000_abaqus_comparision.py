#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 00:11:49 2021

@author: sg
"""

import matplotlib.pyplot as plt
import numpy as np
mise_abaqusy0 = np.load('mise_abaqusy0.npy')
pred_mise = np.load('pre_mise_4000.npy')
y0 = np.load('y0.npy')
plt.plot(y0-10, mise_abaqusy0, color = 'red', label = 'abaqus solution', lw = 2) 
plt.plot(y0-10, pred_mise, label = 'deep Nitsche ritz solution')   
plt.xlabel('r coordinate') 
plt.xlim(0, 10)
plt.ylabel('mise')
plt.title('mise : y=0,x>0, number points : 596')
plt.legend(fontsize=7)
plt.show()