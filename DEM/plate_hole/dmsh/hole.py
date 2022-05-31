#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:14:02 2022


@author: 王一铮, 447650327@qq.com
"""

import dmsh
import numpy as np
import matplotlib.pyplot as plt
rect = dmsh.Rectangle(0.0, +20.0, 0.0, 20.0)
c = dmsh.Circle([0.0, 0.0], 5.0)
geo = dmsh.Difference(rect, c)
X, cells = dmsh.generate(geo, lambda pts: np.abs(c.dist(pts))/18+0.20, tol=1.0e-15) # 划分693个点
geo.show()
plt.scatter(X[:,0], X[:,1])
