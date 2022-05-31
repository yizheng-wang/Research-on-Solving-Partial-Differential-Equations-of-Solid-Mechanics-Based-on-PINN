# -*- coding: utf-8 -*-
"""
Created on Wed May 26 17:36:22 2021

@author: Administrator
"""
import dmsh
import numpy as np
geo = dmsh.Rectangle(0.0, 20.0, 0.0, 20.0)
p1 = dmsh.Circle([10.0, 0.0], 0.0)

X, cells = dmsh.generate(geo, lambda pts: 0.20 * p1.dist(pts)+ 0.20 , tol=1.0e-10)
dmsh.helpers.show(X, cells, geo)
#lambda pts: 0.2 + 0.23 * p1.dist(pts)
#lambda pts: 0.10 + 0.23 * p1.dist(pts)
#lambda pts: 0.20 + 0.20 * p1.dist(pts) 不错
#lambda pts: 0.05 + 0.20 * p1.dist(pts)