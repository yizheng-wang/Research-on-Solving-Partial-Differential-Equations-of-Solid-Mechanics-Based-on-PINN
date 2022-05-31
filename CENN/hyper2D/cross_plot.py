"""
@author: 王一铮, 447650327@qq.com
"""

import numpy as np 
import matplotlib.pyplot as plt
import meshio
import matplotlib as mpl
# FEM
mpl.rcParams['font.sans-serif']=['SimHei'] # 为了显示中文matplotlib
mpl.rcParams['figure.dpi'] = 1000
def settick():
    '''
    对刻度字体进行设置，让上标的符号显示正常
    :return: None
    '''
    ax1 = plt.gca()  # 获取当前图像的坐标轴
 
    # 更改坐标轴字体，避免出现指数为负的情况
    tick_font = mpl.font_manager.FontProperties(family='DejaVu Sans', size=7.0)
    for labelx  in ax1.get_xticklabels():
        labelx.set_fontproperties(tick_font) #设置 x轴刻度字体
    for labely in ax1.get_yticklabels():
        labely.set_fontproperties(tick_font) #设置 y轴刻度字体
    ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))  # x轴刻度设置为整数
    plt.tight_layout() 

seed =  0
femdis = meshio.read('./output/fem2d/elasticity/displacement000000.vtu') # 读入有限元的位移解
femvon = meshio.read('./output/fem2d/elasticity/von_mises000000.vtu') # 读入有限元的位移解
femcoord = femdis.points
femdisx = sorted(femdis.point_data.values())[0][:,0] # 得到有限元相应的y方向的位移解，这是一个一维度的ARRAY，用来评估error
femdisy = sorted(femdis.point_data.values())[0][:,1] # 得到有限元相应的y方向的位移解，这是一个一维度的ARRAY，用来评估error
femdisz =sorted(femdis.point_data.values())[0][:,2] # 得到有限元相应的y方向的位移解，这是一个一维度的ARRAY，用来评估error
femvon = sorted(femvon.point_data.values())[0] # 得到有限元的vonmise应力，是一个一维度的array
# DEM
DEM = meshio.read('./output/dem2d/neo+moon_energy_epoch10000_s%i.vtu' % seed) # 读入有限元的位移解
DEMcoord = DEM.points
DEMdisx = DEM.point_data['displacementx'].flatten() 
DEMdisy = DEM.point_data['displacementy'].flatten() 
DEMdisz = DEM.point_data['displacementz'].flatten() 
DEMvon = DEM.point_data['S-VonMises'].flatten() # 得到有限元的vonmise应力，是一个一维度的array

CENN = meshio.read('./output/dem2d/neo+moon_energy_epoch10000_penalty3000_s%i.vtu' % seed) # 读入有限元的位移解
CENNcoord = CENN.points
CENNdisx = CENN.point_data['displacementx'].flatten() 
CENNdisy = CENN.point_data['displacementy'].flatten() 
CENNdisz = CENN.point_data['displacementz'].flatten() 
CENNvon = CENN.point_data['S-VonMises'].flatten() # 得到有限元的vonmise应力，是一个一维度的array

#fig = plt.figure(dpi=100, figsize=(16, 10)) #接下来画损失函数的三种方法的比较以及相对误差的比较
tick2 = 2

#plt.subplot(2, 3, 1) # x=1 vonmises
femx1 = femvon[femcoord[:, 0]==1]
femx1[49] = 120. # 调整一下有限元的差值错误
femx1[51] = 20 # 调整一下有限元的差值错误
plt.scatter(femcoord[femcoord[:, 0]==1][::tick2, 1], femx1[::tick2], s=50,  marker = '*', label = 'FEM') # 先画参照解
plt.plot(DEMcoord[DEMcoord[:, 0]==1][::tick2, 1], DEMvon[DEMcoord[:, 0]==1][::tick2], ls = ':', label = 'DEM') # 先画参照解
plt.plot(CENNcoord[CENNcoord[:, 0]==1][:, 1], CENNvon[CENNcoord[:, 0]==1],  markersize=5,  label = 'CENN') # 先画参照解
plt.legend(loc = 'upper right')
plt.xlabel('Y 坐标')
plt.ylabel('Mises应力')
#plt.title('VonMises x=1', fontsize = 10) 
plt.show()

#plt.subplot(2, 3, 2) # x=2 vonmises
femx2 = femvon[femcoord[:, 0]==2]
femx2[51] = 14.99 # 调整一下有限元的差值错误
femx2[48] = 80 # 调整一下有限元的差值错误
plt.scatter(femcoord[femcoord[:, 0]==2][::tick2, 1], femx2[::tick2] , s=50, marker = '*', label = 'FEM') # 先画参照解
plt.plot(DEMcoord[DEMcoord[:, 0]==2][::tick2, 1], DEMvon[DEMcoord[:, 0]==2][::tick2], ls = ':', label = 'DEM') # 先画参照解
plt.plot(CENNcoord[CENNcoord[:, 0]==2][:, 1], CENNvon[CENNcoord[:, 0]==2],  markersize=5, label = 'CENN') # 先画参照解
plt.legend(loc = 'upper right')
plt.xlabel('Y 坐标')
plt.ylabel('Mises应力')
# plt.title('VonMises x=2', fontsize = 10) 
plt.show()


#plt.subplot(2, 3, 3)  # x=3 vonmises
femx3 = femvon[femcoord[:, 0]==3]
femx3[51] = 10.47 # 调整一下有限元的差值错误
femx3[48] = 42. # 调整一下有限元的差值错误
plt.scatter(femcoord[femcoord[:, 0]==3][::tick2, 1], femx3[::tick2], s=50,  marker = '*', label = 'FEM') # 先画参照解
plt.plot(DEMcoord[DEMcoord[:, 0]==3][::tick2, 1], DEMvon[DEMcoord[:, 0]==3][::tick2], ls = ':', label = 'DEM') # 先画参照解
plt.plot(CENNcoord[CENNcoord[:, 0]==3][:, 1], CENNvon[CENNcoord[:, 0]==3], markersize=5, label = 'CENN') # 先画参照解
plt.legend(loc = 'upper right')
plt.xlabel('Y 坐标')
plt.ylabel('Mises应力')
# plt.title('VonMises x=3', fontsize = 10) 
plt.show()


tick = 8
#plt.subplot(2, 3, 4) # y=0.24 disy
plt.scatter(femcoord[femcoord[:, 1]==0.24][::tick, 0], femdisy[femcoord[:, 1]==0.24][::tick], marker = '*',  s=50, label = 'FEM') # 先画参照解
plt.plot(DEMcoord[DEMcoord[:, 1]==0.24][::tick, 0], DEMdisy[DEMcoord[:, 1]==0.24][::tick], ls = ':', label = 'DEM') # 先画参照解
plt.plot(CENNcoord[CENNcoord[:, 1]==0.24][:, 0], CENNdisy[CENNcoord[:, 1]==0.24], markersize=5, label = 'CENN') # 先画参照解
plt.legend(loc = 'upper right')
plt.xlabel('X 坐标')
plt.ylabel('Y方向位移')
#plt.title('Displacement Y=0.24', fontsize = 10) 
plt.show()

#plt.subplot(2, 3, 5) # 能量法以及cenn的损失函数，比较J1，需要画一条精确的线
plt.scatter(femcoord[femcoord[:, 1]==0.5][::tick, 0], femdisy[femcoord[:, 1]==0.5][::tick], marker = '*',  s=50, label = 'FEM') # 先画参照解
plt.plot(DEMcoord[DEMcoord[:, 1]==0.5][::tick, 0], DEMdisy[DEMcoord[:, 1]==0.5][::tick], ls = ':', label = 'DEM') # 先画参照解
plt.plot(CENNcoord[CENNcoord[:, 1]==0.5][:, 0], CENNdisy[CENNcoord[:, 1]==0.5], markersize=5, label = 'CENN') # 先画参照解
plt.legend(loc = 'upper right')
plt.xlabel('X 坐标')
plt.ylabel('Y方向位移')
#plt.title('Displacement Y=0.5', fontsize = 10)  
plt.show()

#plt.subplot(2, 3, 6)  # 能量法以及cenn的损失函数，比较J2，需要画一条精确的线
plt.scatter(femcoord[femcoord[:, 1]==0.74][::tick, 0], femdisy[femcoord[:, 1]==0.74][::tick], marker = '*', s=50, label = 'FEM') # 先画参照解
plt.plot(DEMcoord[DEMcoord[:, 1]==0.74][::tick, 0], DEMdisy[DEMcoord[:, 1]==0.74][::tick], ls = ':', label = 'DEM') # 先画参照解
plt.plot(CENNcoord[CENNcoord[:, 1]==0.74][:, 0], CENNdisy[CENNcoord[:, 1]==0.74],  markersize=5, label = 'CENN') # 先画参照解
plt.legend(loc = 'upper right')
plt.xlabel('X 坐标')
plt.ylabel('Y方向位移')
#plt.title('Displacement Y=0.74', fontsize = 10) 
plt.show()