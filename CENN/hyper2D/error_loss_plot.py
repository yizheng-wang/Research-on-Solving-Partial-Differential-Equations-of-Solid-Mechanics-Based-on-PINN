"""
@author: 王一铮, 447650327@qq.com
"""
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    
    
mpl.rcParams['font.sans-serif']=['SimHei'] # 为了显示中文matplotlib
mpl.rcParams['figure.dpi'] = 1000
seed = 0
DEMloss = np.load('./loss_error/hyper_energy_loss_epoch10000_s%i.npy' % seed)
CENNloss = np.load('./loss_error/hyper_energy_loss_epoch10000_penalty3000_s%i.npy' % seed)
# DEMloss[DEMloss>5] = DEMloss[DEMloss>5]/100
# CENNloss[CENNloss>5] = CENNloss[CENNloss>5]/100
DEML2 = np.load('./loss_error/hyper_energy_errorL2_epoch10000_s%i.npy' % seed)
CENNL2 = np.load('./loss_error/hyper_energy_errorL2_epoch10000_penalty3000_s%i.npy' % seed)

# DEMVON = np.load('./loss_error/hyper_energy_errorvon_epoch300_s%i.npy' % seed)
# CENNVON = np.load('./loss_error/hyper_energy_errorvon_epoch300_penalty2000_s%i.npy' % seed)

#fig = plt.figure(dpi=1000, figsize=(15, 3)) #接下来画损失函数的三种方法的比较以及相对误差的比较


plt.ylim(-1.5, -0.5)
plt.xlim(0, 10000)
iteration = np.array(range(0, 10000, 10))
plt.plot(iteration, DEMloss[::10], '--')
plt.plot(iteration, CENNloss[::10], '-.')
plt.legend([ 'DEM', 'CENN'], loc = 'upper right')
plt.xlabel('Iteration')
plt.ylabel('Loss')
#plt.title('Loss comparision of DEM and CENN', fontsize = 10) 
plt.show()


plt.yscale('log')
plt.xlim(0, 10000)
iteration = np.array(range(0, 10000, 10))
plt.plot(iteration, DEML2, '--')
plt.plot(iteration, CENNL2, '-.')
plt.legend([ 'DEM', 'CENN'], loc = 'upper right')
plt.xlabel('Iteration')
plt.ylabel(r'$L_{2}$ error')
settick()
#plt.title('L2 comparision of DEM and CENN', fontsize = 10) 
plt.show()
