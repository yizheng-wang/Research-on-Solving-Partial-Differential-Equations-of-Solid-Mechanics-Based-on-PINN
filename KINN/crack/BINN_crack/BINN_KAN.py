# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 23:20:52 2022

@author: yludragon
"""

import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import Bie_trainer_rizzo_again
import Geometry_rizzo_HB_again
#import Geometry_potential_arc
#import Geometry_potential_more_arc
from kan import*
from  torch.autograd import grad

def setup_seed(seed):
# random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2024)


class MultiLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)

        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)

        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        y1 = torch.tanh(self.linear1(x))
        y2 = torch.tanh(self.linear2(y1))
        y3 = torch.tanh(self.linear3(y2))
        y4 = torch.tanh(self.linear4(y3))
        y = self.linear5(y4)
        return y
    

    
def exactvalue_crack_III(x,norm=0,bctype = 0,para = 0):
    r= np.linalg.norm(x,axis=1)
    costheta = x[:,0]/r
    sintheta = x[:,1]/r
    sintheta2 = np.sqrt((1-costheta)/2)
    costheta2 = np.sqrt((1+costheta)/2)
    #这里没有用对偶边界积分方程，传统边界元法只能取一半分析
    # sign = np.where(x[:,1]<0)[0]
    # sintheta2[sign] = -sintheta2[sign]
    
    if bctype == 0:
        y = np.sqrt(r)*sintheta2
    elif bctype ==1:
        
        dfdr =  0.5/(np.sqrt(r))*sintheta2
        dfdt = 0.5/(np.sqrt(r))*costheta2
        dfd1 = dfdr*costheta - dfdt*sintheta
        dfd2 = dfdr*sintheta + dfdt*costheta
        
        if para==0:
            y = dfd1*norm[0]+dfd2*norm[1]
        elif para==1:
            y = dfd1*norm[:,0]+dfd2*norm[:,1]
    # if bctype == 0:
    #     y = 10*(x[:,0]*x[:,0]-x[:,1]*x[:,1])
    # elif bctype ==1:
    #     if para==0:
    #         y = 10*((2*x[:,0])*norm[0]+(-2*x[:,1])*norm[1])
    #     elif para==1:
    #         y = 10*((2*x[:,0])*norm[:,0]+(-2*x[:,1])*norm[:,1])
    
    return np.reshape(y,[-1,1])  

def exactvalue_parabolic(x,norm=0,bctype = 0,para = 0):
    if bctype == 0:
        y = 10*(x[:,0]*x[:,0]-x[:,1]*x[:,1]-0.25*x[:,1]*x[:,0])
    elif bctype ==1:
        if para==0:
            y = 10*((2*x[:,0]-0.25*x[:,1])*norm[0]+(-2*x[:,1]-0.25*x[:,0])*norm[1])
        elif para==1:
            y = 10*((2*x[:,0]-0.25*x[:,1])*norm[:,0]+(-2*x[:,1]-0.25*x[:,0])*norm[:,1])
    # if bctype == 0:
    #     y = 10*(x[:,0]*x[:,0]-x[:,1]*x[:,1])
    # elif bctype ==1:
    #     if para==0:
    #         y = 10*((2*x[:,0])*norm[0]+(-2*x[:,1])*norm[1])
    #     elif para==1:
    #         y = 10*((2*x[:,0])*norm[:,0]+(-2*x[:,1])*norm[:,1])
    
    return np.reshape(y,[-1,1])  


def exactvalue_linear(x,norm=0,bctype = 0,para = 0):
    if bctype == 0:
        y = 50*x[:,0]-30*x[:,1]
    elif bctype ==1:
        if para==0:
            y = (50*norm[0]-30*norm[1])*np.ones([x.shape[0],1])
        elif para==1:
            y = (50*norm[:,0]-30*norm[:,1])
    return np.reshape(y,[-1,1])  



def exactvalue_sscc(x,norm=0,bctype = 0,para=0):
    if bctype == 0:
        y = np.sin(x[:,0])*np.sinh(x[:,1])+np.cos(x[:,0])*np.cosh(x[:,1])
    elif bctype ==1:
        if para==0:
            y = (np.cos(x[:,0])*np.sinh(x[:,0])+np.sin(x[:,0])*np.cosh(x[:,0]))*norm[0]+\
                (-np.sin(x[:,1])*np.cosh(x[:,1])+np.cos(x[:,1])*np.sinh(x[:,1]))*norm[1]
        elif para==1:
            y = (np.cos(x[:,0])*np.sinh(x[:,1])-np.sin(x[:,0])*np.cosh(x[:,1]))*norm[:,0]+\
                (np.sin(x[:,0])*np.cosh(x[:,1])+np.cos(x[:,0])*np.sinh(x[:,1]))*norm[:,1]
   
    return np.reshape(y,[-1,1]) 

def exactvalue_zero(x,norm=0,bctype = 0,para=0):
    if bctype == 0:
        y = x[:,0]*0
    elif bctype ==1:
        if para==0:
            y = x[:,0]*0
        elif para==1:
            y = x[:,0]*0
   
    return np.reshape(y,[-1,1]) 

def exactvalue_cylinder(x,norm=0,bctype = 0,para=1):
    #势函数
    v0=3
    R=1.5
    r2=x[:,0]*x[:,0]+x[:,1]*x[:,1]
    if bctype == 0:
        
        y = R*R*v0*x[:,0]/r2
    elif bctype ==1:
        if para==0:
            print("check  para")
        elif para==1:
            y = R*R*v0*(x[:,1]*x[:,1]-x[:,0]*x[:,0])/r2/r2*norm[:,0]+\
                (-R*R*v0*2*x[:,0]*x[:,1]/r2/r2)*norm[:,1]
   
    return np.reshape(y,[-1,1]) 

def exactvalue_cylinder_flux(x,norm=0,bctype = 0,para=1):
    #流函数
    v0=3
    R=1.5
    r2=x[:,0]*x[:,0]+x[:,1]*x[:,1]
    if bctype == 0:
        
        y = -R*R*v0*x[:,1]/r2
    elif bctype ==1:
        if para==0:
            print("check  para")
        elif para==1:
            y = R*R*v0*(2*x[:,0]*x[:,1])/r2/r2*norm[:,0]+\
                (-R*R*v0*(x[:,0]*x[:,0]-x[:,1]*x[:,1])/r2/r2)*norm[:,1]
   
    return np.reshape(y,[-1,1]) 

def exactvalue_flux(x,norm=0,bctype = 0,para=1):
    
    R=1.5
    
    r=np.linalg.norm(x,axis=-1)
    if bctype == 0:
        
        y = v0*np.log(r)
    elif bctype ==1:
        if para==0:
            print("check  para")
        elif para==1:
            y = v0*(norm[:,0]*x[:,0]+norm[:,1]*x[:,1])/r/r
   
    return np.reshape(y,[-1,1]) 

def draw_colormap(x, y, fvalue,title='',mode=0):
 
     # 目前只有上半部分，拼成一个整体   
    x_ex =  np.vstack((x, x))
    y_ex =  np.vstack((-y[::-1], y))
    fvalue_ex = np.vstack((fvalue[::-1], fvalue))
    
    h1 = plt.contourf(x_ex, y_ex, fvalue_ex, levels=100 ,cmap = 'jet')

    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h1).ax.set_title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    if mode == 1: # save fig
        plt.savefig('../results/BINN_KAN/abs_error_BINN_KAN.pdf', dpi = 300)
    plt.show()  
    
def get_inner_error(Grids0,Net,N,area = np.array([0,0,2,2])):
   
    x0=area[0]-area[2]/2+0.03
    x1=area[0]+area[2]/2-0.03
    y0 = area[1]-area[3]/2+0.03
    y1 = area[1]+area[3]/2-0.03
    
    x=np.zeros([N*N,2])
    x[:,0]=np.repeat(np.linspace(x0,x1,N),N)
    x[:,1]=np.tile(np.linspace(y0,y1,N),N)
    u_pred=Grids0.inner(Net,x).cpu().numpy()
    u_exact = Grids0.func(x).reshape([-1])
    error = np.abs(u_exact-u_pred)
    error_t = np.linalg.norm(error)/np.linalg.norm(u_exact)
    
    return error_t

def plot_inner(Grids0,Net,N,area=np.array([0,0,2,2]),device='cpu'):
    
    x=np.zeros([N*N,2])
    
    #确定绘图范围
    #三型裂纹
    x = np.linspace(area[0]-area[2]/2+0.03, area[0]+area[2]/2-0.03, N).astype(np.float32)
    y = np.linspace(area[1]-area[3]/2+0.03, area[1]+area[3]/2-0.03, N).astype(np.float32)
    x, y = np.meshgrid(x, y)
    xy_test = np.stack((x.flatten(), y.flatten()),1)    
    

    #花朵
    # x0=area[0]-area[2]/2*0.95
    # x1=area[0]+area[2]/2*0.95
    # y0 = area[1]-area[3]/2*0.95
    # y1 = area[1]+area[3]/2*0.95
    #圆柱绕流
    # x0=area[0]-area[2]/2
    # x1=area[0]+area[2]/2
    # y0 = area[1]-area[3]/2
    # y1 = area[1]+area[3]/2
    
    # SX=np.linspace(x0,x1,N)
    # SY=np.linspace(y0,y1,N)
    # [SX,SY] = np.meshgrid(np.linspace(x0,x1,N),np.linspace(y0,y1,N))
    #-------------------------------

    #---------------------------------------
    # DU=Grids0.inner_D(Net,x)
    # plot_stremline(DU,SX,SY)
    # del SX,SY,DU
    
    # 圆柱绕流
    # DR = np.linalg.norm(x,axis=-1)-1.5
    # x = x[np.where(DR>0.03)[0]]
    
    #花朵问题
    # x = In_flower(x)
    
    #裂纹
    
    #-------------------------
    
    U_pred=Grids0.inner(Net,xy_test).detach().cpu().numpy().reshape(N, N)
    U_exact = Grids0.func(xy_test).reshape(N, N)
    abs_error = np.abs(U_pred - U_exact)
    
    draw_colormap(x, y, U_pred)
    
    draw_colormap(x, y, U_exact)
    
    title = 'abs error'
    draw_colormap(x, y, abs_error, title, mode=1)

    
def plot_stremline(DU,x,y):
    plt.figure(dpi=400)
    u = np.reshape(DU[0,:],[x.shape[0],-1])+3
    v = np.reshape(DU[1,:],[y.shape[0],-1])
    DR = np.linalg.norm(x,axis=-1)-1.5
    u[np.where(DR<=0.05)[0]]=0
    v[np.where(DR<=0.05)[0]]=0
    plt.streamplot(x,y,u,v,density=[0.5,1])
  


def plot_bound_DU(Grids0,Net):
    
    
    _,DU = Grids0.update_func_any(Net,Grids0.testpoint,Grids0.testpoint_norm)
    DU=DU.reshape([-1])
    DUac = Grids0.func(Grids0.testpoint,norm=Grids0.testpoint_norm,bctype = 1,para=1).reshape([-1])
    ind = np.arange(Grids0.testpoint.shape[0])
    
    dpi0=300
    
    
    plt.figure(dpi=dpi0)
    
    fvalue = DUac
    title = 'f, exact'
    plt.plot(ind,fvalue)
    fvalue = DU
    title = 'f, predict, NSource = '+str(Grids0.Source.shape[0])
    plt.plot(ind,fvalue,ls='--')
    
    font = {'family': 'serif',
    'serif': 'Times New Roman',
    'weight': 'normal',
    'size': 15}
    plt.rc('font',**font)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #matplotlib画图中中文显示会有问题，需要这两行设置默认字体
    plt.legend(["exact","BINN"])
    ax=plt.gca()
    x_locator = plt.MultipleLocator(420)
    ax.xaxis.set_major_locator(x_locator)
    plt.xticks([0,1000,2000,3000,4000])
    plt.ylim(-5,2)
    
    # plt.title(title)
    
    # plt.figure(dpi=dpi0)
    
    # plt.title(title)
    # ax=plt.gca()
    # ax.xaxis.set_major_locator(x_locator)
    
    plt.figure(dpi=dpi0)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #matplotlib画图中中文显示会有问题，需要这两行设置默认字体
    
    fvalue = DUac-DU
    err = np.linalg.norm(fvalue)/np.linalg.norm(DUac)
    title = 'f,error, NSource = '+str(Grids0.Source.shape[0])+',err='+str(err)
    plt.plot(ind,fvalue)
    ax=plt.gca()
    plt.xticks([0,1000,2000,3000,4000])
    plt.ylim(-0.05,0.05)
    # plt.title(title)
    return DU



def plot_bound_U(Grids0,Net):
    U,_ = Grids0.update_func_any(Net,Grids0.testpoint,Grids0.testpoint_norm)
    U=U.reshape([-1])
    Uac = Grids0.func(Grids0.testpoint,norm=Grids0.testpoint_norm,bctype = 0,para=1).reshape([-1])
    ind = np.arange(Grids0.testpoint.shape[0])
    
    dpi0=300
    
    
    plt.figure(dpi=dpi0)
    fvalue = Uac
    title = 'f, exact'
    plt.plot(ind,fvalue)
    fvalue = U
    title = 'f, BINN, NSource = '+str(Grids0.Source.shape[0])
    plt.plot(ind,fvalue,ls='--')
    
    font = {'family': 'serif',
    'serif': 'Times New Roman',
    'weight': 'normal',
    'size': 15}
    plt.rc('font',**font)
    plt.ylim(ymax=8)
    plt.legend(["exact","BINN"],loc='upper right')
    # plt.legend (loc='lower right', fontsize=40)
    ax=plt.gca()
    x_locator = plt.MultipleLocator(420)
    ax.xaxis.set_major_locator(x_locator)
    # plt.title(title)
    
    # plt.figure(dpi=dpi0)
    
    # plt.title(title)
    # ax=plt.gca()
    # ax.xaxis.set_major_locator(x_locator)
    
    plt.figure(dpi=dpi0)
    fvalue = Uac-U
    err = np.linalg.norm(fvalue)/np.linalg.norm(Uac)
    title = 'f,error, NSource = '+str(Grids0.Source.shape[0])+',err='+str(err)
    plt.plot(ind,fvalue)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_locator)
    # plt.title(title)
    return U

def get_DU_anypoint(Grids0,Net,testpoint,testpoint_norm):
    
    
    _,DU = Grids0.update_func_any(Net,testpoint,testpoint_norm)
    DU=DU.reshape([-1])
    
    return DU
def In_flower(x):
    
    c=np.array(
    [[1.11022302462516e-16,	1.37638192047117],\
    [-1.30901699437495,	0.425325404176020],\
    [-0.809016994374948,	-1.11351636441161],\
    [0.809016994374947,	-1.11351636441161],\
    [1.30901699437495,	0.425325404176020]])
        
    # xr = np.linspace(0.9, 1,10)
    # thetar = np.linspace(0, np.pi*2,100)
    # xap,thetap = np.meshgrid(xr,thetar)
    # xap=xap.reshape([-1])
    # thetap=thetap.reshape([-1])
    # XX = np.zeros([xap.shape[0],2])
    # XX[:,0] = c[0,0]+xap*np.cos(thetap)
    # XX[:,1] = c[0,1]+xap*np.sin(thetap)
    # x=np.concatenate([x,XX],axis=0)
    

    LR=np.linalg.norm(x,axis=1)
    Lall = np.where(LR>=0)[0]
    L1 = np.where(LR>=1.37638192047117*0.98)[0]
    xrest = x[L1]
    for i in range(5):
        r = xrest-c[i]
        LR=np.linalg.norm(r,axis=1)
        L2 = np.where(LR>=1*0.97)[0]
        L1 = L1[L2]
        xrest = x[L1]
    Lme=[i for i in Lall if i not in L1]
    y=x[Lme]
    
    return y



if __name__=='__main__': 
    #tname="crack_III"
    # tname = 'multiarc1.2'
    #------------------
    #论文用例
    #tname = 'multiarc1.1'
    #--------------------
    # tname="flow2"
    # tname="flow"
    # tname="flow40"
    #tname = "slender1.0"
    # tname = "testarc"
    tname = 'BINN_KAN'
    workspace = "./outputs/"
    Netinvname0 = workspace+"BIE"+str(tname)+".pt"
    Netinvname00 = workspace+"BIE"+str(tname)+"0.pt"
    # func = exactvalue_crack_III
    # func = exactvalue_sscc
    # func = exactvalue_cylinder
    # func = exactvalue_cylinder
    v0=3
    # func = exactvalue_flux
    # ---------------------------------------------------------
    # func = exactvalue_sscc
    # # func = exactvalue_zero
    # x=y=0
    # b=10
    # h=4
    # area = np.array([x,y,b,h])
    # Bienet = BIENET(h)
    # rect = np.array([x,y,b,h])
    # Grids0 = Geometry_rizzo_HB_again.Generatepoints(x,y,b,h,func,20)
    

    
    # ---------------------------------------------------------
    func = exactvalue_crack_III
    x=0
    y=0.5
    b=2
    h=y*2
    # x=y=0
    # b=2
    # h=2
    area = np.array([x,y,b,h])
    #Bienet =  KAN([2, 5,5,5, 1], base_activation=torch.nn.SiLU, grid_size=30, grid_range=[-b, b], spline_order=3)
    #Bienet = BIENET(h)
    Bienet = KAN([2, 5,5,5, 1], base_activation=torch.nn.SiLU, grid_size=10, grid_range=[-1, 1], spline_order=3)
    rect = np.array([x,y,b,h])
    #device =torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Grids0 = Geometry_rizzo_HB_again.Generatepoints(x,y,b,h,func,20,device)
    
    # x=0
    # y=0
    # b=5.2
    # h=5.2
    #--------------------------------------------------------
    
    
    Bienet = Bienet.to(device)
    Grids0.variable_to_device(device)
    # trainmode=True
    trainmode=True
    if trainmode==True:
        # Xerr0 = Bie_trainer.train_datadriven_Node(Nodes0,Bienet,device,workspace,20000,0.01,Netinvname0)       
        # Xerr = Bie_trainer.train_partitial_Node(Source,Nodes0,Bienet,device,workspace,5000,0.01,Netinvname0)   
        loss_array, error_array = Bie_trainer_rizzo_again.train_partitial_rizzo_again(Grids0,Bienet,device,workspace,2500,0.001) #算每一步的误差
        # error_array = Bie_trainer_rizzo_again.train_partitial_rizzo_again_efficient_mode(Grids0,Bienet,device,workspace,2500,0.001) #性能模式 
        torch.save(Bienet, '../model/BINN_KAN/BINN_KAN')
        
        plot_inner(Grids0,Bienet,300,area=np.array([x,y,b,h]),device = device)
        # 获取交界面的奇异应变
        x = np.linspace(0, 1, 11)[1:-1]  

        y = np.zeros_like(x)

        testpoint = np.column_stack((x, y)) 
        xnorm = np.zeros_like(x)
        ynorm = np.ones_like(x)
        testpoint_norm = np.column_stack((xnorm, ynorm)) 
        
        DU = get_DU_anypoint(Grids0, Bienet, testpoint, testpoint_norm)#输入一组边界点的坐标testpoint，以及点的外反向testpoint_norm，输出DU
        inter_strain = np.vstack([x,DU])
        np.save( '../results/BINN_KAN/interface.npy', inter_strain)
        error = Bie_trainer_rizzo_again.plot_inner(Grids0,Bienet,100,area = np.array([0,0.5,2,1]))
        print('The BINN_KAN error is '+str(error))
        # plot_bound_U(Grids0,Bienet)
        # plot_bound_DU(Grids0,Bienet)

