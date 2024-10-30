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
# import Geometry_rizzo_HB_again
#import Geometry_potential_arc
import Geometry_potential_more_arc
from kan import *
from pyevtk.hl import pointsToVTK

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
    
class BIENET(nn.Module):
    def __init__(self,H):
        super().__init__()
        self.H=H
        # self.n=10
        self.act = nn.Tanh()
        # self.act = nn.ELU()
        # self.act = nn.LeakyReLU()
        # self.act = nn.ReLU()
        
        # self.a1 = nn.Parameter(torch.randn([1]))
        # self.a2 = nn.Parameter(torch.randn([1]))
        # self.a3 = nn.Parameter(torch.randn([1]))
        # self.a4 = nn.Parameter(torch.randn([1]))

        self.FC1=nn.Linear(2,30)
        self.FC2=nn.Linear(30,30)
        self.FC3=nn.Linear(30,30)
        self.FC4=nn.Linear(30,30)
        self.FC5=nn.Linear(30,30)
        self.Net2=nn.Linear(30,1)

    def forward(self,X):
        

        
        # 下面的残差块两个resnet block ，保证包括恒等变换，新的SOTA 0.0014202251
        X = X/self.H
        X=self.FC1(X)
        inden = X
        X=self.FC3(self.act(self.FC2(self.act(X))))
        X=inden+X
        inden = X
        X=self.FC5(self.act(self.FC4(self.act(X))))
        X=inden+X
        inden = X
        out=self.Net2(self.act(X)+inden)
        
        
        # 下面的残差块 只用了一个Resnet block, 也能达到0.2035727
        # X = X/self.H
        # X=self.FC1(X)
        # inden = X
        # X=self.FC3(self.act(self.FC2(self.act(X))))
        # X=inden+X
        # inden = X
        # out=self.Net2(self.act(X)+inden)
        
        return out
    
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

def draw_colormap(x,fvalue,title0,mode=0):

    fig = plt.figure(dpi=400)
    cmap = plt.get_cmap("jet")
    min_c = np.amin(fvalue)
    max_c = np.amax(fvalue)
    # range_c = max_c-min_c
    norm = plt.Normalize(vmin=min_c, vmax=max_c)
    a=plt.scatter(x[:,0], x[:,1], c=fvalue, s=1, cmap=cmap, norm=norm)
    fig.colorbar(a, shrink=1.0, aspect=10)
    # plt.plot([-0.97,0],[0.01,0.01],c='deeppink',linewidth=2)
    plt.axis("equal")
    plt.axis('off')
    if mode==0:
        plt.title(title0)
    # fig2 = plt.figure()
    # cmap = plt.get_cmap("jet")
    # min_c = np.amin(err)
    # max_c = np.amax(err)
    # # range_c = max_c-min_c
    # norm = plt.Normalize(vmin=min_c, vmax=max_c)        
    # b=plt.scatter(x[:,0], x[:,1], c=err, s=2, cmap=cmap, norm=norm)
    # fig2.colorbar(b, shrink=1.0, aspect=10)
    # plt.title('file = '+str(tname)+', meanerr = '+str(meanerr))
    
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

def write_arr2DVTK(filename, coordinates, values, name):
    '''
    这是一个将tensor转化为VTK数据可视化的函数，输入coordinates坐标，是一个二维的二列数据，
    value是相应的值，是一个列向量
    name是转化的名称，比如RBF距离函数，以及特解网路等等
    '''
    # displacement = np.concatenate((values[:, 0:1], values[:, 1:2], values[:, 0:1]), axis=1)
    x = np.array(coordinates[:, 0].flatten(), dtype='float32')
    y = np.array(coordinates[:, 1].flatten(), dtype='float32')
    z = np.zeros(len(coordinates), dtype='float32')
    disX = np.array(values[:, 0].flatten(), dtype='float32')
    pointsToVTK(filename, x, y, z, data={name: disX}) # 二维数据的VTK文件导出

def plot_inner(Grids0,Net,N,area=np.array([0,0,2,2])):
    
    x=np.zeros([N*N,2])
    
    #确定绘图范围
    #三型裂纹
    # x0=area[0]-area[2]/2+0.03
    # x1=area[0]+area[2]/2-0.03
    # y0 = area[1]-area[3]/2+0.03
    # y1 = area[1]+area[3]/2-0.03
    #花朵
    x0=area[0]-area[2]/2*0.95
    x1=area[0]+area[2]/2*0.95
    y0 = area[1]-area[3]/2*0.95
    y1 = area[1]+area[3]/2*0.95
    #圆柱绕流
    # x0=area[0]-area[2]/2
    # x1=area[0]+area[2]/2
    # y0 = area[1]-area[3]/2
    # y1 = area[1]+area[3]/2
    
    # SX=np.linspace(x0,x1,N)
    # SY=np.linspace(y0,y1,N)
    # [SX,SY] = np.meshgrid(np.linspace(x0,x1,N),np.linspace(y0,y1,N))
    #-------------------------------
    x[:,0]=np.repeat(np.linspace(x0,x1,N),N)
    x[:,1]=np.tile(np.linspace(y0,y1,N),N)
    #---------------------------------------
    # DU=Grids0.inner_D(Net,x)
    # plot_stremline(DU,SX,SY)
    # del SX,SY,DU
    
    # 圆柱绕流
    # DR = np.linalg.norm(x,axis=-1)-1.5
    # x = x[np.where(DR>0.03)[0]]
    
    # 花朵问题
    x = In_flower(x)
    
    #裂纹
    
    #-------------------------
    
    U=Grids0.inner(Net,x).cpu().numpy()
    Uac = Grids0.func(x).reshape([-1])
    
    
    fvalue = U
    

    title = 'f, predict, NSource = '+str(Grids0.Source.shape[0])
    draw_colormap(x,fvalue,title)
    
    fvalue = Uac
    title = 'f, exact'
    draw_colormap(x,fvalue,title)
    
    
    error_abs = np.abs(Uac-U)[:, np.newaxis]
    err = np.linalg.norm(error_abs)/np.linalg.norm(Uac)
    title = 'f,error, NSource = '+str(Grids0.Source.shape[0])+',err='+str(err)
    draw_colormap(x,fvalue,title)
    write_arr2DVTK('../output_ntk/error_BINN_KAN', x, error_abs, 'error_binn')
    
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
    # ax=plt.gca()
    # x_locator = plt.MultipleLocator(420)
    # ax.xaxis.set_major_locator(x_locator)
    # plt.xticks([0,1000,2000,3000,4000])
    # plt.ylim(-5,2)
    
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
    # ax=plt.gca()
    # plt.xticks([0,1000,2000,3000,4000])
    # plt.ylim(-0.05,0.05)
    plt.title(title)
    return DU


def get_DU_anypoint(Grids0,Net,testpoint,testpoint_norm):
    
    
    _,DU = Grids0.update_func_any(Net,testpoint,testpoint_norm)
    DU=DU.reshape([-1])
    
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

    tname = 'test_kan'
    workspace = "./results"
    Netinvname0 = workspace+"BIE"+str(tname)+".pt"
    Netinvname00 = workspace+"BIE"+str(tname)+"0.pt"

    #---------------------------------------------------
    # 多圆弧问题
    # 五瓣花
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    func = exactvalue_sscc
    x=0
    y=0
    b=5.2
    h=5.2
    r = [1,1,1,1,1]
    #train:
    NE=[20,20,20,20,20]
    #test:
    c=np.array(
    [[1.11022302462516e-16,	1.37638192047117],\
    [-1.30901699437495,	0.425325404176020],\
    [-0.809016994374948,	-1.11351636441161],\
    [0.809016994374947,	-1.11351636441161],\
    [1.30901699437495,	0.425325404176020]])
    
    BCtype = np.repeat([0], sum(NE))
    theta1=np.array([0.,2*np.pi/5,4*np.pi/5,6*np.pi/5,8*np.pi/5])
    theta2=np.array([np.pi,2*np.pi/5+np.pi,4*np.pi/5+np.pi,6*np.pi/5+np.pi,8*np.pi/5+np.pi])
    Grids0 = Geometry_potential_more_arc.arcs_HB(r,c,NE,func,BCtype,theta1,theta2,device = device)
    Bienet =  KAN([2, 5,5,5, 1], base_activation=torch.nn.SiLU, grid_size=10, grid_range=[-b, b], spline_order=3)
    # Bienet = MultiLayerNet(2, 30, 1)
    
    Bienet = Bienet.to(device)
    Grids0.variable_to_device(device)
    
    # ---------------------------------------------------------

    trainmode=True
    # trainmode=False
    if trainmode==True:
        # Xerr0 = Bie_trainer.train_datadriven_Node(Nodes0,Bienet,device,workspace,20000,0.01,Netinvname0)       
        # Xerr = Bie_trainer.train_partitial_Node(Source,Nodes0,Bienet,device,workspace,5000,0.01,Netinvname0)       
        # Xerr,inner_err = Bie_trainer_rizzo_again.train_partitial_rizzo_again(Grids0,Bienet,device,workspace,20000,0.001,Netinvname0) 
        
        Xerr = Bie_trainer_rizzo_again.train_partitial_rizzo_again_efficient_mode(Grids0,Bienet,device,workspace,10000,0.001,Netinvname0) #性能模式
        #loss_array, error_array = Bie_trainer_rizzo_again.train_partitial_rizzo_again(Grids0,Bienet,device,workspace,20000,0.001) #算每一步的误差        
        #np.save( '../results/flower_BINN_KAN_error.npy', error_array)        
        
        #np.save(workspace+str(tname)+"loss.npy",Xerr)
        plot_inner(Grids0,Bienet,300,area=np.array([x,y,b,h]))#这里的100是生成100*100个点（还要删去花朵外的点）
        
        plot_bound_DU(Grids0,Bienet)#输出边界未知量（法向导数值）

    elif trainmode==False:
        Bienet.load_state_dict(torch.load(Netinvname0))
        plot_inner(Grids0,Bienet,100,area=np.array([x,y,b,h]))
        # plot_bound_U(Grids0,Bienet)
        # plot_bound_DU(Grids0,Bienet)
        