# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:18:12 2022

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
import Bie_trainer_elastic
import Geometry_elastic_HB
import Geometry_elastic_HB_reg
import Geometry_elastic_arc
import Geometry_elastic_more_geometry
from kan import*
import xlrd
from pyevtk.hl import pointsToVTK
def setup_seed(seed):
# random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2024)
def write_vtk_v2p(filename, dom, U_mag): #已经将输出的感兴趣场进行了分类VTK导出
    xx = np.ascontiguousarray(dom[:, 0])
    yy = np.ascontiguousarray(dom[:, 1])
    zz = np.zeros(len(xx)) # 点的VTK
    pointsToVTK(filename, xx, yy, zz, data={ "U-mag": U_mag})


class BIENET(nn.Module):
    def __init__(self,H):
        super().__init__()
        self.H=torch.tensor(H).float().to(device)
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
        self.Net2=nn.Linear(30,2)

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
    
def const_function(C):    
        def func(x,norm=0,bctype = 0,para=0, E=1,v=0.3, flaten = 1):
            y=np.empty(x.shape)
            y[:,0] = C[0]
            y[:,1] = C[1]
            if  flaten==1:
                y = np.append(y[:,0],y[:,1])
            return y
        return func
    
    
# def linear_function(x0,x1,y0,y1):
    
#     def func(x,norm=0,bctype = 0,para=0, E=1,v=0.3, flaten = 1):
#         y=np.empty(x.shape)
        
        
        
#         if bctype == 0:
#             y=np.empty(x.shape)
#             y[:,0] = e1*x[:,0]
#             y[:,1] = e2*x[:,1]
#         elif bctype ==1:
#             S = np.array([[s1,0],[0,s2]])
#             if para==0:
#                 y = np.matmul(S,norm)
#             elif para==1:
#                 y = np.matmul(norm,S)
                
#         if  flaten==1:
#             y = np.append(y[:,0],y[:,1])
#         return y
    

def elastic_01(x,norm=0,bctype = 0,para=0, E=1,v=0.3, flaten=1):
    y=np.zeros(x.shape)
    
    if x[0,0]==20:
        y[:,0] = 100
        
    if  flaten==1:
        y = np.append(y[:,0],y[:,1])
    return y
def elastic_linear(x,norm=0,bctype = 0,para=0, E=1,v=0.3, flaten = 1):
    s1 = 2
    s2 = 3
    # v1 = v/(1-v)
    # E1 = E/(1-v*v)
    v1,E1 = 0.3,1000
    e1 = s1/E1-v1*s2/E1
    e2 = s2/E1-v1*s1/E1
    
    if bctype == 0:
        y=np.empty(x.shape)
        y[:,0] = e1*x[:,0]
        y[:,1] = e2*x[:,1]
    elif bctype ==1:
        S = np.array([[s1,0],[0,s2]])
        if para==0:
            y = np.matmul(S,norm)
        elif para==1:
            y = np.matmul(norm,S)
    if  flaten==1:
        y = np.append(y[:,0],y[:,1])
    return y

def elastic_zero(x,norm=0,bctype = 0,para=0, E=1,v=0.3, flaten = 1):
    # s1 = 2
    # s2 = 3
    # v1 = v/(1-v)
    # E1 = E/(1-v*v)
    # e1 = s1/E1-v1*s2/E1
    # e2 = s2/E1-v1*s1/E1
    y=np.zeros(x.shape)
    # if bctype == 0:
    #     y=np.empty(x.shape)
    #     y[:,0] = 0
    #     y[:,1] = 0
    # elif bctype ==1:
    #     y=np.empty(x.shape)
    #     y[:,0] = 0
    #     y[:,1] = 0
    if  flaten==1:
        y = np.append(y[:,0],y[:,1])
    return y


def elastic_constant(x,norm=0,bctype = 0,para=0, E=1,v=0.3, flaten = 1,constant0 =[ 1,1]):
   
    y=np.zeros(x.shape)
    if bctype == 0:
        
        y[:,0] = constant0[0]
        y[:,1] = constant0[1]
    else:
        y=np.zeros(x.shape)
    if  flaten==1:
        y = np.append(y[:,0],y[:,1])
    return y

def elastic_force_constant(x,norm=0,bctype = 0,para=0, E=1,v=0.3, flaten = 1,constant0 =[ 1,1]):
    y=np.zeros(x.shape)
    if bctype == 0:
        y=np.zeros(x.shape)
    else:
        y[:,0] = constant0[0]
        y[:,1] = constant0[1]
    if  flaten==1:
        y = np.append(y[:,0],y[:,1])
    return y

def elastic_beam(x,norm=0,bctype = 0,para=0, E=1,v=0.3, flaten = 1):
    #这是有解析解的特殊问题，本问题不需要满足长高比的假设
    P = 1
    v1 = v/(1-v)
    E1 = E/(1-v*v)
    I = h*h*h/12#这里用了全局变量，即梁高h
    y=np.empty(x.shape)
    if bctype == 0:
        
        y[:,0] = -P*x[:,1]/6/E1/I*((6*b-3*x[:,0])*x[:,0]+(2+v1)*(x[:,1]*x[:,1]-h*h/4))
        y[:,1] = P/6/E1/I*(3*v1*x[:,1]*x[:,1]*(b-x[:,0])+(4+5*v1)*h*h*x[:,0]/4+(3*b-x[:,0])*x[:,0]*x[:,0])
    elif bctype ==1:
        S = np.empty([x.shape[0],4])
        S[:,0] = -P*(b-x[:,0])*x[:,1]/I
        S[:,1] = P/2/I*(h*h/4-x[:,1]*x[:,1])
        S[:,2] = S[:,1]
        S[:,3] = 0
        if para==0:
            y[:,0] = S[:,0]*norm[0]+S[:,1]*norm[1]
            y[:,1] = S[:,2]*norm[0]+S[:,3]*norm[1]
        elif para==1:
            y[:,0] = S[:,0]*norm[:,0]+S[:,1]*norm[:,1]
            y[:,1] = S[:,2]*norm[:,0]+S[:,3]*norm[:,1]
            
    if  flaten==1:
        y = np.append(y[:,0],y[:,1])
    return y

def draw_colormap(x,fvalue, filename, title='',mode=0):

    fig = plt.figure(dpi=400)
    cmap = plt.get_cmap("jet")
    min_c = np.amin(fvalue)
    max_c = np.amax(fvalue)
    # range_c = max_c-min_c
    norm = plt.Normalize(vmin=min_c, vmax=max_c)
    a=plt.scatter(x[:,0], x[:,1], c=fvalue, s=3, cmap=cmap, norm=norm)
    fig.colorbar(a, shrink=1.0, aspect=10).ax.set_title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis("equal")
    if mode == 1: # save fig
        plt.savefig(filename, dpi = 300)
    plt.show()
    # plt.axis('off')
def plot_inner2(Grids0,Net,N,mode=0):
    
    if mode==0:
        x=np.zeros([N*N,2])
        #梁
        # x[:,0]=np.repeat(np.linspace(0.03,1.97,N),N)
        # x[:,1]=np.tile(np.linspace(-0.97,0.97,N),N)
        #孔板
        x[:,0]=np.repeat(np.linspace(-2.47,2.47,N),N)
        x[:,1]=np.tile(np.linspace(-2.47,2.47,N),N)
    elif mode==1:
        x=np.zeros([N,2])
        x[:,0]=np.linspace(0.0,2.0,N)
        x[:,1]=0
        
    #----------------------------
    #四分之一孔板，孔半径5，板半宽20
    elif mode==2:
        x=np.zeros([N*N,2])
        #梁
        x[:,0]=np.repeat(np.linspace(0.3,19.7,N),N)
        x[:,1]=np.tile(np.linspace(0.3,19.7,N),N)
        LR=np.linalg.norm(x,axis=1)
        L1 = np.where(LR>=5.3)[0]
        x = x[L1]
    # LR=np.linalg.norm(x,axis=1)
    # L1 = np.where(LR<=1*0.98)[0]
    
    # L2 = np.where(LR>= 1*1.02)[0]
    # L1 = np.concatenate([L1,L2])
    # x = x[L1]
    #----------------------------
    
    
    U=Grids0.inner(Net,x).cpu().numpy()
    
    N_inner = x.shape[0]
    fvalue = U[0:N_inner]
    title = 'U1, predict, NSource = '+str(Grids0.Source.shape[0])
    draw_colormap(x,fvalue,title)
    
    fvalue = U[N_inner:2*N_inner]
    title = 'U2, predict, NSource = '+str(Grids0.Source.shape[0])
    draw_colormap(x,fvalue,title)
def plot_inner(Grids0,Net,N,mode=0):
    
    if mode==0:
        x=np.zeros([N*N,2])
        #梁
        # x[:,0]=np.repeat(np.linspace(0.03,1.97,N),N)
        # x[:,1]=np.tile(np.linspace(-0.97,0.97,N),N)
        #孔板
        x[:,0]=np.repeat(np.linspace(-2.47,2.47,N),N)
        x[:,1]=np.tile(np.linspace(-2.47,2.47,N),N)
    elif mode==1:
        x=np.zeros([N,2])
        x[:,0]=np.linspace(0.0,2.0,N)
        x[:,1]=0
        
    #----------------------------
    #四分之一孔板，孔半径5，板半宽20
    elif mode==2:
        x=np.zeros([N*N,2])
        #梁
        x[:,0]=np.repeat(np.linspace(0.3,19.7,N),N)
        x[:,1]=np.tile(np.linspace(0.3,19.7,N),N)
        LR=np.linalg.norm(x,axis=1)
        L1 = np.where(LR>=5.3)[0]
        x = x[L1]
    # LR=np.linalg.norm(x,axis=1)
    # L1 = np.where(LR<=1*0.98)[0]
    
    # L2 = np.where(LR>= 1*1.02)[0]
    # L1 = np.concatenate([L1,L2])
    # x = x[L1]
    #----------------------------
    
    
    N_inner = x.shape[0]
    
    U=Grids0.inner(Net,x).cpu().numpy()
    sigma=Grids0.inner_stress(Net,x).cpu().numpy()
    
    sigmaxx = sigma[0:N_inner]
    sigmaxy = sigma[N_inner:2*N_inner]
    sigmaxy_sym = sigma[2*N_inner:3*N_inner]
    sigmayy = sigma[3*N_inner:4*N_inner]
    
    ux = U[0:N_inner]
    uy = U[N_inner:2*N_inner]
    u = (ux**2+uy**2)**(0.5)
    
    SVonMises =(0.5 * ((sigmaxx - sigmayy) ** 2 + (sigmayy) ** 2 + (-sigmaxx) ** 2 + 6 * (sigmaxy ** 2)))**(0.5)
    
    # draw_colormap(x, u, filename = '../results/BINN_MLP/abs_error_dis_BINN_MLP.pdf',  mode=1)
    # draw_colormap(x, SVonMises, filename = '../results/BINN_MLP/abs_error_mise_BINN_MLP.pdf', mode=1)
    draw_colormap(x, u, filename = '../abs_error_dis_KINN_BINN.pdf',  mode=1)
    draw_colormap(x, SVonMises, filename = '../abs_error_mise_KINN_BINN.pdf', mode=1)
    write_vtk_v2p('../results/KINN_BINN/Plate_hole_KINN_BINNs', x, u)
    zeros_column = np.zeros((x.shape[0], 1))
    x_with_zeros = np.hstack((x, zeros_column))
    np.save('../abaqus_reference/onehole/coordinate_binn.npy', x_with_zeros)
    return U

def calculate_error(Grids0,Net):
    N = 300
    x=np.zeros([N*N,2])
    #梁
    x[:,0]=np.repeat(np.linspace(0.3,19.7,N),N)
    x[:,1]=np.tile(np.linspace(0.3,19.7,N),N)
    LR=np.linalg.norm(x,axis=1)
    L1 = np.where(LR>=5.3)[0]
    x = x[L1]
    U_mag_exact = np.load('../abaqus_reference/onehole/U_mag_binn.npy')[:,1]    
    Von_exact = np.load('../abaqus_reference/onehole/Von_mises_binn.npy')[:,1]   
    
    N_inner = x.shape[0]
    
    U=Grids0.inner(Net,x).cpu().numpy()

    ux = U[0:N_inner]
    uy = U[N_inner:2*N_inner]
    U_mag = (ux**2+uy**2)**(0.5)
    L2_u = np.linalg.norm(U_mag - U_mag_exact) / np.linalg.norm(U_mag_exact)    
    
    sigma=Grids0.inner_stress(Net,x).cpu().numpy()
    sigmaxx = sigma[0:N_inner]
    sigmaxy = sigma[N_inner:2*N_inner]
    sigmaxy_sym = sigma[2*N_inner:3*N_inner]
    sigmayy = sigma[3*N_inner:4*N_inner]
    SVonMises =(0.5 * ((sigmaxx - sigmayy) ** 2 + (sigmayy) ** 2 + (-sigmaxx) ** 2 + 6 * (sigmaxy ** 2)))**(0.5)
    
    L2_von = np.linalg.norm(SVonMises - Von_exact) / np.linalg.norm(Von_exact)  
    return L2_u, L2_von

if __name__=='__main__': 
    tname='BEAM_kongban'
    # tname = 'linear'
    
    # tname='BEAM_ori1'
    #论文用例：
    # tname='BEAM_art1'
    workspace = "./output/"
    Netinvname0 = workspace+"BIE_kan"+str(tname)+".pt"
    Netinvname00 = workspace+"BIE_kan"+str(tname)+"0.pt"
    #-----------------------------------
    # 梁模型
    # func = elastic_beam
    # func = elastic_linear
    # func = elastic_zero
    # b=4
    # h=4
    # x=b/2
    # y=0
    # #Bienet = BIENET(np.array([b,h]))
    
    # rect = np.array([x,y,b,h])
    # BCtype = [0,0,0,0,0,0,0,0]
    # NL = 0.01
    # NE = [int(num) for num in [b/NL,h/NL,b/NL,h/NL]]
    # # BCtype = [0,0,0,0,0,0,0,0]
    # Grids0 = Geometry_elastic_HB_reg.Generatepoints(x,y,b,h,func,BCtype,NE,E=1,v=0.3)
    #-----------------------------------------------
    #圆弧测试
    # func = elastic_linear
    # x=5
    # y=0
    # b=10
    # h=2
    # r=5
    # Bienet = BIENET(np.array([r,r]))
    # rect = np.array([x,y,b,h])
    # # BCtype = [1,1,1,0,1,1,1,0]
    # BCtype = np.array([0,0])
    # Grids0 = Geometry_elastic_arc.arc_HB(r,[0,0],func,BCtype,20,E=1,v=0.3)
    
    #------------------------------------------------
    #多几何测试
    # func = elastic_beam
    # x=1
    # y=0
    # b=2
    # h=2
    # Bienet = BIENET(np.array([b,h]))
    
    # # BCtype = [1,1,1,0,1,1,1,0]
    # # BCtype = [0,0,0,0,0,0,0,0]
    # # BCtype = np.array([[0,0],
    # #                     [0,0],
    # #                     [0,0],
    # #                     [0,0]])
    # BCtype = np.array([[1,1],
    #                     [1,1],
    #                     [1,1],
    #                     [0,0]])
    # Grids0 = Geometry_elastic_more_geometry.rect(x,y,b,h,func,BCtype,20,E=1,v=0.3)
    #------------------------------------------------
    #多几何测试
    # func = elastic_linear
    # x=5
    # y=0
    # b=10
    # h=2
    # r=5
    # Bienet = BIENET(np.array([r,r]))
    
    # # BCtype = [1,1,1,0,1,1,1,0]
    # # BCtype = [0,0,0,0,0,0,0,0]
    # # BCtype = np.array([[0,0],
    # #                     [0,0],
    # #                     [0,0],
    # #                     [0,0]])
    # BCtype = np.array([[0,0]])
    # Grids0 = Geometry_elastic_more_geometry.circle(x,y,r,func,BCtype,20,E=1,v=0.3)
    #------------------------------------------------
    #多几何测试 孔板算例
    # func = elastic_linear
    # x=0
    # y=0
    # b=5
    # h=5
    # r=1
    # c=[0,0]
    # Bienet = BIENET(np.array([b,h]))
    # NE = 40
    # # BCtype = [1,1,1,0,1,1,1,0]
    # # BCtype = [0,0,0,0,0,0,0,0]
    # # BCtype = np.array([[0,0],
    # #                     [0,0],
    # #                     [0,0],
    # #                     [0,0]])
    # BCtype = np.array([[1,1],
    #                     [1,1],
    #                     [1,1],
    #                     [0,0],
    #                     [1,1]])
   
           
    # # func=[const_function([0,0]),
    # #       const_function([1,0]),
    # #       const_function([0,0]),
    # #       const_function([0,0]),
    # #       const_function([0,0]),]
    # func=[elastic_linear,
    #       elastic_linear,
    #       elastic_linear,
    #       elastic_linear,
    #       elastic_linear]
 
    # Grids0 = Geometry_elastic_more_geometry.orifice(x,y,h,b,c,r,func,BCtype,NE,E=1,v=0.3)
    
#------------------------------------------------------------ 

    # line1 = {'mode':'line','NE':50,'x0':[5,0],'x1':[20,0],'type':[1,0],'func':elastic_linear}
    # line2 = {'mode':'line','NE':50,'x0':[20,0],'x1':[20,20],'type':[1,1],'func':elastic_linear}
    # line3 = {'mode':'line','NE':50,'x0':[20,20],'x1':[0,20],'type':[1,1],'func':elastic_linear}
    # line4 = {'mode':'line','NE':50,'x0':[0,20],'x1':[0,5],'type':[0,1],'func':elastic_linear}
    # arc1 = {'mode':'arc','NE':20,'c':[0,0],'theta1':np.pi/2,'theta2':0,'type':[1,1],'r':5,'func':elastic_linear}
    # geoset = [line1,line2,line3,line4,arc1]
    # Grids0 = Geometry_elastic_more_geometry.Any_HB(geoset,elastic_linear,E=1,v=0.3)
    # # Bienet = KAN([2, 5, 5, 5, 2], base_activation=torch.nn.SiLU, grid_size=10, grid_range=[0.0, 20.0],spline_order=3)
    # Bienet = BIENET(np.array([20,20]))
 #------------------------------------------------------------     
#------------------------------------------------------------ 
    func = elastic_01
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    line1 = {'mode':'line','NE':30,'x0':[5,0],'x1':[20,0],'type':[1,0],'func':func}
    line2 = {'mode':'line','NE':40,'x0':[20,0],'x1':[20,20],'type':[1,1],'func':func}
    line3 = {'mode':'line','NE':40,'x0':[20,20],'x1':[0,20],'type':[1,1],'func':func}
    line4 = {'mode':'line','NE':30,'x0':[0,20],'x1':[0,5],'type':[0,1],'func':func}
    arc1 = {'mode':'arc','NE':20,'c':[0,0],'theta1':np.pi/2,'theta2':0,'type':[1,1],'r':5,'func':func}
    geoset = [line1,line2,line3,line4,arc1]
    
    E=1000
    v=0.3
    
    # if pl_type == 'palnestress':
        
    
    Grids0 = Geometry_elastic_more_geometry.Any_HB(geoset,func,pl_type = 'planestress',E=E,v=v,device = device)
    # Bienet = KAN([2, 5, 5, 5, 2], base_activation=torch.nn.SiLU, grid_size=10, grid_range=[0.0, 20.0],spline_order=3)
    # Bienet = MultiLayerNet(2,30,2)
    
    Bienet = KAN([2,  5, 5,5, 2], base_activation=torch.nn.SiLU, grid_size=10, grid_range=[0.0, 20.0],spline_order=3).cuda()
    
 #------------------------------------------------------------     
    
    # device =torch.device("cpu")
    
    Bienet = Bienet.to(device)
    Grids0.variable_to_device(device)
    trainmode=True
    # Bienet.load_state_dict(torch.load(Netinvname0))
    # torch.save(Bienet.state_dict(), Netinvname00)
    if trainmode==True:
        # Xerr0 = Bie_trainer.train_datadriven_Node(Nodes0,Bienet,device,workspace,20000,0.01,Netinvname0)       
        # Xerr = Bie_trainer.train_partitial_Node(Source,Nodes0,Bienet,device,workspace,5000,0.01,Netinvname0)       

        Xerr = Bie_trainer_elastic.train_partitial_rizzo_again(Grids0,Bienet,device,workspace,20000,0.001,Netinvname0)
        L2_u, L2_von = calculate_error(Grids0,Bienet)
        # Xerr = Bie_trainer_rizzo.train_full_rizzo_b(Grids0,Bienet,device,workspace,500,0.01,Netinvname0) 
        plot_inner(Grids0,Bienet,300,mode=2)
        
        
        datax0 = xlrd.open_workbook("hole_abaqus_x=0.xls") # 获得x=0的数据，需要提取y方向位移以及mise应力
        tablex0 = datax0.sheet_by_index(0)
        x0 = tablex0.col_values(0)
        x0 = np.array(x0)
        x0_2d = np.zeros((len(x0), 2))
        x0_2d[:, 1] = x0    
        disy_abqusx0 = np.array(tablex0.col_values(1))
        mise_abaqusx0 = tablex0.col_values(2)    
        
        datay0 = xlrd.open_workbook("hole_abaqus_y=0.xls") # 获得x=0的数据，需要提取y方向位移以及mise应力
        tabley0 = datay0.sheet_by_index(0)
        y0 = tabley0.col_values(0)
        y0 = np.array(y0)
        y0_2d = np.zeros((len(y0), 2))
        y0_2d[:, 0] = y0
        disx_abqusy0 = np.array(tabley0.col_values(1))        
        

        x_axis = torch.tensor(y0_2d, dtype = torch.float32,requires_grad=True).to(device)
        y_axis = torch.tensor(x0_2d, dtype = torch.float32,requires_grad=True).to(device)
        pred_disy_r,pred_mise_r = Grids0.plot_x0(Bienet,y_axis)
        pred_disx_r = Grids0.plot_y0(Bienet,x_axis)
        
        
        # 画曲线评估，需要画3幅图，第一幅是x0的y方向位移
        plt.figure()
        plt.plot(x0, disy_abqusx0, color = 'red', label = 'FEM', lw = 2)
        plt.plot(x0, pred_disy_r, color = 'blue', label = 'BINN', lw = 2)
        plt.xlabel('Y') 
        plt.ylabel('Dis_Y')
        plt.legend()

        # 第二幅是x0的mise应力
        plt.figure()
        plt.plot(x0, mise_abaqusx0, color = 'red', label = 'FEM', lw = 2)
        plt.plot(x0, pred_mise_r, color = 'blue', label = 'BINN', lw = 2)
        plt.xlabel('Y') 
        plt.ylabel('Mises')
        plt.legend()

        # 第三幅是y0的x方向位移
        plt.figure()
        plt.plot(y0, disx_abqusy0, color = 'red', label = 'FEM', lw = 2)
        plt.plot(y0, pred_disx_r, color = 'blue', label = 'BINN', lw = 2)
        plt.xlabel('X') 
        plt.ylabel('Dis_X')
        plt.legend()
        plt.show()

        filename_out =  '../results/KINN_BINN'
        
        # 把x=0和y=0的一些量储存起来
        x0disy_fem = np.vstack([x0, disy_abqusx0])
        x0disy_pred = np.vstack([x0, pred_disy_r])
        
        x0mise_fem = np.vstack([x0, mise_abaqusx0])
        x0mise_pred = np.vstack([x0, pred_mise_r])
        
        y0disx_fem = np.vstack([y0, disx_abqusy0])
        y0disx_pred = np.vstack([y0, pred_disx_r])
        
        filename_out =  '../results/KINN_BINN'
        
        np.save(filename_out+'/exact_x0disy.npy', x0disy_fem)
        np.save(filename_out+'/KINN_BINN_x0disy.npy', x0disy_pred)
        
        np.save(filename_out+'/exact_x0mise.npy', x0mise_fem)
        np.save(filename_out+'/KINN_BINN_x0mise.npy', x0mise_pred)
        
        np.save(filename_out+'/exact_y0disx.npy', y0disx_fem)
        np.save(filename_out+'/KINN_BINN_y0disx.npy', y0disx_pred)
           


    else:
        Bienet.load_state_dict(torch.load(Netinvname0))
        Grids0.update_func(Bienet)
        
        # Bie_trainer_elastic.draw(Grids0,-1)
        #--------------------------------------------
        # rect0 = [x,y,b,h]
        
        # Geometry_elastic_HB_reg.draw_BEAM(Grids0,Bienet,rect0,NE=1000)
        #-----------------------------------------------------------------
        
        # Geometry_elastic_more_geometry.draw_inclusion(Grids0,Bienet,rect0,R0=[c[0],c[1],r],NE=1000)
        # line1 = {'mode':'line','NE':15,'x0':[5,0],'x1':[20,0],'type':[1,0],'func':elastic_01}
        # line2 = {'mode':'line','NE':20,'x0':[20,0],'x1':[20,20],'type':[1,1],'func':elastic_01}
        # line3 = {'mode':'line','NE':20,'x0':[20,20],'x1':[0,20],'type':[1,1],'func':elastic_01}
        # line4 = {'mode':'line','NE':15,'x0':[0,20],'x1':[0,5],'type':[0,1],'func':elastic_01}
        # arc1 = {'mode':'arc','NE':10,'c':[0,0],'theta1':0,'theta2':np.pi/2,'type':[1,1],'r':5,'func':elastic_01}
        # geoset = [line1,line2,line3,line4,arc1]
        # Grids0 = Geometry_elastic_more_geometry.Any_HB(geoset,elastic_01,E=1,v=0.3)
        # plot_inner(Grids0,Bienet,300,mode=2)
        
        plot_inner(Grids0,Bienet,100,mode=2)
        datax0 = xlrd.open_workbook("hole_abaqus_x=0.xls")  # 获得x=0的数据，需要提取y方向位移以及mise应力
        tablex0 = datax0.sheet_by_index(0)
        x0 = tablex0.col_values(0)
        x0 = np.array(x0)
        x0_2d = np.zeros((len(x0), 2))
        x0_2d[:, 1] = x0
        disy_abqusx0 = np.array(tablex0.col_values(1))
        mise_abaqusx0 = tablex0.col_values(2)

        datay0 = xlrd.open_workbook("hole_abaqus_y=0.xls")  # 获得x=0的数据，需要提取y方向位移以及mise应力
        tabley0 = datay0.sheet_by_index(0)
        y0 = tabley0.col_values(0)
        y0 = np.array(y0)
        y0_2d = np.zeros((len(y0), 2))
        y0_2d[:, 0] = y0
        disx_abqusy0 = np.array(tabley0.col_values(1))

        x_axis = torch.tensor(y0_2d, dtype=torch.float32, requires_grad=True).to(device)
        y_axis = torch.tensor(x0_2d, dtype=torch.float32, requires_grad=True).to(device)
        pred_disy_r, pred_mise_r = Grids0.plot_x0(Bienet, y_axis)
        pred_disx_r = Grids0.plot_y0(Bienet, x_axis)

        # 画曲线评估，需要画3幅图，第一幅是x0的y方向位移
        plt.figure()
        plt.plot(x0, disy_abqusx0, color='red', label='FEM', lw=2)
        plt.plot(x0, pred_disy_r, color='blue', label='BINN', lw=2)
        plt.xlabel('Y')
        plt.ylabel('Dis_Y')
        plt.legend()
        # plt.show()
        # 第二幅是x0的mise应力
        plt.figure()
        plt.plot(x0, mise_abaqusx0, color='red', label='FEM', lw=2)
        plt.plot(x0, pred_mise_r, color='blue', label='BINN', lw=2)
        plt.xlabel('Y')
        plt.ylabel('Mises')
        plt.legend()
        # plt.show()
        # 第三幅是y0的x方向位移
        plt.figure()
        plt.plot(y0, disx_abqusy0, color='red', label='FEM', lw=2)
        plt.plot(y0, pred_disx_r, color='blue', label='BINN', lw=2)
        plt.xlabel('X')
        plt.ylabel('Dis_X')
        plt.legend()
        plt.show()