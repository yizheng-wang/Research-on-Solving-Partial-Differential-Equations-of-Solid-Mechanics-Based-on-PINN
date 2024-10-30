# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:54:47 2022

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
MSE = torch.nn.MSELoss(reduction='mean')
Ngauss_log = 10
Gausspoint_log = np.array([0.0090425944,0.053971054,0.13531134,0.24705169,0.38021171,\
                            0.52379159,0.66577472,0.79419019,0.89816102,0.96884798])
Gaussweight_log = -np.array([0.12095474,0.18636310,0.19566066,0.17357723,0.13569597,\
                            0.093647084,0.055787938,0.027159893,0.0095151992,0.0016381586])    

Gau =np.array([[0.0666713443086881, -0.9739065285171717],
               [0.1494513491505806, -0.8650633666889845],
               [0.2190863625159820, -0.6794095682990244],
               [0.2692667193099963, -0.4333953941292472],
               [0.2955242247147529, -0.1488743389816312],
               [0.2955242247147529, 0.1488743389816312],
               [0.2692667193099963, 0.4333953941292472],
               [0.2190863625159820, 0.6794095682990244],
               [0.1494513491505806, 0.8650633666889845],
               [0.0666713443086881, 0.9739065285171717]])
Gausspoint = Gau[:,1]
Gaussweight = Gau[:,0]
Ngauss= 10
class grids_HB():
    def __init__(self,p1,p2,func,type0,E=1,v=0.3):
        '''
        Parameters
        ----------
        p1 : array
            DESCRIPTION.
        p2 : array
            DESCRIPTION.
        func : function
            DESCRIPTION.
        type0 : element type
            长度是单元数的两倍，表示该单元两个方向的位移是给定（0）或者没给定（1）
            前N个表示第一个方向，后N个表示第二个方向
        Returns
        -------
        None.

        '''
        self.E = E
        self.v = v
        self.shear = E/2/(1+v)
        self.A1=-1/8/np.pi/(1-v)/(E/2/(1+v))
        self.A2=3-4*v
        self.func = func
        self.NE=p1.shape[0]
        self.p1 = p1
        self.p2 = p2
        self.GP = np.array((p1+p2)/2)
        # self.GP = np.random.rand(p1.shape[0]).reshape(-1,1)*(p2-p1)+p1
        self.type = type0
        self.ulist = np.array(np.where(type0==0))[0]#给定位移
        self.tlist = np.array(np.where(type0==1))[0]#给定力

        
        rE = p2-p1
        LE = np.linalg.norm(rE,axis=1)
        self.Source = (p2+p1)/2
        self.SourceT = torch.tensor(self.Source,dtype=torch.float32,requires_grad=True)
        self.LE =LE
        #weights include jacobi
        jacobi = self.LE/2
        jacobi_line = np.repeat(jacobi,Ngauss)
        self.weightT =torch.tensor(np.tile(Gaussweight,self.NE)*jacobi_line).float()
        
        self.GP = np.empty([self.NE*Ngauss,2])
        for i in range(self.NE):
            self.GP[i*Ngauss:(i+1)*Ngauss,:] = self.Source[i]+ np.outer(Gausspoint,rE[i])/2 
        self.GPT = torch.tensor(self.GP,dtype = torch.float32,requires_grad=True)   
        
        self.GPlog = np.empty([2*self.NE*Ngauss_log,2])
          
        for i in range(self.NE):
            self.GPlog[2*i*Ngauss_log:2*(i+1)*Ngauss_log,:] = np.append(self.Source[i]- np.outer(Gausspoint_log,rE[i]),\
                                                                self.Source[i]+ np.outer(Gausspoint_log,rE[i]),axis=0)
        self.GPTlog = torch.tensor(self.GPlog,dtype = torch.float32,requires_grad=True)
        
        self.norm = np.empty(rE.shape)
        self.norm[:,0] = rE[:,1]/LE
        self.norm[:,1] = -rE[:,0]/LE
        self.normT = torch.tensor(self.norm,dtype = torch.float32)  
        self.fac = torch.tensor(self.func(self.GP,self.norm.repeat(Ngauss,axis=0),0,para=1,E=self.E,v=self.v),dtype = torch.float32)
        self.fac_source = torch.tensor(self.func(self.Source,self.norm,0,para=1,E=self.E,v=self.v),dtype = torch.float32)
        self.dfac = torch.tensor(self.func(self.GP,self.norm.repeat(Ngauss,axis=0),1,para=1,E=self.E,v=self.v),dtype = torch.float32)
        self.dfac_log = torch.tensor(self.func(self.GPlog,self.norm.repeat(2*Ngauss_log,axis=0),1,para=1,E=self.E,v=self.v),dtype = torch.float32)
        #这两个数组用于存储哪些积分点矩阵元素值对应的变量是已知的
        # self.ucol_index = np.tile(np.array(range(Ngauss)),self.ulist.shape[0])
        # self.ucol_index+=self.ulist.repeat(Ngauss,axis=0)*Ngauss
        # self.tcol_index = np.tile(np.array(range(Ngauss)),self.tlist.shape[0])
        # self.tcol_index+=self.tlist.repeat(Ngauss,axis=0)*Ngauss
        #如果变量顺序是先所有节点的1方向，然后是所有节点的2方向，这里不用做变化
        self.ucol_index = np.tile(np.array(range(Ngauss)),self.ulist.shape[0])
        self.ucol_index+=self.ulist.repeat(Ngauss,axis=0)*Ngauss
        self.tcol_index = np.tile(np.array(range(Ngauss)),self.tlist.shape[0])
        self.tcol_index+=self.tlist.repeat(Ngauss,axis=0)*Ngauss
        
        
        self.solution = self.fac.numpy().copy()
        self.solution[self.ucol_index] = self.dfac.numpy()[self.ucol_index].copy()
        self.assemble_matrix()
        
    def assemble_matrix(self):
        Nrow = self.Source.shape[0]*2
        Ncol = self.GP.shape[0]*2
        self.H = torch.zeros([Nrow,Ncol]).float()
        self.G = torch.zeros([Nrow,Ncol]).float()
        self.G_log = torch.zeros([Nrow,2*Ngauss_log]).float()
        self.C = 0.5*torch.ones([Nrow]).float()
        logLE = np.log(self.LE/2)#虽然很蠢但是为了效率最好把这个运算放在外面并行
        for i in range(Nrow//2):
            R = self.GP-self.Source[i]
            fs,dfs = fundamental_CPVLOG(R,self.norm.repeat(Ngauss,axis=0),logLE[i],i,para=1,E=self.E,v=self.v)
            fs = fs*self.weightT.reshape([-1,1])
            dfs = dfs*self.weightT.reshape([-1,1])
            self.G[i,:] = torch.cat([fs[:,0],fs[:,1]])
            self.G[Nrow//2+i,:] = torch.cat([fs[:,2],fs[:,3]])
            self.H[i,:] = torch.cat([dfs[:,0],dfs[:,1]])
            self.H[Nrow//2+i,:] = torch.cat([dfs[:,2],dfs[:,3]])
            
            # self.G[i,:],self.H[i,:]=fundamental(R,self.norm.repeat(Ngauss,axis=0),para=1)
            # self.G[i,i*Ngauss:(i+1)*Ngauss] = torch.tensor(np.log(self.LE[i]/2)/-2/np.pi).float()
            # self.G[i,:]  =self.G[i,:]*self.weightT
            # self.H[i,:]  =self.H[i,:]*self.weightT
            # self.H[i,i*Ngauss:(i+1)*Ngauss]=0.
            self.G_log[i,:] = torch.tensor(self.LE[i]/2*self.A1*self.A2*np.tile(Gaussweight_log,2)).float()
            self.G_log[Nrow//2+i,:] = torch.tensor(self.LE[i]/2*self.A1*self.A2*np.tile(Gaussweight_log,2)).float()
        # self.b = torch.zeros([Nrow]).float()
        U_geo = self.C*self.fac_source.view(-1)
        
        N_log = self.GPlog.shape[0]#有多少个对数型高斯积分点
        # T_log = torch.cat([torch.sum(self.G_log[0:Nrow//2,0:Ngauss_log*2]*self.dfac_log[0:N_log].reshape([-1,Ngauss_log*2])+\
        #                             self.G_log[0:Nrow//2,Ngauss_log*2:Ngauss_log*4]*self.dfac_log[N_log:2*N_log].reshape([-1,Ngauss_log*2]),axis=1),\
        #                    torch.sum(self.G_log[Nrow//2:Nrow,0:Ngauss_log*2]*self.dfac_log[0:N_log].reshape([-1,Ngauss_log*2])+\
        #                              self.G_log[Nrow//2:Nrow,Ngauss_log*2:Ngauss_log*4]*self.dfac_log[N_log:2*N_log].reshape([-1,Ngauss_log*2]),axis=1)])
        T_log = torch.cat([torch.sum(self.G_log[0:Nrow//2,:]*self.dfac_log[0:N_log].reshape([-1,Ngauss_log*2]),axis=1),\
                           torch.sum(self.G_log[Nrow//2:Nrow,:]*self.dfac_log[N_log:2*N_log].reshape([-1,Ngauss_log*2]),axis=1)])
        D_work = torch.mv(self.H,self.fac.view(-1))+ U_geo\
        -torch.mv(self.G,self.dfac.view(-1)) - T_log
         
            
        U_geo[self.tlist] =0.
        T_log[self.ulist] =0.
        
       
        self.b = torch.mv(self.H[:,self.ucol_index],self.fac[self.ucol_index].view(-1))+ U_geo\
        -torch.mv(self.G[:,self.tcol_index],self.dfac[self.tcol_index].view(-1)) - T_log
        self.b=-self.b#这里算b需要加负号，因为后面b2没加但是都默认H为正，为了用MSE必须让它们反号

    def update_func(self,Net):
        normT = torch.tensor(self.norm.repeat(Ngauss,axis=0)).float()
        normT2 = torch.tensor(self.norm.repeat(2*Ngauss_log,axis=0)).float()
        # self.f = Net(self.GPT)
        # gradient = torch.ones(self.f[:,0].size())
        # #self.f.backward(gradient)
        # # self.df=torch.mv(self.GPT.grad,self.normT)
        # df1=torch.autograd.grad(self.f[:,0],self.GPT,grad_outputs=gradient,
        #                        retain_graph=True,
        #                        create_graph=True,
        #                        only_inputs=True,
        #                        allow_unused=True)[0]
        # df2=torch.autograd.grad(self.f[:,1],self.GPT,grad_outputs=gradient,
        #                        retain_graph=True,
        #                        create_graph=True,
        #                        only_inputs=True,
        #                        allow_unused=True)[0]
        
        # self.df = 2*self.shear*self.v/(1-2*self.v)*torch.cat([(df1[:,0]+df2[:,1])*normT[:,0],\
        #                                            (df1[:,0]+df2[:,1])*normT[:,1]])+\
        #     self.shear*(torch.cat([2*df1[:,0]*normT[:,0]+(df1[:,1]+df2[:,0])*normT[:,1],\
        #                        2*df2[:,1]*normT[:,1]+(df1[:,1]+df2[:,0])*normT[:,0]]))
        
        self.f,self.df = self.compute_traction(Net,self.GPT,normT)
        
        _,self.df_log = self.compute_traction(Net,self.GPTlog,normT2)
        
        # GP_f2 = Net(self.GPTlog)
        # gradient2 = torch.ones(GP_f2[:,0].size())
        # df_log1=torch.autograd.grad(GP_f2[:,0],self.GPTlog,grad_outputs=gradient2,
        #                        retain_graph=True,
        #                        create_graph=True,
        #                        only_inputs=True,
        #                        allow_unused=True)[0]
        # df_log2=torch.autograd.grad(GP_f2[:,1],self.GPTlog,grad_outputs=gradient2,
        #                        retain_graph=True,
        #                        create_graph=True,
        #                        only_inputs=True,
        #                        allow_unused=True)[0]
        # normT2 = torch.tensor(self.norm.repeat(2*Ngauss_log,axis=0))
        
        # self.df_log = 2*self.shear*self.v/(1-2*self.v)*torch.cat([(df1[:,0]+df2[:,1])*normT[:,0],\
        #                                            (df1[:,0]+df2[:,1])*normT[:,1]])+\
        #     self.shear*(torch.cat([2*df1[:,0]*normT[:,0]+(df1[:,1]+df2[:,0])*normT[:,1],\
        #                        2*df2[:,1]*normT[:,1]+(df1[:,1]+df2[:,0])*normT[:,0]]))
        
        
        # self.df_log=torch.sum(torch.autograd.grad(GP_f2,self.GPTlog,grad_outputs=gradient2,
        #                        retain_graph=True,
        #                        create_graph=True,
        #                        only_inputs=True,
        #                        allow_unused=True)[0]*torch.tensor(self.norm.repeat(2*Ngauss_log,axis=0)).float(),axis=1).reshape([-1,1])
        
        f_sou = Net(self.SourceT)
        self.f_source = torch.cat([f_sou[:,0],f_sou[:,1]])
    
    def compute_traction(self,Net,x,normT):
        #输入网络，坐标点，每个坐标点的法向
        #返回t，并且是t1，t2分开返回
        f = Net(x)
        gradient = torch.ones(f[:,0].size())
        #self.f.backward(gradient)
        # self.df=torch.mv(self.GPT.grad,self.normT)
        df1=torch.autograd.grad(f[:,0],x,grad_outputs=gradient,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True,
                               allow_unused=True)[0]
        df2=torch.autograd.grad(f[:,1],x,grad_outputs=gradient,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True,
                               allow_unused=True)[0]
        
        df = 2*self.shear*self.v/(1-2*self.v)*torch.cat([(df1[:,0]+df2[:,1])*normT[:,0],\
                                                   (df1[:,0]+df2[:,1])*normT[:,1]])+\
            self.shear*(torch.cat([2*df1[:,0]*normT[:,0]+(df1[:,1]+df2[:,0])*normT[:,1],\
                               2*df2[:,1]*normT[:,1]+(df1[:,1]+df2[:,0])*normT[:,0]]))
        f = torch.cat([f[:,0],f[:,1]])
        return f,df
        
        
    def update_loss(self,Net):
        Nrow = self.Source.shape[0]*2
        Ncol = self.GP.shape[0]*2
        U_geo0 = self.C*self.f_source.view(-1)#几何项与C相关
        U_geo0[self.ulist] =0.
        N_log = self.GPlog.shape[0]
        T_log0 = torch.cat([torch.sum(self.G_log[0:Nrow//2,:]*self.df_log[0:N_log].reshape([-1,Ngauss_log*2]),axis=1),\
                           torch.sum(self.G_log[Nrow//2:Nrow,:]*self.df_log[N_log:2*N_log].reshape([-1,Ngauss_log*2]),axis=1)])

        # T_log0 = torch.cat([torch.sum(self.G_log[0:Nrow//2,0:Ngauss_log*2]*self.df_log[0:N_log].reshape([-1,Ngauss_log*2])+\
        #                             self.G_log[0:Nrow//2,Ngauss_log*2:Ngauss_log*4]*self.df_log[N_log:2*N_log].reshape([-1,Ngauss_log*2]),axis=1),\
        #                    torch.sum(self.G_log[Nrow//2:Nrow,0:Ngauss_log*2]*self.df_log[0:N_log].reshape([-1,Ngauss_log*2])+\
        #                              self.G_log[Nrow//2:Nrow,Ngauss_log*2:Ngauss_log*4]*self.df_log[N_log:2*N_log].reshape([-1,Ngauss_log*2]),axis=1)])
        T_log0[self.tlist] =0.
        b2 = torch.mv(self.H[:,self.tcol_index],self.f[self.tcol_index].view(-1))+ U_geo0\
        -torch.mv(self.G[:,self.ucol_index],self.df[self.ucol_index].view(-1)) - T_log0
        loss = MSE(b2,self.b)
        return loss
       
    def inner(self,Net,x):
        #内点积分方案
        self.update_func(Net)
        Known_f = self.f.clone().detach()
        Known_df = self.df.clone().detach()
        Known_f[self.ucol_index] = self.fac[self.ucol_index]
        Known_df[self.tcol_index] = self.dfac[self.tcol_index]
        
        
        
        Nrow = x.shape[0]*2
        Ncol = self.GP.shape[0]*2
        H_in = torch.zeros([Nrow,Ncol]).float()
        G_in = torch.zeros([Nrow,Ncol]).float()
        for i in range(x.shape[0]):
            R = self.GP-x[i]
            fs,dfs = fundamental(R,self.norm.repeat(Ngauss,axis=0),para=1,E=self.E,v=self.v)
            fs = fs*self.weightT.reshape([-1,1])
            dfs = dfs*self.weightT.reshape([-1,1])
            G_in[i,:] = torch.cat([fs[:,0],fs[:,1]])
            G_in[Nrow//2+i,:] = torch.cat([fs[:,2],fs[:,3]])
            H_in[i,:] = torch.cat([dfs[:,0],dfs[:,1]])
            H_in[Nrow//2+i,:] = torch.cat([dfs[:,2],dfs[:,3]])
            
            
        U = -(torch.mv(H_in,Known_f)-torch.mv(G_in,Known_df))
        return U
        
def Generatepoints(x,y,b,h,func,BCtype0,NE = 100,E=1,v=0.3):
    #先将原来边界划分为数个网格，每个网格上计算高斯积分点对应的:1.物理坐标 2. 外法线矢量 3.边界函数值 4.边界导数值 5.附上对应的高斯积分权重 6.雅可比
    #选取数个源点，对每个源点计算基本解在积分点上的函数值与导数值，然后高斯积分求和计算互等定理的两个积分值，计算残差
    #将残差累加，得到Loss函数
    #对于给定的边界条件，积分应该采用给定值，但是是否需要约束使神经网络满足边界条件？
    #
    #
    
    
    #rect = np.array([x,y,b,h])
    lines  = np.zeros([4,2,2])
    lines[0,:,:] = [[x-b/2,y-h/2],[x+b/2,y-h/2]]
    lines[1,:,:] = [[x+b/2,y-h/2],[x+b/2,y+h/2]]
    lines[2,:,:] = [[x+b/2,y+h/2],[x-b/2,y+h/2]]
    lines[3,:,:] = [[x-b/2,y+h/2],[x-b/2,y-h/2]]
    
    # BCtype0 = [1,1,1,1,1,1,1,1];
    # BCtype0 = [0,1,1,1,0,1,1,1];
    # BCvalue = [np.ones([Ngauss,1]),np.zeros([Ngauss,1]),11*np.ones([Ngauss,1]),np.zeros([Ngauss,1])]
    p0 = np.zeros([4*NE,2])
    p1 = np.zeros([4*NE,2])
    p0[0:NE,0]=np.linspace(lines[0,0,0],lines[0,1,0],NE,endpoint=False)
    p0[NE:2*NE,0]=np.linspace(lines[1,0,0],lines[1,1,0],NE,endpoint=False)
    p0[2*NE:3*NE,0]=np.linspace(lines[2,0,0],lines[2,1,0],NE,endpoint=False)
    p0[3*NE:4*NE,0]=np.linspace(lines[3,0,0],lines[3,1,0],NE,endpoint=False)
    
    p0[0:NE,1]=np.linspace(lines[0,0,1],lines[0,1,1],NE,endpoint=False)
    p0[NE:2*NE,1]=np.linspace(lines[1,0,1],lines[1,1,1],NE,endpoint=False)
    p0[2*NE:3*NE,1]=np.linspace(lines[2,0,1],lines[2,1,1],NE,endpoint=False)
    p0[3*NE:4*NE,1]=np.linspace(lines[3,0,1],lines[3,1,1],NE,endpoint=False)
    
    p1[0:4*NE-1,:] = p0[1:4*NE,:]
    p1[4*NE-1,:] =p0[0,:]
    BCtype = np.zeros([4*NE])
    
    BCtype = np.repeat(BCtype0, NE)
        
    Grids = grids_HB(p0, p1, func,BCtype,E,v)
    
                 
    return Grids                

def fundamental(R,Norm,para=0,E=1,v=0.3):
    A1=-1/8/np.pi/(1-v)/(E/2/(1+v))
    A2=3-4*v
    B1 = -1/4/np.pi/(1-v)
    B2 = 1-2*v
    
    LR = np.linalg.norm(R,axis=-1)
    r = R/LR.reshape([-1,1])
   
    
    fs = np.zeros([R.shape[0],4])
    dfs = np.zeros([R.shape[0],4])
    fs[:,0] = A1*(A2*np.log(LR)-r[:,0]*r[:,0])
    fs[:,1] = A1*(-r[:,0]*r[:,1])
    fs[:,2] = A1*(-r[:,1]*r[:,0])
    fs[:,3] = A1*(A2*np.log(LR)-r[:,1]*r[:,1])
    if para==0:
        drdn = r[:,0]*Norm[0]+r[:,1]*Norm[1]
        dfs[:,0] = B1/LR*((B2+2*r[:,0]*r[:,0])*drdn)
        dfs[:,1] = B1/LR*((   2*r[:,0]*r[:,1])*drdn - B2*(r[:,0]*Norm[1]-r[:,1]*Norm[0]))
        dfs[:,2] = B1/LR*((   2*r[:,0]*r[:,1])*drdn - B2*(r[:,1]*Norm[0]-r[:,0]*Norm[1]))
        dfs[:,3] = B1/LR*((B2+2*r[:,1]*r[:,1])*drdn)
        
    else:
        drdn = r[:,0]*Norm[:,0]+r[:,1]*Norm[:,1]
        dfs[:,0] = B1/LR*((B2+2*r[:,0]*r[:,0])*drdn)
        dfs[:,1] = B1/LR*((   2*r[:,0]*r[:,1])*drdn - B2*(r[:,0]*Norm[:,1]-r[:,1]*Norm[:,0]))
        dfs[:,2] = B1/LR*((   2*r[:,0]*r[:,1])*drdn - B2*(r[:,1]*Norm[:,0]-r[:,0]*Norm[:,1]))
        dfs[:,3] = B1/LR*((B2+2*r[:,1]*r[:,1])*drdn)
    fs = torch.tensor(fs,dtype=torch.float32)
    dfs = torch.tensor(dfs,dtype=torch.float32)
    return fs,dfs

def fundamental_CPVLOG(R,Norm,logL,i,para=0,E=1,v=0.3):
    #这个算法加上了i，指代源点在第i个单元,logL是第i个单元的长度一半的对数ln(LE/2)
    A1=-1/8/np.pi/(1-v)/(E/2/(1+v))
    A2=3-4*v
    B1 = -1/4/np.pi/(1-v)
    B2 = 1-2*v  
    #以下代码块是从fundamental（）直接复制而来，应该同步更新
    LR = np.linalg.norm(R,axis=-1)
    r = R/LR.reshape([-1,1])
   
    
    fs = np.zeros([R.shape[0],4])
    dfs = np.zeros([R.shape[0],4])
    fs[:,0] = A1*(A2*np.log(LR)-r[:,0]*r[:,0])
    fs[:,1] = A1*(-r[:,0]*r[:,1])
    fs[:,2] = A1*(-r[:,1]*r[:,0])
    fs[:,3] = A1*(A2*np.log(LR)-r[:,1]*r[:,1])
    if para==0:
        drdn = r[:,0]*Norm[0]+r[:,1]*Norm[1]
        dfs[:,0] = B1/LR*((B2+2*r[:,0]*r[:,0])*drdn)
        dfs[:,1] = B1/LR*((   2*r[:,0]*r[:,1])*drdn - B2*(r[:,0]*Norm[1]-r[:,1]*Norm[0]))
        dfs[:,2] = B1/LR*((   2*r[:,0]*r[:,1])*drdn - B2*(r[:,1]*Norm[0]-r[:,0]*Norm[1]))
        dfs[:,3] = B1/LR*((B2+2*r[:,1]*r[:,1])*drdn)
        
    else:
        drdn = r[:,0]*Norm[:,0]+r[:,1]*Norm[:,1]
        dfs[:,0] = B1/LR*((B2+2*r[:,0]*r[:,0])*drdn)
        dfs[:,1] = B1/LR*((   2*r[:,0]*r[:,1])*drdn - B2*(r[:,0]*Norm[:,1]-r[:,1]*Norm[:,0]))
        dfs[:,2] = B1/LR*((   2*r[:,0]*r[:,1])*drdn - B2*(r[:,1]*Norm[:,0]-r[:,0]*Norm[:,1]))
        dfs[:,3] = B1/LR*((B2+2*r[:,1]*r[:,1])*drdn)
    # fs = torch.tensor(fs,dtype=torch.float32)
    # dfs = torch.tensor(dfs,dtype=torch.float32)
    #-----------------------------------------------
    ind = np.arange(i*Ngauss,(i+1)*Ngauss)
    fs[ind,0] = A1*A2*logL-A1*r[ind,0]*r[ind,0]
    fs[ind,3] = A1*A2*logL-A1*r[ind,1]*r[ind,1]
    dfs[ind,:] = 0
    dfs[ind,1] = B1/LR[ind]*( - B2*(r[ind,0]*Norm[ind,1]-r[ind,1]*Norm[ind,0]))
    dfs[ind,2] = B1/LR[ind]*( - B2*(r[ind,1]*Norm[ind,0]-r[ind,0]*Norm[ind,1]))
    
    fs = torch.tensor(fs,dtype=torch.float32)
    dfs = torch.tensor(dfs,dtype=torch.float32)
    return fs,dfs
    