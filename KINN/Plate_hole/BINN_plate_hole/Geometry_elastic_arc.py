# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 21:57:21 2022

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

class arc_geometry():
    def __init__(self,r,x,y,NE,p1=0,p2=2*np.pi):
        '''
        Parameters
        ----------
        p1 : array
            start angle of the arc.
        p2 : array
            end angle of the arc.
        r : array
            radius of the arc.
        c : array
            center of the arc.
        func : function
            DESCRIPTION.
        type0 : element type
            表示该单元函数值是给定（0）或者没给定（1）
            
        Returns
        -------
        None.

        '''
        self.r=r
        theta = np.linspace(p1,p2,NE+1,endpoint=True)
        self.sourcetheta = (theta[0:-1]+theta[1:NE+1])/2
        dtheta = (p2-p1)/NE
        self.dtheta = dtheta
        self.source = np.zeros([NE,2])
        self.source[:,0] = r*np.cos(self.sourcetheta)+x
        self.source[:,1] = r*np.sin(self.sourcetheta)+y
        self.c_jacobi = dtheta*r/2#jacobi to ksi
        self.c_gp = np.zeros([NE*Ngauss,2])
        self.weak_jacobi = r*np.cos(Gausspoint*dtheta/2)*dtheta/2
        self.gptheta = np.repeat(self.sourcetheta,Ngauss)+np.tile(Gausspoint*dtheta/2,NE)
        self.c_gp[:,0] = x+r*np.cos(self.gptheta)
        self.c_gp[:,1] = y+r*np.sin(self.gptheta)
        #算法线时要注意是朝外还是朝里，孔问题是朝里
        self.c_gpnorm = np.array([np.cos(self.gptheta),np.sin(self.gptheta)]).T*np.sign(p2-p1)
        self.source_norm = np.array([np.cos(self.sourcetheta),np.sin(self.sourcetheta)]).T*np.sign(p2-p1)
        # self.ds = 2*self.r*abs(np.sin(dtheta/2))
        self.string = 2*r*np.sin(dtheta/4)
        
        plt.scatter(self.c_gp[:,0],self.c_gp[:,1])


class arc_HB():
    def __init__(self,r,c,func,type0,NE,E=1,v=0.3):
        '''
        Parameters
        ----------
        p1 : array
            start angle of the grids.
        p2 : array
            end angle of the grids.
        r : array
            radius of each grid.
        c : array
            center of each grid.
        func : function
            DESCRIPTION.
        type0 : element type
            表示该单元函数值是给定（0）或者没给定（1）
            
        Returns
        -------
        None.

        '''
        self.E = E
        self.v = v
        self.shear = E/2/(1+v)
        self.c=c
        self.r=r
        self.NE=NE
        type0 = np.repeat(type0, NE)
        self.type = type0
        C1 = arc_geometry(r,c[0],c[1],NE,p1=0,p2=2*np.pi)
        # C1 = arc_geometry(r,c[0],c[1],NE)
        self.GP = C1.c_gp
        self.norm = C1.c_gpnorm
        self.jacobi = C1.c_jacobi
        self.string = C1.string
        self.Source = C1.source
        self.Sourcenorm = C1.source_norm
        self.weakjacobi = C1.weak_jacobi
        
        self.GPT = torch.tensor(self.GP,dtype = torch.float32,requires_grad=True)   
        self.SourceT = torch.tensor(self.Source,dtype = torch.float32,requires_grad=True)   
        self.weightT = torch.tensor(np.tile(Gaussweight,self.NE)*self.jacobi).float()
        # ------------------------------------------------
        
        self.func = func
        
        
        self.ulist = np.array(np.where(type0==0))[0]#给定位移
        self.tlist = np.array(np.where(type0==1))[0]#给定力
        self.Nulist = np.array(np.where(type0!=0))[0]#未给定位移
        self.Ntlist = np.array(np.where(type0!=1))[0]#未给定力
        
        self.fac = torch.tensor(self.func(self.GP,self.norm,0,para=1),dtype = torch.float32)
        self.fac_source = torch.tensor(self.func(self.Source,self.Sourcenorm,0,para=1),dtype = torch.float32)
        self.dfac = torch.tensor(self.func(self.GP,self.norm,1,para=1),dtype = torch.float32)
        self.dfac_source = torch.tensor(self.func(self.Source,self.Sourcenorm,1,para=1),dtype = torch.float32)
        
        self.ucol_index = np.tile(np.array(range(Ngauss)),self.ulist.shape[0])
        self.ucol_index+=self.ulist.repeat(Ngauss,axis=0)*Ngauss
        self.tcol_index = np.tile(np.array(range(Ngauss)),self.tlist.shape[0])
        self.tcol_index+=self.tlist.repeat(Ngauss,axis=0)*Ngauss
        self.Nu_index = np.tile(np.array(range(Ngauss)),self.Nulist.shape[0])
        self.Nu_index+=self.Nulist.repeat(Ngauss,axis=0)*Ngauss
        self.Nt_index = np.tile(np.array(range(Ngauss)),self.Ntlist.shape[0])
        self.Nt_index+=self.Ntlist.repeat(Ngauss,axis=0)*Ngauss
        
        self.solution = self.fac.cpu().numpy().copy()
        self.solution[self.ucol_index] = self.dfac.cpu().numpy()[self.ucol_index].copy()
        self.assemble_matrix()
        
        
    def assemble_matrix(self):
        #这里是圆弧单元，并且源点在单元中心，柯西主值积分是采用longman的方法计算的
        #注意本部分的所有变量都不含梯度，本部分的计算与网络无关
        Nrow = self.Source.shape[0]*2
        Ncol = self.GP.shape[0]*2
        self.H = torch.zeros([Nrow,Ncol]).float()
        self.G = torch.zeros([Nrow,Ncol]).float()
        self.G_log = torch.zeros([Nrow//2]).float()
        self.C = 0.5*torch.ones([Nrow]).float()
        for i in range(Nrow//2):
            R = self.GP-self.Source[i]
            fs,dfs,fslog = fundamental_CPVLOG2(R,self.norm,i,self.weakjacobi,para=1)
            fs = fs*self.weightT.reshape([-1,1])
            dfs = dfs*self.weightT.reshape([-1,1])
            
            fslog = fslog*torch.tensor(Gaussweight).float()
            
            self.G_log[i] = torch.sum(fslog)
            
            self.G[i,:] = torch.cat([fs[:,0],fs[:,1]])
            self.G[Nrow//2+i,:] = torch.cat([fs[:,2],fs[:,3]])
            self.H[i,:] = torch.cat([dfs[:,0],dfs[:,1]])
            self.H[Nrow//2+i,:] = torch.cat([dfs[:,2],dfs[:,3]])
            
        self.G_log+=external_term(self.string,E=1,v=0.3,mode='arc')
        # self.b = torch.zeros([Nrow]).float()
        U_geo = self.C*self.fac_source.view(-1)
        U_geo[self.tlist] =0.
        
        T_log = torch.cat([self.G_log*self.dfac_source[0:self.Source.shape[0]],\
                          self.G_log*self.dfac_source[self.Source.shape[0]:2*self.Source.shape[0]]])
        T_log[self.ulist] =0.
        D_work = torch.mv(self.H,self.fac.view(-1))+ U_geo\
        -torch.mv(self.G,self.dfac.view(-1)) - T_log
        
        self.b = torch.mv(self.H[:,self.ucol_index],self.fac[self.ucol_index].view(-1))+ U_geo\
        -torch.mv(self.G[:,self.tcol_index],self.dfac[self.tcol_index].view(-1)) - T_log
        self.b=-self.b#这里算b需要加负号，因为后面b2没加但是都默认H为正，为了用MSE必须让它们反号

    def update_func(self,Net):
        normT = torch.tensor(self.norm).float()
        normSource = torch.tensor(self.Sourcenorm).float()
        
        self.f,self.df = self.compute_traction(Net,self.GPT,normT)
        self.f_source,self.df_source = self.compute_traction(Net,self.SourceT,normSource)
        
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
       
        T_log0 = torch.cat([self.G_log*self.df_source[0:self.Source.shape[0]],\
                          self.G_log*self.df_source[self.Source.shape[0]:2*self.Source.shape[0]]])

        # T_log0 = torch.cat([torch.sum(self.G_log[0:Nrow//2,0:Ngauss_log*2]*self.df_log[0:N_log].reshape([-1,Ngauss_log*2])+\
        #                             self.G_log[0:Nrow//2,Ngauss_log*2:Ngauss_log*4]*self.df_log[N_log:2*N_log].reshape([-1,Ngauss_log*2]),axis=1),\
        #                    torch.sum(self.G_log[Nrow//2:Nrow,0:Ngauss_log*2]*self.df_log[0:N_log].reshape([-1,Ngauss_log*2])+\
        #                              self.G_log[Nrow//2:Nrow,Ngauss_log*2:Ngauss_log*4]*self.df_log[N_log:2*N_log].reshape([-1,Ngauss_log*2]),axis=1)])
        T_log0[self.tlist] =0.
        
        # b2 = torch.mv(self.H[:,self.tcol_index],self.f[self.tcol_index].view(-1))+ U_geo0\
        # -torch.mv(self.G[:,self.ucol_index],self.df[self.ucol_index].view(-1)) - T_log0
        b2 = torch.mv(self.H[:,self.Nu_index],self.f[self.Nu_index].view(-1))+ U_geo0\
        -torch.mv(self.G[:,self.Nt_index],self.df[self.Nt_index].view(-1)) - T_log0
        
        loss = MSE(b2,self.b)
        return loss
       
    def inner(self,Net,x,mode=0):
        #内点积分方案
        self.update_func(Net)
        Known_f = self.f.clone().detach()
        Known_df = self.df.clone().detach()
        Known_f[self.ucol_index] = self.fac[self.ucol_index]
        Known_df[self.tcol_index] = self.dfac[self.tcol_index]
        if mode==0:
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

def fundamental_CPVLOG2(R,Norm,i,weakjacobi,para=0,E=1,v=0.3):
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
    fs[:,3] = A1*(A2*np.log(LR)-r[:,1]*r[:,1])#按照新的算法可以不对fs做处理
    
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
    # dfs[ind,:] = 0
    # dfs[ind,1] = B1/LR[ind]*( - B2*(r[ind,0]*Norm[ind,1]-r[ind,1]*Norm[ind,0]))
    # dfs[ind,2] = B1/LR[ind]*( - B2*(r[ind,1]*Norm[ind,0]-r[ind,0]*Norm[ind,1]))
    
    fs_log = -A1*(A2*np.log(LR[ind]))*weakjacobi
    fs = torch.tensor(fs,dtype=torch.float32)
    dfs = torch.tensor(dfs,dtype=torch.float32)
    fs_log = torch.tensor(fs_log,dtype=torch.float32)
    return fs,dfs,fs_log
def external_term(string,E=1,v=0.3,mode='arc'):
    #计算近奇异积分中加回来的解析部分
    A1=-1/8/np.pi/(1-v)/(E/2/(1+v))
    A2=3-4*v
    if mode=='arc':
        
        ext_f = 2*A1*A2*string*(np.log(abs(string))-1)
    elif mode=='line':
        ext_f = 2*A1*A2*string*(np.log(abs(string))-1)
    return ext_f
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
# def fundamental_CPVLOG(R,Norm,i,weakjacobi,para=0):
#     LR = np.linalg.norm(R,axis=-1)
#     fs = -np.log(LR)/2/np.pi
#     if para==0:
#         dfs = -np.dot(R,Norm)/2/np.pi/LR/LR
#     else:
#         dfs = -(R[:,0]*Norm[:,0]+R[:,1]*Norm[:,1])/2/np.pi/LR/LR

#     ind = np.arange(i*Ngauss,(i+1)*Ngauss)
#     fs_log = np.log(LR[ind])/2/np.pi*weakjacobi
#     fs = torch.tensor(fs,dtype=torch.float32)
#     dfs = torch.tensor(dfs,dtype=torch.float32)
#     fs_log = torch.tensor(fs_log,dtype=torch.float32)
#     return fs,dfs,fs_log