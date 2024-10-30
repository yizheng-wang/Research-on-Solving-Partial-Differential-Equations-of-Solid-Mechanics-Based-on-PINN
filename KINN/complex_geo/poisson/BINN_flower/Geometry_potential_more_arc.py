# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 21:17:21 2022

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
        修改：
        1.在self.c_jacobi与self.weak_jacobi都加上了绝对值
        2.法向要根据方向进行改变

        '''
        
        self.r=r
        theta = np.linspace(p1,p2,NE+1,endpoint=True)
        self.sourcetheta = (theta[0:-1]+theta[1:NE+1])/2
        dtheta = (p2-p1)/NE
        self.dtheta = dtheta
        self.source = np.zeros([NE,2])
        self.source[:,0] = r*np.cos(self.sourcetheta)+x
        self.source[:,1] = r*np.sin(self.sourcetheta)+y
        self.c_jacobi = np.abs(dtheta)*r/2#jacobi to ksi
        self.c_gp = np.zeros([NE*Ngauss,2])
        self.weak_jacobi = r*np.cos(Gausspoint*dtheta/2)*np.abs(dtheta)/2
        self.gptheta = np.repeat(self.sourcetheta,Ngauss)+np.tile(Gausspoint*dtheta/2,NE)
        self.c_gp[:,0] = x+r*np.cos(self.gptheta)
        self.c_gp[:,1] = y+r*np.sin(self.gptheta)
        #算法线时要注意是朝外还是朝里，孔问题是朝里
        self.c_gpnorm = np.array([np.cos(self.gptheta),np.sin(self.gptheta)]).T*np.sign(p2-p1)
        self.source_norm = np.array([np.cos(self.sourcetheta),np.sin(self.sourcetheta)]).T*np.sign(p2-p1)
        # self.ds = 2*self.r*abs(np.sin(dtheta/2))
        self.string = 2*r*np.sin(dtheta/4)
        
        self.testtheta = np.linspace(p1,p2,20*(NE+1),endpoint=True)
        self.testpoint = np.zeros([20*(NE+1),2])
        self.testpoint[:,0] = r*np.cos(self.testtheta)+x
        self.testpoint[:,1] = r*np.sin(self.testtheta)+y
        self.testpoint_norm = np.array([np.cos(self.testtheta),np.sin(self.testtheta)]).T*np.sign(p2-p1)
        plt.scatter(self.c_gp[:,0],self.c_gp[:,1],s=1)
        
    # def arc_generate_GP(self,p1,p2,r,c):
    #     NE=p1.shape[0]
    #     p1 = p1
    #     p2 = p2
    #     dtheta = p2-p1
    #     LE = dtheta*r
    #     jacobi = LE/2
    #     jacobi_line = np.repeat(jacobi,Ngauss)
    #     weightT =torch.tensor(np.tile(Gaussweight,NE)*jacobi_line).float()
    #     Source =np.concatenate((r*np.cos((p2+p1)/2),r*np.sin((p2+p1)/2)),axis=0).T
        
    #     GP = np.zeros([NE*Ngauss,2])
        
    #     GPtheta = np.repeat(Source,Ngauss)+np.tile(Gausspoint*dtheta/2,NE)
    #     GP[:,0] = x+r*np.cos(GPtheta)
    #     GP[:,1] = y+r*np.sin(GPtheta)
        
    #     GPnorm = np.array([np.cos(GPtheta),np.sin(GPtheta)]).T
    #     return Source,GP,GPtheta,GPnorm,weightT
    
    # def arc_generate_logGP(self):
        
    #     tjacobi,t_theta = self.caulculate_tjacobi(Gausspoint_log)
    #     self.log_gptheta = np.repeat([self.sourcetheta],Ngauss_log,axis=0).T+t_theta
    #     self.log_gptheta = np.concatenate((self.log_gptheta,\
    #     np.repeat([self.nodetheta],Ngauss_log,axis=0).T-t_theta),axis=1)
        
    #     self.t_gp = np.zeros([self.n,2*Ngauss_log,2])
    #     self.t_gp[:,:,0] = self.c[0]+self.r*np.cos(self.log_gptheta)
    #     self.t_gp[:,:,1] = self.c[1]+self.r*np.sin(self.log_gptheta)
        
    #     self.t_gpnorm = np.zeros([self.n,2*Ngauss_log,2])
    #     self.t_gpnorm[:,:,0]=np.cos(self.log_gptheta)
    #     self.t_gpnorm[:,:,1]=np.sin(self.log_gptheta)
        
    #     self.t_logint = -np.tile(tjacobi,2)/2/np.pi*np.tile(Gaussweight_log,2)
        
    # def caulculate_tjacobi(self,t):
    #     tjacobi = self.ds/np.sqrt(1-(t*self.ds/2/self.r)**2)
    #     t_theta = 2*np.arcsin(t*abs(np.sin(self.dtheta/2)))
    #     return tjacobi,t_theta
        

class arcs_HB():
    def __init__(self,r,c,NE,func,type0,theta1=[0],theta2=[2*np.pi],device = 'cpu'):
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
        # self.NE=p1.shape[0]
        # self.p1 = p1
        # self.p2 = p2
        # # self.GP = np.array((p1+p2)/2)
        # # self.GP = np.random.rand(p1.shape[0]).reshape(-1,1)*(p2-p1)+p1
        # rE = p2-p1
        # LE = np.linalg.norm(rE,axis=1)
        # self.Source = (p2+p1)/2
        # self.SourceT = torch.tensor(self.Source,dtype=torch.float32,requires_grad=True)
        # self.LE =LE
        # #weights including jacobi
        # jacobi = self.LE/2
        # jacobi_line = np.repeat(jacobi,Ngauss)
        # self.weightT =torch.tensor(np.tile(Gaussweight,self.NE)*jacobi_line).float()
        
        # self.GP = np.empty([self.NE*Ngauss,2])
        # for i in range(self.NE):
        #     self.GP[i*Ngauss:(i+1)*Ngauss,:] = self.Source[i]+ np.outer(Gausspoint,rE[i])/2 
        # self.GPT = torch.tensor(self.GP,dtype = torch.float32,requires_grad=True)   
        
        # self.GPlog = np.empty([2*self.NE*Ngauss_log,2])
          
        # for i in range(self.NE):
        #     self.GPlog[2*i*Ngauss_log:2*(i+1)*Ngauss_log,:] = np.append(self.Source[i]- np.outer(Gausspoint_log,rE[i]),\
        #                                                         self.Source[i]+ np.outer(Gausspoint_log,rE[i]),axis=0)
        # self.GPTlog = torch.tensor(self.GPlog,dtype = torch.float32,requires_grad=True)
        
        # self.norm = np.empty(rE.shape)
        # self.norm[:,0] = rE[:,1]/LE
        # self.norm[:,1] = -rE[:,0]/LE
        # self.normT = torch.tensor(self.norm,dtype = torch.float32) 
        self.N_arc = len(r)
        self.c=c
        self.r=r
        self.NE=sum(NE)
        self.type = type0
        
        # C1 = arc_geometry(r,c[0],c[1],NE,p1=2*np.pi,p2=0)
        # # C1 = arc_geometry(r,c[0],c[1],NE)
        # self.GP = C1.c_gp
        # self.norm = C1.c_gpnorm
        # self.jacobi = C1.c_jacobi
        # self.string = C1.string
        # self.Source = C1.source
        # self.Sourcenorm = C1.source_norm
        # self.weakjacobi = C1.weak_jacobi
        
        # self.GPT = torch.tensor(self.GP,dtype = torch.float32,requires_grad=True)   
        # self.SourceT = torch.tensor(self.Source,dtype = torch.float32,requires_grad=True)   
        # self.weightT = torch.tensor(np.tile(Gaussweight,self.NE)*self.jacobi).float()

        # ------------------------------------------------
        self.device  = device
        self.type = type0
        self.GP = np.zeros([0,2],dtype=np.float64)
        self.norm = np.zeros([0,2],dtype=np.float64)
        self.jacobi = np.zeros([0])
        self.string = np.zeros([0])
        self.Source = np.zeros([0,2],dtype=np.float64)
        self.Sourcenorm = np.zeros([0,2],dtype=np.float64)
        self.weakjacobi = np.zeros([0,Ngauss])
        self.testpoint  = np.zeros([0,2],dtype=np.float64)
        self.testpoint_norm = np.zeros([0,2],dtype=np.float64)
        for i in range(self.N_arc):
            arc = arc_geometry(r[i],c[i,0],c[i,1],NE[i],p1=theta1[i],p2=theta2[i])
            self.GP = np.concatenate([self.GP,arc.c_gp],axis=0)
            self.norm = np.concatenate([self.norm,arc.c_gpnorm],axis=0)
            self.jacobi = np.concatenate([self.jacobi,np.repeat(arc.c_jacobi,NE[i]*Ngauss)],axis=0)
            self.string = np.concatenate([self.string,np.repeat(arc.string,NE[i])],axis=0)
            self.Source = np.concatenate([self.Source,arc.source],axis=0)
            self.Sourcenorm = np.concatenate([self.Sourcenorm,arc.source_norm],axis=0)
            self.weakjacobi = np.concatenate([self.weakjacobi,np.tile(arc.weak_jacobi,[NE[i],1])],axis=0)
            self.testpoint = np.concatenate([self.testpoint,arc.testpoint],axis=0)
            self.testpoint_norm = np.concatenate([self.testpoint_norm,arc.testpoint_norm],axis=0)
        self.GPT = torch.tensor(self.GP,dtype = torch.float32,requires_grad=True)   
        self.SourceT = torch.tensor(self.Source,dtype = torch.float32,requires_grad=True)   
        self.weightT = torch.tensor(np.tile(Gaussweight,self.NE)*self.jacobi).float()
        self.plot_points()
        
        #-------------------------------------------------
        self.func = func
        
        
        self.ulist = np.array(np.where(self.type==0))[0]#给定位移
        self.tlist = np.array(np.where(self.type==1))[0]#给定力
        
        
        self.fac = torch.tensor(self.func(self.GP,self.norm,0,para=1),dtype = torch.float32)
        self.fac_source = torch.tensor(self.func(self.Source,self.Sourcenorm,0,para=1),dtype = torch.float32)
        self.dfac = torch.tensor(self.func(self.GP,self.norm,1,para=1),dtype = torch.float32)
        self.dfac_source = torch.tensor(self.func(self.Source,self.Sourcenorm,1,para=1),dtype = torch.float32)
        
        self.ucol_index = np.tile(np.array(range(Ngauss)),self.ulist.shape[0])
        self.ucol_index+=self.ulist.repeat(Ngauss,axis=0)*Ngauss
        self.tcol_index = np.tile(np.array(range(Ngauss)),self.tlist.shape[0])
        self.tcol_index+=self.tlist.repeat(Ngauss,axis=0)*Ngauss
        self.solution = self.fac.cpu().numpy().copy()
        self.solution[self.ucol_index] = self.dfac.cpu().numpy()[self.ucol_index].copy()
        self.assemble_matrix()
    
        
        
    def assemble_matrix(self):
        #这里是圆弧单元，并且源点在单元中心，柯西主值积分是采用longman的方法计算的
        #注意本部分的所有变量都不含梯度，本部分的计算与网络无关
        Nrow = self.Source.shape[0]
        Ncol = self.GP.shape[0]
        self.H = torch.zeros([Nrow,Ncol]).float()
        self.G = torch.zeros([Nrow,Ncol]).float()
        self.G_log = torch.zeros([Nrow]).float()
        self.C = 0.5*torch.ones([Nrow]).float()
        for i in range(Nrow):
            R = self.GP-self.Source[i]
            fs,dfs,fslog = fundamental_CPVLOG(R,self.norm,i,self.weakjacobi[i],para=1)

            self.G[i,:]  =fs*self.weightT
            self.H[i,:]  =dfs*self.weightT
            fslog = fslog*torch.tensor(Gaussweight).float()
            self.G_log[i] = torch.sum(fslog)
        self.G_log+=torch.tensor(-np.abs(self.string)*(np.log(np.abs(self.string))-1)/np.pi).float() 
        # self.b = torch.zeros([Nrow]).float()
        
        U_geo = self.C*self.fac_source.view(-1)
        T_log = self.G_log*self.dfac_source.view(-1)
        
        # D_work = torch.mv(self.H,self.fac.view(-1))+ U_geo\
        # -torch.mv(self.G,self.dfac.view(-1)) - T_log
        
        U_geo[self.tlist] =0.
        T_log[self.ulist] =0.
        self.b = torch.mv(self.H[:,self.ucol_index],self.fac[self.ucol_index].view(-1))+ U_geo\
        -torch.mv(self.G[:,self.tcol_index],self.dfac[self.tcol_index].view(-1)) - T_log
        self.b=-self.b#这里算b需要加负号，因为后面b2没加但是都默认H为正，为了用MSE必须让它们反号
        
    def update_func(self,Net):
        self.f = Net(self.GPT)
        gradient = torch.ones(self.f.size()).to(self.device)
        #self.f.backward(gradient)
        # self.df=torch.mv(self.GPT.grad,self.normT)
        self.df=torch.sum(torch.autograd.grad(self.f,self.GPT,grad_outputs=gradient,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True,
                               allow_unused=True)[0]*torch.tensor(self.norm).float().to(self.device),axis=1).reshape([-1,1])
        
        
        self.f_source = Net(self.SourceT)
        gradient2 = torch.ones(self.f_source.size()).to(self.device)
        #self.f.backward(gradient)
        # self.df=torch.mv(self.GPT.grad,self.normT)
        self.df_source=torch.sum(torch.autograd.grad(self.f_source,self.SourceT,grad_outputs=gradient2,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True,
                               allow_unused=True)[0]*torch.tensor(self.Sourcenorm).to(self.device).float(),axis=1).reshape([-1,1])
        
    def update_func_any(self,Net,xnumpy,xnormnumpy):  
        x=torch.tensor(xnumpy,dtype = torch.float32,requires_grad=True).to(self.device)
        
        fx = Net(x)
        gradient = torch.ones(fx.size()).to(self.device)
        #self.f.backward(gradient)
        # self.df=torch.mv(self.GPT.grad,self.normT)
        dfx=torch.sum(torch.autograd.grad(fx,x,grad_outputs=gradient,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True,
                               allow_unused=True)[0]*torch.tensor(xnormnumpy).float().to(self.device),axis=1).reshape([-1,1])
        return fx.detach().cpu().numpy(),dfx.detach().cpu().numpy()
        
    def update_loss(self,Net):
        U_geo0 = self.C*self.f_source.view(-1)
        U_geo0[self.ulist] =0.
        T_log0 = self.G_log*self.df_source.view(-1)
        T_log0[self.tlist] =0.
        
        b2 = torch.mv(self.H[:,self.tcol_index],self.f[self.tcol_index].view(-1))+ U_geo0\
        -torch.mv(self.G[:,self.ucol_index],self.df[self.ucol_index].view(-1)) - T_log0
        loss = MSE(b2,self.b)
        return loss
    
    def inner(self,Net,x,mode=1):
        #内点积分方案
        self.update_func(Net)
        Known_f = self.f.clone().detach()
        Known_df = self.df.clone().detach()
        Known_f[self.ucol_index] = self.fac[self.ucol_index]
        Known_df[self.tcol_index] = self.dfac[self.tcol_index]
        Known_f = Known_f.view(-1)
        Known_df = Known_df.view(-1)
        if mode==0:
            Nrow = x.shape[0]
            Ncol = self.GP.shape[0]
            H_in = torch.zeros([Nrow,Ncol]).float()
            G_in = torch.zeros([Nrow,Ncol]).float()
            
            
            for i in range(x.shape[0]):
                R = self.GP-x[i]
                fs,dfs = fundamental(R,self.norm,para=1)
                fs = fs*self.weightT
                dfs = dfs*self.weightT
                G_in[i,:] = fs
                
                H_in[i,:] = dfs
            U = -(torch.mv(H_in,Known_f)-torch.mv(G_in,Known_df))
        elif mode==1:
            Nrow = x.shape[0]
            Ncol = self.GP.shape[0]
            U = torch.zeros(Nrow).to(self.device)
            for i in range(Ncol):
                R = self.GP[i]-x
                fs,dfs = fundamental(R,self.norm[i],para=0)
                fs = fs*self.weightT[i]
                dfs = dfs*self.weightT[i]
                fs = fs.to(self.device)
                dfs=dfs.to(self.device)
                U+= -(dfs*Known_f[i]-fs*Known_df[i])
            
        # U = -(-torch.mv(G_in,Known_df.view(-1)))
        # U = -torch.mv(H_in,Known_f.view(-1))
        # print(torch.mv(G_in,Known_df.view(-1)))
        # print(-torch.mv(H_in,Known_f.view(-1)))
        # print(-(torch.mv(H_in,Known_f.view(-1))-torch.mv(G_in,Known_df.view(-1))))
        # print(U)
        return U
    
    
    def inner_D(self,Net,x,mode=1):
        #内点积分方案
        self.update_func(Net)
        Known_f = self.f.clone().detach().cpu().numpy()
        Known_df = self.df.clone().detach().cpu().numpy()
        Known_f[self.ucol_index] = self.fac[self.ucol_index].cpu().numpy()
        Known_df[self.tcol_index] = self.dfac[self.tcol_index].cpu().numpy()

        if mode==0:#按源点循环
            Nrow = x.shape[0]
            Ncol = self.GP.shape[0]
            DH_in = np.zeros([2,Nrow,Ncol])
            DG_in = np.zeros([2,Nrow,Ncol])
            
            
            for i in range(x.shape[0]):
                R = self.GP-x[i]
                Dfs,Ddfs = D_fundamental(R,self.norm,para=1)
                Dfs = (Dfs.T*self.weightT.cpu().numpy()).T
                Ddfs = (Ddfs.T*self.weightT.cpu().numpy()).T
                DG_in[0,i,:] = Dfs[:,0]
                DG_in[1,i,:] = Dfs[:,1]
                DH_in[0,i,:] = Ddfs[:,0]
                DH_in[1,i,:] = Ddfs[:,1]
            U1 = -((DH_in[0]@Known_f)-(DG_in[0]@Known_df)).flatten()
            U2 = -((DH_in[1]@Known_f)-(DG_in[1]@Known_df)).flatten()
        elif mode==1:#按场点循环
            Nrow = x.shape[0]
            Ncol = self.GP.shape[0]
            U1 = torch.zeros(Nrow)
            U2 = torch.zeros(Nrow)
            for i in range(Ncol):
                R = self.GP[i]-x
                Dfs,Ddfs = D_fundamental(R,self.norm[i],para=0)
                
                Dfs = Dfs*self.weightT[i].cpu().numpy()
                Ddfs = Ddfs*self.weightT[i].cpu().numpy()
                U1+=-(Ddfs[:,0]*Known_f[i]-Dfs[:,0]*Known_df[i])
                U2+=-(Ddfs[:,1]*Known_f[i]-Dfs[:,1]*Known_df[i])
                # U+= -(dfs*Known_f[i]-fs*Known_df[i])
        # U = -(-torch.mv(G_in,Known_df.view(-1)))
        # U = -torch.mv(H_in,Known_f.view(-1))
        # print(torch.mv(G_in,Known_df.view(-1)))
        # print(-torch.mv(H_in,Known_f.view(-1)))
        # print(-(torch.mv(H_in,Known_f.view(-1))-torch.mv(G_in,Known_df.view(-1))))
        # print(U)
        U = np.vstack((U1,U2))
        return U
    
    def plot_points(self):
        plt.figure(dpi=300,figsize=(4,4.0))
        plt.scatter(self.GP[:,0],self.GP[:,1],s=1,c='blue')#c='tab:blue'
        plt.scatter(self.Source[:,0],self.Source[:,1],s=4,c='red')#c='tab:orange'
        # plt.xlim(xmax=50,xmin=-50)
        # plt.ylim(ymax=50,ymin=-50)
        plt.axis('off')#隐藏坐标系
        # font = {'family': 'serif',
        # 'serif': 'Times New Roman',
        # 'weight': 'normal',
        # 'size': 14}
        # plt.rc('font',**font)
        # plt.legend(["exact","prediction"],loc='upper left', bbox_to_anchor=(0.01,1.21))
        # plt.plot(rect[0,:],rect[1,:],c='black')
        plt.axis("equal")
        
        plt.figure(dpi=300,figsize=(4,4.0))
        plt.plot(self.GP[0:40,0],self.GP[0:40,1],linewidth=1,c='black',zorder=1)
        plt.scatter(self.GP[0:40,0],self.GP[0:40,1],c='blue',s=4,zorder=2)#c='tab:blue',,
        plt.scatter(self.Source[0:4,0],self.Source[0:4,1],c='red',s=10,zorder=2)#c='tab:orange',
        
        plt.axis('off')#隐藏坐标系
        plt.axis("equal")
        
        plt.figure(dpi=300,figsize=(4,4.0))
        # plt.plot(self.GP[0:40,0],self.GP[0:40,1],linewidth=1,c='black',zorder=1)
        
        plt.scatter(self.Source[0:4,0],self.Source[0:4,1],s=10,c='tab:orange',zorder=3)#c='tab:blue',,
        plt.scatter(self.GP[0:40,0],self.GP[0:40,1],s=4,c='tab:blue',zorder=2)#c='tab:orange',
        font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 14}
        plt.rc('font',**font)
        plt.legend(["Source point","Integration point"],loc='upper left', bbox_to_anchor=(0.01,1.21))
        
        plt.axis('off')#隐藏坐标系
        plt.axis("equal")
    def variable_to_device(self,device):
        self.GPT = self.GPT.to(device)
        # self.GPTlog = self.GPTlog.to(device)
        self.SourceT = self.SourceT.to(device)
        # self.normT = self.normT.to(device)
        self.H = self.H.to(device)
        self.G = self.G .to(device)
        self.G_log =  self.G_log.to(device)
        self.C = self.C.to(device)
        self.b = self.b.to(device)
        self.fac = self.fac.to(device)
        self.fac_source = self.fac_source .to(device)
        self.dfac = self.dfac.to(device)
        # self.dfac_log =self.dfac_log.to(device)
        # self.normT_gp = self.normT_gp.to(device)
        # self.normT_loggp  =self.normT_loggp.to(device)
# def Generatepoints(x,y,b,h,func,NE = 100):
#      #先将原来边界划分为数个网格，每个网格上计算高斯积分点对应的:1.物理坐标 2. 外法线矢量 3.边界函数值 4.边界导数值 5.附上对应的高斯积分权重 6.雅可比
#      #选取数个源点，对每个源点计算基本解在积分点上的函数值与导数值，然后高斯积分求和计算互等定理的两个积分值，计算残差
#      #将残差累加，得到Loss函数
#      #对于给定的边界条件，积分应该采用给定值，但是是否需要约束使神经网络满足边界条件？
#      #
#      #
     
     
#      #rect = np.array([x,y,b,h])
#      lines  = np.zeros([4,2,2])
#      lines[0,:,:] = [[x-b/2,y-h/2],[x+b/2,y-h/2]]
#      lines[1,:,:] = [[x+b/2,y-h/2],[x+b/2,y+h/2]]
#      lines[2,:,:] = [[x+b/2,y+h/2],[x-b/2,y+h/2]]
#      lines[3,:,:] = [[x-b/2,y+h/2],[x-b/2,y-h/2]]
     
#      BCtype0 = [0,0,0,0];
#      # BCvalue = [np.ones([Ngauss,1]),np.zeros([Ngauss,1]),11*np.ones([Ngauss,1]),np.zeros([Ngauss,1])]
#      p0 = np.zeros([4*NE,2])
#      p1 = np.zeros([4*NE,2])
#      p0[0:NE,0]=np.linspace(lines[0,0,0],lines[0,1,0],NE,endpoint=False)
#      p0[NE:2*NE,0]=np.linspace(lines[1,0,0],lines[1,1,0],NE,endpoint=False)
#      p0[2*NE:3*NE,0]=np.linspace(lines[2,0,0],lines[2,1,0],NE,endpoint=False)
#      p0[3*NE:4*NE,0]=np.linspace(lines[3,0,0],lines[3,1,0],NE,endpoint=False)
     
#      p0[0:NE,1]=np.linspace(lines[0,0,1],lines[0,1,1],NE,endpoint=False)
#      p0[NE:2*NE,1]=np.linspace(lines[1,0,1],lines[1,1,1],NE,endpoint=False)
#      p0[2*NE:3*NE,1]=np.linspace(lines[2,0,1],lines[2,1,1],NE,endpoint=False)
#      p0[3*NE:4*NE,1]=np.linspace(lines[3,0,1],lines[3,1,1],NE,endpoint=False)
     
#      p1[0:4*NE-1,:] = p0[1:4*NE,:]
#      p1[4*NE-1,:] =p0[0,:]
#      BCtype = np.zeros([4*NE])
     
#      BCtype = np.repeat(BCtype0, NE)
         
#      Grids = arc_HB(p0, p1, func,BCtype)
     
                  
#      return Grids                

def fundamental(R,Norm,para=0):
    LR = np.linalg.norm(R,axis=-1)
    fs = -np.log(LR)/2/np.pi
    if para==0:
        dfs = -np.dot(R,Norm)/2/np.pi/LR/LR
    else:
        dfs = -(R[:,0]*Norm[:,0]+R[:,1]*Norm[:,1])/2/np.pi/LR/LR
    fs = torch.tensor(fs,dtype=torch.float32)
    dfs = torch.tensor(dfs,dtype=torch.float32)
    return fs,dfs

def D_fundamental(R,Norm,para=1):
    LR = np.linalg.norm(R,axis=-1)
    LR2 = LR*LR
    LR4=LR2*LR2
    
    
    Dfs = (R.T/LR/LR).T/2/np.pi
    if para==0:
        RiNi = np.dot(R,Norm)
        Ddfs = -(-Norm.reshape([-1,1])/LR2+2*RiNi/LR4*R.T).T/2/np.pi
    else:
        RiNi = R[:,0]*Norm[:,0]+R[:,1]*Norm[:,1]
        Ddfs = -(-Norm.T/LR2+2*RiNi/LR4*R.T).T/2/np.pi
    # Dfs = torch.tensor(Dfs,dtype=torch.float32)
    # Ddfs = torch.tensor(Ddfs,dtype=torch.float32)
    return Dfs,Ddfs


def fundamental_CPVLOG(R,Norm,i,weakjacobi,para=0):
    LR = np.linalg.norm(R,axis=-1)
    fs = -np.log(LR)/2/np.pi
    if para==0:
        dfs = -np.dot(R,Norm)/2/np.pi/LR/LR
    else:
        dfs = -(R[:,0]*Norm[:,0]+R[:,1]*Norm[:,1])/2/np.pi/LR/LR

    ind = np.arange(i*Ngauss,(i+1)*Ngauss)
    fs_log = np.log(LR[ind])/2/np.pi*weakjacobi
    
    fs = torch.tensor(fs,dtype=torch.float32)
    dfs = torch.tensor(dfs,dtype=torch.float32)
    fs_log = torch.tensor(fs_log,dtype=torch.float32)
    return fs,dfs,fs_log

def plot_points(x):
    
    plt.scatter(x[:,0],x[:,1],s=1,c='blue')
    # plt.xlim(xmax=50,xmin=-50)
    # plt.ylim(ymax=50,ymin=-50)
    plt.axis('off')#隐藏坐标系
    # font = {'family': 'serif',
    # 'serif': 'Times New Roman',
    # 'weight': 'normal',
    # 'size': 14}
    # plt.rc('font',**font)
    # plt.legend(["exact","prediction"],loc='upper left', bbox_to_anchor=(0.01,1.21))
    # plt.plot(rect[0,:],rect[1,:],c='black')
    plt.axis("equal")
# def fundamental_CPVLOG2(R,Norm,logL,i,para=0,E=1,v=0.3):
#     #这个算法加上了i，指代源点在第i个单元,logL是第i个单元的长度一半的对数ln(LE/2)
#     A1=-1/8/np.pi/(1-v)/(E/2/(1+v))
#     A2=3-4*v
#     B1 = -1/4/np.pi/(1-v)
#     B2 = 1-2*v  
#     #以下代码块是从fundamental（）直接复制而来，应该同步更新
#     LR = np.linalg.norm(R,axis=-1)
#     r = R/LR.reshape([-1,1])
   
    
#     fs = np.zeros([R.shape[0],4])
#     dfs = np.zeros([R.shape[0],4])
#     fs[:,0] = A1*(A2*np.log(LR)-r[:,0]*r[:,0])
#     fs[:,1] = A1*(-r[:,0]*r[:,1])
#     fs[:,2] = A1*(-r[:,1]*r[:,0])
#     fs[:,3] = A1*(A2*np.log(LR)-r[:,1]*r[:,1])#按照新的算法可以不对fs做处理
    
#     if para==0:
#         drdn = r[:,0]*Norm[0]+r[:,1]*Norm[1]
#         dfs[:,0] = B1/LR*((B2+2*r[:,0]*r[:,0])*drdn)
#         dfs[:,1] = B1/LR*((   2*r[:,0]*r[:,1])*drdn - B2*(r[:,0]*Norm[1]-r[:,1]*Norm[0]))
#         dfs[:,2] = B1/LR*((   2*r[:,0]*r[:,1])*drdn - B2*(r[:,1]*Norm[0]-r[:,0]*Norm[1]))
#         dfs[:,3] = B1/LR*((B2+2*r[:,1]*r[:,1])*drdn)
        
#     else:
#         drdn = r[:,0]*Norm[:,0]+r[:,1]*Norm[:,1]
#         dfs[:,0] = B1/LR*((B2+2*r[:,0]*r[:,0])*drdn)
#         dfs[:,1] = B1/LR*((   2*r[:,0]*r[:,1])*drdn - B2*(r[:,0]*Norm[:,1]-r[:,1]*Norm[:,0]))
#         dfs[:,2] = B1/LR*((   2*r[:,0]*r[:,1])*drdn - B2*(r[:,1]*Norm[:,0]-r[:,0]*Norm[:,1]))
#         dfs[:,3] = B1/LR*((B2+2*r[:,1]*r[:,1])*drdn)
#     # fs = torch.tensor(fs,dtype=torch.float32)
#     # dfs = torch.tensor(dfs,dtype=torch.float32)
#     #-----------------------------------------------
#     ind = np.arange(i*Ngauss,(i+1)*Ngauss)
#     dfs[ind,:] = 0
#     dfs[ind,1] = B1/LR[ind]*( - B2*(r[ind,0]*Norm[ind,1]-r[ind,1]*Norm[ind,0]))
#     dfs[ind,2] = B1/LR[ind]*( - B2*(r[ind,1]*Norm[ind,0]-r[ind,0]*Norm[ind,1]))
    
#     fs_log = -A1*(A2*np.log(LR[ind]))
#     fs = torch.tensor(fs,dtype=torch.float32)
#     dfs = torch.tensor(dfs,dtype=torch.float32)
#     fs_log = torch.tensor(fs_log,dtype=torch.float32)
#     return fs,dfs,fs_log