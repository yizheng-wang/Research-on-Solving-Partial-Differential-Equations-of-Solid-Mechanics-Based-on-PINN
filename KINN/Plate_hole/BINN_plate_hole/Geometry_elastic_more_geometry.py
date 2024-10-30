# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:35:10 2022

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
        plt.scatter(self.c_gp[:,0],self.c_gp[:,1])

class line_geometry:
    def __init__(self,NE,x0,x1):
        
        p1=np.linspace(x0,x1,NE+1,endpoint=True)
        p2=p1[1:]
        p1=p1[0:-1]
        rE = p2-p1
        LE = np.linalg.norm(rE,axis=1)
        self.DLE = LE[0]
        self.LE =LE
        self.source = (p2+p1)/2
        # self.GP = np.random.rand(p1.shape[0]).reshape(-1,1)*(p2-p1)+p1
        self.c_jacobi = self.DLE/2#jacobi to ksi
        
        t_LE = rE[0]
        self.c_gp = np.repeat(self.source,Ngauss,axis=0)+np.tile(np.reshape(Gausspoint,[-1,1])@np.reshape(t_LE/2,[1,-1]),[NE,1])
        
        +np.tile(Gausspoint*self.DLE/2,[NE,1])
        
        self.c_gpnorm = np.empty([NE*Ngauss,2])
        self.c_gpnorm[:,0] = rE[0,1]/self.DLE
        self.c_gpnorm[:,1] = -rE[0,0]/self.DLE
        
        self.source_norm = np.empty([NE,2])
        self.source_norm[:,0] = rE[0,1]/self.DLE
        self.source_norm[:,1] = -rE[0,0]/self.DLE
        
        self.string = self.DLE/2
        self.weak_jacobi = np.repeat(self.DLE/2,Ngauss)
        
        
        
class Any_HB():
    # def __init__(self,r,c,func,type0,NE,E=1,v=0.3):
    def __init__(self,geoset,func,E=1,v=0.3,device='cpu',pl_type = 'planestress'):
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
        self.pl_type = pl_type
        
        self.device = device
        self.Nset = len(geoset)
        self.E0=E
        self.E = 1
        self.v = v
        self.shear = self.E/2/(1+v)
        self.shear0 = self.E0/2/(1+v)
        #-------------------------------------------
        self.type = np.zeros([0,2],dtype=np.float64)#之后会reshape成一行
        self.GP = np.zeros([0,2],dtype=np.float64)
        self.norm = np.zeros([0,2],dtype=np.float64)
        self.jacobi = np.zeros([0])
        self.string = np.zeros([0])
        self.Source = np.zeros([0,2],dtype=np.float64)
        self.Sourcenorm = np.zeros([0,2],dtype=np.float64)
        self.weakjacobi = np.zeros([0,Ngauss])
        self.mode = np.zeros([0],dtype=int)
        self.NEALL = 0
        BC_f = np.zeros([0,2])
        BC_df = np.zeros([0,2])
        BC_f_source = np.zeros([0,2])
        BC_df_source = np.zeros([0,2])
        # self.testpoint  = np.zeros([0,2],dtype=np.float64)
        # self.testpoint_norm = np.zeros([0,2],dtype=np.float64)
        #-------------------------------------------------
        for geo_parameter in geoset:
            NE = geo_parameter['NE']
            if geo_parameter['mode']=='arc':
                self.mode = np.concatenate([self.mode,[1]],axis=0)
                # geo = arc_geometry(r[i],c[i,0],c[i,1],NE[i],p1=theta1[i],p2=theta2[i])
                geo = arc_geometry(geo_parameter['r'],geo_parameter['c'][0],geo_parameter['c'][1],\
                                   geo_parameter['NE'],p1=geo_parameter['theta1'],p2=geo_parameter['theta2'])
            elif geo_parameter['mode']=='line':
                self.mode = np.concatenate([self.mode,[0]],axis=0)
                geo = line_geometry(geo_parameter['NE'],geo_parameter['x0'],geo_parameter['x1'])
            self.NEALL+=geo_parameter['NE']
            self.GP = np.concatenate([self.GP,geo.c_gp],axis=0)
            self.norm = np.concatenate([self.norm,geo.c_gpnorm],axis=0)
            self.jacobi = np.concatenate([self.jacobi,np.repeat(geo.c_jacobi,NE*Ngauss)],axis=0)
            self.string = np.concatenate([self.string,np.repeat(geo.string,NE)],axis=0)
            self.Source = np.concatenate([self.Source,geo.source],axis=0)
            self.Sourcenorm = np.concatenate([self.Sourcenorm,geo.source_norm],axis=0)
            self.weakjacobi = np.concatenate([self.weakjacobi,np.tile(geo.weak_jacobi,[NE,1])],axis=0)
            self.type = np.concatenate([self.type,np.tile(geo_parameter['type'],[NE,1])],axis=0)
            BC_f = np.concatenate([BC_f,self.allocate_BC(geo_parameter['func'],geo.c_gp,norm=geo.c_gpnorm,mode=0)],axis=0)
            BC_df = np.concatenate([BC_df,self.allocate_BC(geo_parameter['func'],geo.c_gp,norm=geo.c_gpnorm,mode=1)],axis=0)
            BC_f_source = np.concatenate([BC_f_source,self.allocate_BC(geo_parameter['func'],geo.source,norm=geo.source_norm,mode=0)],axis=0)
            BC_df_source = np.concatenate([BC_df_source,self.allocate_BC(geo_parameter['func'],geo.source,norm=geo.source_norm,mode=1)],axis=0)
            
            
            
            # self.testpoint = np.concatenate([self.testpoint,geo.testpoint],axis=0)
            # self.testpoint_norm = np.concatenate([self.testpoint_norm,geo.testpoint_norm],axis=0)
        
        self.fac = torch.tensor(np.concatenate([BC_f[:,0],BC_f[:,1]]),dtype = torch.float32)
        self.fac_source = torch.tensor(np.concatenate([BC_f_source[:,0],BC_f_source[:,1]]),dtype = torch.float32)
        self.dfac = torch.tensor(np.concatenate([BC_df[:,0],BC_df[:,1]]),dtype = torch.float32)/self.E0
        self.dfac_source = torch.tensor(np.concatenate([BC_df_source[:,0],BC_df_source[:,1]]),dtype = torch.float32)/self.E0
        self.normT = torch.tensor(self.norm).float()
        self.normSource = torch.tensor(self.Sourcenorm).float()
        
        self.GPT = torch.tensor(self.GP,dtype = torch.float32,requires_grad=True)   
        self.SourceT = torch.tensor(self.Source,dtype = torch.float32,requires_grad=True)   
        self.weightT = torch.tensor(np.tile(Gaussweight,self.NEALL)*self.jacobi).float()
        self.type = np.reshape(self.type.T,[-1])        
        self.plot_points()
        #----------------------------------------------
        
        
        # self.func = func
        
        
        self.ulist = np.array(np.where(self.type==0))[0]#给定位移
        self.tlist = np.array(np.where(self.type==1))[0]#给定力
        self.Nulist = np.array(np.where(self.type!=0))[0]#未给定位移
        self.Ntlist = np.array(np.where(self.type!=1))[0]#未给定力
        # self.u_couple_list = np.array(np.where(self.type==2))[0]#耦合边界，本边界给定位移返回力
        # self.t_couple_list = np.array(np.where(self.type==-2))[0]#耦合边界，本边界给定力返回位移
        
        
        self.ucol_index = np.tile(np.array(range(Ngauss)),self.ulist.shape[0])
        self.ucol_index+=self.ulist.repeat(Ngauss,axis=0)*Ngauss
        self.tcol_index = np.tile(np.array(range(Ngauss)),self.tlist.shape[0])
        self.tcol_index+=self.tlist.repeat(Ngauss,axis=0)*Ngauss
        
        self.Nu_index = np.tile(np.array(range(Ngauss)),self.Nulist.shape[0])
        self.Nu_index+=self.Nulist.repeat(Ngauss,axis=0)*Ngauss
        self.Nt_index = np.tile(np.array(range(Ngauss)),self.Ntlist.shape[0])
        self.Nt_index+=self.Ntlist.repeat(Ngauss,axis=0)*Ngauss
        
        # self.u_couple_index = np.tile(np.array(range(Ngauss)),self.u_couple_list.shape[0])#耦合边界，本边界给定位移返回力
        # self.u_couple_index+= self.u_couple_list.repeat(Ngauss,axis=0)*Ngauss
        # self.t_couple_index = np.tile(np.array(range(Ngauss)),self.t_couple_list.shape[0])#耦合边界，本边界给定力返回位移
        # self.t_couple_index+= self.t_couple_list.repeat(Ngauss,axis=0)*Ngauss
        
        
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
            fs,dfs,fslog = fundamental_CPVLOG2(R,self.norm,i,self.weakjacobi[i],pl_type = self.pl_type,para=1,Eoriginal=self.E,voriginal=self.v)
            fs = fs*self.weightT.reshape([-1,1])
            dfs = dfs*self.weightT.reshape([-1,1])
            
            fslog = fslog*torch.tensor(Gaussweight).float()
            
            self.G_log[i] = torch.sum(fslog)
            
            self.G[i,:] = torch.cat([fs[:,0],fs[:,1]])
            self.G[Nrow//2+i,:] = torch.cat([fs[:,2],fs[:,3]])
            self.H[i,:] = torch.cat([dfs[:,0],dfs[:,1]])
            self.H[Nrow//2+i,:] = torch.cat([dfs[:,2],dfs[:,3]])
            
        self.G_log+=external_term(self.string,pl_type=self.pl_type,Eoriginal=self.E,voriginal=self.v,mode='arc')
        # self.b = torch.zeros([Nrow]).float()
        U_geo = self.C*self.fac_source.view(-1)

        
        T_log = torch.cat([self.G_log*self.dfac_source[0:self.Source.shape[0]],\
                          self.G_log*self.dfac_source[self.Source.shape[0]:2*self.Source.shape[0]]])
        D_work = torch.mv(self.H,self.fac.view(-1))+ U_geo\
        -torch.mv(self.G,self.dfac.view(-1)) - T_log
        U_geo[self.tlist] = 0.
        T_log[self.ulist] =0.

        
        self.b = torch.mv(self.H[:,self.ucol_index],self.fac[self.ucol_index].view(-1))+ U_geo\
        -torch.mv(self.G[:,self.tcol_index],self.dfac[self.tcol_index].view(-1)) - T_log
        self.b=-self.b#这里算b需要加负号，因为后面b2没加但是都默认H为正，为了用MSE必须让它们反号
    def allocate_BC(self,func,GP,norm=0,mode=0):
            y = func(GP,norm,mode,para=1,E=self.E,v=self.v,flaten=0)
            return y
            
    
    def update_func(self,Net):
        
        
        self.f,self.df = self.compute_traction(Net,self.GPT,self.normT)
        self.f_source,self.df_source = self.compute_traction(Net,self.SourceT,self.normSource)
        
    def compute_traction(self,Net,x,normT):
        #输入网络，坐标点，每个坐标点的法向
        #返回t，并且是t1，t2分开返回
        f = Net(x)
        gradient = torch.ones(f[:,0].size()).to(self.device)
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
        
        #--------------------------------------
        #给定了应力，是用另一侧的本构计算的，本算例中另一侧（夹杂）模量是基体的10倍
        # self.df[self.t_couple_index] = 10*self.df[self.t_couple_index]
        # self.df_source[self.t_couple_list] = 10*self.df_source[self.t_couple_list]
        
        #------------------------------------
        
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
        
        # D1 = b2-self.b
        # D1[self.ulist]=D1[self.ulist]/
        
        # loss = MSE(D1,torch.zeros_like(D1))
        loss = MSE(b2,self.b)
        return loss
       
    def inner(self,Net,x,mode=1):
        #内点积分方案
        self.update_func(Net)
        Known_f = self.f.clone().detach()
        Known_df = self.df.clone().detach()
        Known_f[self.ucol_index] = self.fac[self.ucol_index]
        Known_df[self.tcol_index] = self.dfac[self.tcol_index]
        
        
        #Known_df[self.t_couple_index] =10* Known_df[self.t_couple_index]
    
        Nrow = x.shape[0]*2
        Ncol = self.GP.shape[0]*2
        if mode==0:
            
            H_in = torch.zeros([Nrow,Ncol]).float().to(self.device)
            G_in = torch.zeros([Nrow,Ncol]).float().to(self.device)
            for i in range(x.shape[0]):
                R = self.GP-x[i]
                fs,dfs = fundamental(R,self.norm,pl_type=self.pl_type,para=1,Eoriginal=self.E,voriginal=self.v)
                fs = (fs*self.weightT.reshape([-1,1])).to(self.device)
                dfs = (dfs*self.weightT.reshape([-1,1])).to(self.device)
                G_in[i,:] = torch.cat([fs[:,0],fs[:,1]])
                G_in[Nrow//2+i,:] = torch.cat([fs[:,2],fs[:,3]])
                H_in[i,:] = torch.cat([dfs[:,0],dfs[:,1]])
                H_in[Nrow//2+i,:] = torch.cat([dfs[:,2],dfs[:,3]])
            U = -(torch.mv(H_in,Known_f)-torch.mv(G_in,Known_df))      
        elif mode==1:
            
            U = torch.zeros(Nrow).to(self.device)
            for i in range(Ncol//2):
                R = self.GP[i]-x
                fs,dfs = fundamental(R,self.norm[i],pl_type=self.pl_type,para=0,Eoriginal=self.E,voriginal=self.v)
                fs = (fs*self.weightT[i]).to(self.device)
                dfs = (dfs*self.weightT[i]).to(self.device)
                U += -(torch.cat([dfs[:,0],dfs[:,2]])*Known_f[i]
                       +torch.cat([dfs[:,1],dfs[:,3]])*Known_f[i+Ncol//2]
                       -torch.cat([fs[:,0],fs[:,2]])*Known_df[i]
                       -torch.cat([fs[:,1],fs[:,3]])*Known_df[i+Ncol//2])
                # U+= -(dfs*Known_f[i]-fs*Known_df[i])
        return U
    
    
    def inner_stress(self,Net,x,mode=1):
        #内点积分方案
        self.update_func(Net)
        Known_f = self.f.clone().detach()
        Known_df = self.df.clone().detach()*self.E0
        Known_f[self.ucol_index] = self.fac[self.ucol_index]
        Known_df[self.tcol_index] = self.dfac[self.tcol_index]*self.E0
        
        
        #Known_df[self.t_couple_index] =10* Known_df[self.t_couple_index]
        Nrow = x.shape[0]*4
        Ncol = self.GP.shape[0]*2
        if mode==0:
            
            H_in = torch.zeros([Nrow,Ncol]).float()
            G_in = torch.zeros([Nrow,Ncol]).float()
            for i in range(x.shape[0]):
                R = self.GP-x[i]
                usijk,psijk = fundamental_stress(R,self.norm,para=1,Eoriginal=self.E0,voriginal=self.v)
                usijk = usijk*self.weightT.reshape([-1,1])
                psijk = psijk*self.weightT.reshape([-1,1])
                G_in[i,:] = torch.cat([usijk[:,0],usijk[:,1]])
                G_in[Nrow//4+i,:] = torch.cat([usijk[:,2],usijk[:,3]])
                G_in[Nrow//2+i,:] = torch.cat([usijk[:,4],usijk[:,5]])
                G_in[3*Nrow//4+i,:] = torch.cat([usijk[:,6],usijk[:,7]])
                
                H_in[i,:] = torch.cat([psijk[:,0],psijk[:,1]])
                H_in[Nrow//4+i,:] = torch.cat([psijk[:,2],psijk[:,3]])
                H_in[Nrow//2+i,:] = torch.cat([psijk[:,4],psijk[:,5]])
                H_in[3*Nrow//4+i,:] = torch.cat([psijk[:,6],psijk[:,7]])
                
            Sigma = -(torch.mv(H_in,Known_f)-torch.mv(G_in,Known_df))  
        elif mode==1:
            
            Sigma = torch.zeros(Nrow).to(self.device)
            for i in range(Ncol//2):
                R = self.GP[i]-x
                usijk,psijk = fundamental_stress(R,self.norm[i],para=0,pl_type=self.pl_type,Eoriginal=self.E0,voriginal=self.v)
                usijk = (usijk*self.weightT[i]).to(self.device)
                psijk = (psijk*self.weightT[i]).to(self.device)
                Sigma += -(torch.cat([psijk[:,0],psijk[:,2],psijk[:,4],psijk[:,6]])*Known_f[i]
                       +torch.cat([psijk[:,1],psijk[:,3],psijk[:,5],psijk[:,7]])*Known_f[i+Ncol//2]
                       -torch.cat([usijk[:,0],usijk[:,2],usijk[:,4],usijk[:,6]])*Known_df[i]
                       -torch.cat([usijk[:,1],usijk[:,3],usijk[:,5],usijk[:,7]])*Known_df[i+Ncol//2])
                # U+= -(dfs*Known_f[i]-fs*Known_df[i])
        return Sigma
            
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
        
        # plt.axis('off')#隐藏坐标系
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
        
        # plt.axis('off')#隐藏坐标系
        plt.axis("equal")


    def plot_x0(self,Net,x):
        
        normT = torch.zeros_like(x).to(self.device)
        normT[:,0]=-1
        f = Net(x)
        gradient = torch.ones(f[:,0].size()).to(self.device)
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
        
        df = 2*self.shear0*self.v/(1-2*self.v)*(df1[:,0]+df2[:,1])*normT[:,0]+\
            self.shear0*(2*df1[:,0]*normT[:,0]+(df1[:,1]+df2[:,0])*normT[:,1])
        uy = f[:,1]
        s22 = self.E0*df2[:,1]-self.v*df
        s11 = -df
        mises = torch.sqrt(s11*s11-s11*s22+s22*s22)
        uy = uy.detach().cpu()
        mises = mises.detach().cpu()
        return uy,mises

    def plot_y0(self,Net,x):
        
        
        f = Net(x)
        ux = f[:,0]
        ux = ux.detach().cpu()
        return ux
    
    def variable_to_device(self,device):
        self.GPT = self.GPT.to(device)
        # self.GPTlog = self.GPTlog.to(device)
        self.SourceT = self.SourceT.to(device)
        self.normT = self.normT.to(device)
        self.normSource = self.normSource.to(device)
        self.H = self.H.to(device)
        self.G = self.G .to(device)
        self.G_log =  self.G_log.to(device)
        self.C = self.C.to(device)
        self.b = self.b.to(device)
        
        self.fac = self.fac.to(device)
        self.fac_source = self.fac_source .to(device)
        self.dfac = self.dfac.to(device)
        self.dfac_source =self.dfac_source.to(device)
        # self.normT_gp = self.normT_gp.to(device)
        # self.normT_loggp  =self.normT_loggp.to(device)
            
def fundamental_CPVLOG2(R,Norm,i,weakjacobi,pl_type,para=0,Eoriginal=1,voriginal=0.3):
    #这个算法加上了i，指代源点在第i个单元,logL是第i个单元的长度一半的对数ln(LE/2)
    if pl_type == 'planestress':
        v = voriginal/(1+voriginal)
        E = Eoriginal*(1-v*v)#注意这里v已经变了，位置不可调换
    else:
        v = voriginal
        E = Eoriginal
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
def external_term(string,pl_type,Eoriginal=1,voriginal=0.3,mode='arc'):
    #计算近奇异积分中加回来的解析部分
    if pl_type == 'planestress':
        v = voriginal/(1+voriginal)
        E = Eoriginal*(1-v*v)#注意这里v已经变了，位置不可调换
    else:
        v = voriginal
        E = Eoriginal

    A1=-1/8/np.pi/(1-v)/(E/2/(1+v))
    A2=3-4*v
    if mode=='arc':#这里没有对mode做区分，因为二者表达式是一样的
        
        ext_f = 2*A1*A2*np.abs(string)*(np.log(np.abs(string))-1)
    elif mode=='line':
        ext_f = 2*A1*A2*np.abs(string)*(np.log(np.abs(string))-1)
    ext_f=torch.tensor(ext_f).float()
    return ext_f
def fundamental(R,Norm,pl_type,para=0,Eoriginal=1,voriginal=0.3):
    if pl_type == 'planestress':
        v = voriginal/(1+voriginal)
        E = Eoriginal*(1-v*v)#注意这里v已经变了，位置不可调换
    else:
        v = voriginal
        E = Eoriginal
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

#----------------------------------------------------------
#下面是测试算例
def rect(x,y,b,h,func,BCtype,NE,E=1,v=0.3):
    p0 = np.array([x-b/2,y-h/2])
    p1 = np.array([x+b/2,y-h/2])
    p2 = np.array([x+b/2,y+h/2])
    p3 = np.array([x-b/2,y+h/2])
    
    geo1={'mode':'line','NE':NE,'x0':p0,'x1':p1,'type':BCtype[0],'func':func}
    geo2={'mode':'line','NE':NE,'x0':p1,'x1':p2,'type':BCtype[1],'func':func}
    geo3={'mode':'line','NE':NE,'x0':p2,'x1':p3,'type':BCtype[2],'func':func}
    geo4={'mode':'line','NE':NE,'x0':p3,'x1':p0,'type':BCtype[3],'func':func}
    geoset = [geo1,geo2,geo3,geo4]
    geo = Any_HB(geoset,func,E,v)
    return geo
 
def circle(x,y,r,func,BCtype,NE,E=1,v=0.3):
    
    c=np.array([x,y])
    geo1={'mode':'arc','NE':NE,'c':c,'r':r,'theta1':0,'theta2':2*np.pi,'type':BCtype[0],'func':func}
    
    geoset = [geo1]
    geo = Any_HB(geoset,func,E,v)
    return geo
   
def orifice(x,y,h,b,c,r,func,BCtype,NE,E=1,v=0.3):    
    p0 = np.array([x-b/2,y-h/2])
    p1 = np.array([x+b/2,y-h/2])
    p2 = np.array([x+b/2,y+h/2])
    p3 = np.array([x-b/2,y+h/2])
    
    geo1={'mode':'line','NE':NE,'x0':p0,'x1':p1,'type':BCtype[0],'func':func[0]}
    geo2={'mode':'line','NE':NE,'x0':p1,'x1':p2,'type':BCtype[1],'func':func[1]}
    geo3={'mode':'line','NE':NE,'x0':p2,'x1':p3,'type':BCtype[2],'func':func[2]}
    geo4={'mode':'line','NE':NE,'x0':p3,'x1':p0,'type':BCtype[3],'func':func[3]}
    #注意内孔部分圆弧方向是反的
    geo5={'mode':'arc','NE':NE,'c':c,'r':r,'theta1':2*np.pi,'theta2':0,'type':BCtype[4],'func':func[4]}
    
    geoset = [geo1,geo2,geo3,geo4,geo5]
    geo = Any_HB(geoset,func[0],E,v)
    return geo

def fundamental_stress(R,Norm,pl_type,para=0,Eoriginal=1,voriginal=0.3):
    if pl_type == 'planestress':
        v = voriginal/(1+voriginal)
        E = Eoriginal*(1-v*v)#注意这里v已经变了，位置不可调换
    else:
        v = voriginal
        E = Eoriginal
    G = E/2/(1+v)
    C1 = -1/(4*np.pi*(1-v))
    C2=1-2*v
    D1 = G/2/np.pi/(1-v)
    D2=(1-4*v)
    
    
    LR = np.linalg.norm(R,axis=-1)
    LR2=LR*LR
    r = R/LR.reshape([-1,1])
   
    
    
    uijk = np.zeros([R.shape[0],8])
    pijk = np.zeros([R.shape[0],8])
    uijk[:,0] = -C1/LR*(C2*(r[:,0])+2*r[:,0]*r[:,0]*r[:,0])
    uijk[:,1] = -C1/LR*(C2*(-r[:,1])+2*r[:,0]*r[:,0]*r[:,1])
    uijk[:,2] = -C1/LR*(C2*(r[:,1])+2*r[:,0]*r[:,1]*r[:,0])
    uijk[:,3] = -C1/LR*(C2*(r[:,0])+2*r[:,0]*r[:,1]*r[:,1])
    uijk[:,4] = -C1/LR*(C2*(r[:,1])+2*r[:,1]*r[:,0]*r[:,0])
    uijk[:,5] = -C1/LR*(C2*(r[:,0])+2*r[:,1]*r[:,0]*r[:,1])
    uijk[:,6] = -C1/LR*(C2*(-r[:,0])+2*r[:,1]*r[:,1]*r[:,0])
    uijk[:,7] = -C1/LR*(C2*(r[:,1])+2*r[:,1]*r[:,1]*r[:,1])
    
    
    nirjrk = np.zeros([R.shape[0],8])
    
    if para==0:
        nirjrk[:,0] = Norm[0]*r[:,0]*r[:,0]
        nirjrk[:,1] = Norm[0]*r[:,0]*r[:,1]
        nirjrk[:,2] = Norm[0]*r[:,1]*r[:,0]
        nirjrk[:,3] = Norm[0]*r[:,1]*r[:,1]
        nirjrk[:,4] = Norm[1]*r[:,0]*r[:,0]
        nirjrk[:,5] = Norm[1]*r[:,0]*r[:,1]
        nirjrk[:,6] = Norm[1]*r[:,1]*r[:,0]
        nirjrk[:,7] = Norm[1]*r[:,1]*r[:,1]
        
        drdn = r[:,0]*Norm[0]+r[:,1]*Norm[1]
        pijk[:,0] = D1/LR2*(2*drdn*(C2* r[:,0] +v*(2*r[:,0])-4*r[:,0]*r[:,0]*r[:,0])
                    +2*v*(2*nirjrk[:,0]) +           C2*(2*nirjrk[:,0] + 2*Norm[0]) -D2*(Norm[0]))
        pijk[:,1] = D1/LR2*(2*drdn*(C2* r[:,1]              -4*r[:,0]*r[:,0]*r[:,1]  )
                    +2*v*(2*nirjrk[:,1]) +           C2*(2*nirjrk[:,4])               -D2*(Norm[1]))
        pijk[:,2] = D1/LR2*(2*drdn*(            v*(r[:,1])  -4*r[:,0]*r[:,1]*r[:,0] )
                    +2*v*( nirjrk[:,2]+nirjrk[:,4]) +C2*(2*nirjrk[:,1]+Norm[1]) )
        pijk[:,3] = D1/LR2*(2*drdn*(            v*(r[:,0])  -4*r[:,0]*r[:,1]*r[:,1] )
                    +2*v*( nirjrk[:,3]+nirjrk[:,5]) +C2*(2*nirjrk[:,5]+Norm[0]) )
        pijk[:,4] = D1/LR2*(2*drdn*(            v*(r[:,1])  -4*r[:,1]*r[:,0]*r[:,0] )
                    +2*v*( nirjrk[:,4]+nirjrk[:,2]) +C2*(2*nirjrk[:,2]+Norm[1]) )
        pijk[:,5] = D1/LR2*(2*drdn*(            v*(r[:,0])  -4*r[:,1]*r[:,0]*r[:,1] )
                    +2*v*( nirjrk[:,5]+nirjrk[:,3]) +C2*(2*nirjrk[:,6]+Norm[0]) )
        pijk[:,6] = D1/LR2*(2*drdn*(C2* r[:,0]              -4*r[:,1]*r[:,1]*r[:,0] )
                    +2*v*(2*nirjrk[:,6]) +           C2*(2*nirjrk[:,3])               -D2*(Norm[0]))
        pijk[:,7] = D1/LR2*(2*drdn*(C2* r[:,1] +v*(2*r[:,1])-4*r[:,1]*r[:,1]*r[:,1] )
                    +2*v*(2*nirjrk[:,7]) +           C2*(2*nirjrk[:,7]+ 2*Norm[1]) -D2*(Norm[1]))
        
        
    else:
        drdn = r[:,0]*Norm[:,0]+r[:,1]*Norm[:,1]
        nirjrk[:,0] = Norm[:,0]*r[:,0]*r[:,0]
        nirjrk[:,1] = Norm[:,0]*r[:,0]*r[:,1]
        nirjrk[:,2] = Norm[:,0]*r[:,1]*r[:,0]
        nirjrk[:,3] = Norm[:,0]*r[:,1]*r[:,1]
        nirjrk[:,4] = Norm[:,1]*r[:,0]*r[:,0]
        nirjrk[:,5] = Norm[:,1]*r[:,0]*r[:,1]
        nirjrk[:,6] = Norm[:,1]*r[:,1]*r[:,0]
        nirjrk[:,7] = Norm[:,1]*r[:,1]*r[:,1]

        pijk[:,0] = D1/LR2*(2*drdn*(C2* r[:,0] +v*(2*r[:,0])-4*r[:,0]*r[:,0]*r[:,0])
                    +2*v*(2*nirjrk[:,0]) +           C2*(2*nirjrk[:,0] + 2*Norm[:,0]) -D2*(Norm[:,0]))
        pijk[:,1] = D1/LR2*(2*drdn*(C2* r[:,1]              -4*r[:,0]*r[:,0]*r[:,1]  )
                    +2*v*(2*nirjrk[:,1]) +           C2*(2*nirjrk[:,4])               -D2*(Norm[:,1]))
        pijk[:,2] = D1/LR2*(2*drdn*(            v*(r[:,1])  -4*r[:,0]*r[:,1]*r[:,0] )
                    +2*v*( nirjrk[:,2]+nirjrk[:,4]) +C2*(2*nirjrk[:,1]+Norm[:,1]) )
        pijk[:,3] = D1/LR2*(2*drdn*(            v*(r[:,0])  -4*r[:,0]*r[:,1]*r[:,1] )
                    +2*v*( nirjrk[:,3]+nirjrk[:,5]) +C2*(2*nirjrk[:,5]+Norm[:,0]) )
        pijk[:,4] = D1/LR2*(2*drdn*(            v*(r[:,1])  -4*r[:,1]*r[:,0]*r[:,0] )
                    +2*v*( nirjrk[:,4]+nirjrk[:,2]) +C2*(2*nirjrk[:,2]+Norm[:,1]) )
        pijk[:,5] = D1/LR2*(2*drdn*(            v*(r[:,0])  -4*r[:,1]*r[:,0]*r[:,1] )
                    +2*v*( nirjrk[:,5]+nirjrk[:,3]) +C2*(2*nirjrk[:,6]+Norm[:,0]) )
        pijk[:,6] = D1/LR2*(2*drdn*(C2* r[:,0]              -4*r[:,1]*r[:,1]*r[:,0] )
                    +2*v*(2*nirjrk[:,6]) +           C2*(2*nirjrk[:,3])               -D2*(Norm[:,0]))
        pijk[:,7] = D1/LR2*(2*drdn*(C2* r[:,1] +v*(2*r[:,1])-4*r[:,1]*r[:,1]*r[:,1] )
                    +2*v*(2*nirjrk[:,7]) +           C2*(2*nirjrk[:,7]+ 2*Norm[:,1]) -D2*(Norm[:,1]))
        
    uijk = torch.tensor(uijk,dtype=torch.float32)
    pijk = torch.tensor(pijk,dtype=torch.float32)
    return uijk,pijk    

def export_arcpoints(R0,NE=1000):
    
    x1,norm1 = circle1(R0[0],R0[1],R0[2]-0.001,2*NE)
    x2,norm2 = circle1(R0[0],R0[1],R0[2]+0.001,2*NE)
    
    np.save('arc_point1.npy',np.concatenate([x1,np.zeros([x1.shape[0],1])],axis=1))
    
    
    
    np.save('arc_point2.npy',np.concatenate([x2,np.zeros([x2.shape[0],1])],axis=1))

def draw_inclusion(Grids0,Grids1,Net,rect0,R0,NE=1000):
    
    
    x1,norm1 = rect1(rect0[0],rect0[1],rect0[2],rect0[3],NE)
    x2,norm2 = circle1(R0[0],R0[1],R0[2],2*NE)
    plt.figure()
    plt.scatter(x1[:,0],x1[:,1])
    plt.scatter(x2[:,0],x2[:,1])
    
    x = np.concatenate([x1,x2],axis = 0)
    norm = np.concatenate([norm1,norm2],axis = 0)
    np.save('ori_point.npy',np.concatenate([x,np.zeros([x.shape[0],1])],axis=1))
    xT = torch.tensor(x,requires_grad=True).float()
    normT = torch.tensor(norm).float()
    
    
    u,f = Grids0.compute_traction(Net,xT,normT)
    u = u.detach().numpy().reshape((2,x.shape[0])).T
    f = f.detach().numpy().reshape((2,x.shape[0])).T
    
    u1,f1 = Grids1.compute_traction(Net,xT,normT)
    
    # plt.legend (loc='lower right', fontsize=40)
    

    
    # plt.title("int_p = "+str(Grids0.H.size()[1])+", S_p = "+str(Grids0.H.size()[0])+", epoch ="+str(epoch+1)+", err = "+str(err))
    '''
    plt.plot(exact[0:NNODE])
    plt.plot(predict[0:NNODE],ls='--')
    plt.plot(exact[NNODE:],c='black')
    plt.plot(predict[NNODE:],ls='--',c='red')
    '''
    ind_natural = np.arange(0,3*NE)
    ind_essential = np.arange(3*NE,4*NE)
    ind_inter = 4*NE+NE
    ind_interface = np.arange(ind_inter,ind_inter+2*NE)
    #------------------------------------
    #读数据
    uac1 = np.load('inclusion_u1_BC.npy')
    uac2 = np.load('inclusion_u2_BC.npy')
    rfac1 = np.load('inclusion_BC_RF1.npy')
    rfac2 = np.load('inclusion_BC_RF2.npy')


    # 修改这一部分，abaqus输出的数据出现了一些重复的点，对这些重复的点进行
    # 去重求平均
    
    def quchong(u):
        m_index=[]
        ubar=[]
        i=0
        while i <u.shape[0]:
            if i<u.shape[0]-1:
                if abs(u[i,0]-u[i+1,0])<1e-8:
                    m_index.append(i)
                    ubar.append(min(u[i,1],u[i+1,1]))
                    i+=1
                else:
                    ubar.append(u[i,1])
            else:
                ubar.append(u[i,1])
            i+=1
        
        return np.array(ubar)
    uac1 = quchong(uac1)
    uac2 = quchong(uac2)
    
#从ABAQUS导出的界面力在极坐标系下，这里转到笛卡尔坐标系
    interface_tx,interface_ty = read_interface(norm2)
    
    
    rfac1 = quchong(rfac1)
    rfac2 = quchong(rfac2)
    
    
    uac = np.array([uac1,uac2]).T
    
    fac = np.zeros(f.shape)
    
    fac[ind_essential,0] = -rfac1[ind_essential]
    fac[ind_essential,1] = -rfac2[ind_essential]
    fac[ind_interface-NE,0] = interface_tx
    fac[ind_interface-NE,1] = interface_ty
    #-----------------------------------
    plt.figure(dpi=400,figsize=[6,4])
    plt.ylim(ymin=-2,ymax=6.5)
    ax1=plt.gca()
    x_locator = plt.MultipleLocator(NE)
    plt.xticks([0,1000,2000,3000])
    ax1.plot(ind_natural,uac[0:3*NE,0])
    ax1.plot(ind_natural,u[0:3*NE,0],ls='--')
    ax1.plot(ind_natural,uac[0:3*NE,1],c='black')
    ax1.plot(ind_natural,u[0:3*NE,1],ls='--',c='red')
    plt.legend(["u$_1$,FEM","u$_1$,BINN","u$_2$,FEM","u$_2$,BINN"],loc='upper left',ncol=2,columnspacing=0.4)
    plt.xlabel('trajectory')
    
    plt.figure(dpi=400,figsize=[4.5,4])
    plt.ylim(ymin=-3.4,ymax=1.5)
    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(x_locator)
    ax2.plot(ind_essential,fac[3*NE:4*NE,0])
    ax2.plot(ind_essential,f[3*NE:4*NE,0],ls='--')
    ax2.plot(ind_essential,fac[3*NE:4*NE,1],c='black')
    ax2.plot(ind_essential,f[3*NE:4*NE,1],ls='--',c='red')
    plt.legend(["t$_1$,FEM","t$_1$,BINN","t$_2$,FEM","t$_2$,BINN"],loc='upper right',ncol=2,columnspacing=0.4)
    plt.xlabel('trajectory')
    
#------------------------------------------------    
    plt.figure(dpi=400,figsize=[4.0,4])
    # plt.ylim(ymin=-3.8,ymax=5)
    ax1=plt.gca()
    ax1.set_ylim(ymin=-0.2,ymax=2.5)
    plt.xticks([5000,6000,7000])
    ax1.plot(ind_interface,uac[4*NE:,0])
    ax1.plot(ind_interface,u[4*NE:,0],ls='--')
    ax1.plot(ind_interface,uac[4*NE:,1],c='black')
    ax1.plot(ind_interface,u[4*NE:,1],ls='--',c='red')
    # plt.legend(["g$_1$,FEM","g$_1$,BINN","g$_2$,FEM","g$_2$,BINN"],loc='upper left',ncol=2)
    plt.legend(["u$_1$,FEM","u$_1$,BINN","u$_2$,FEM","u$_2$,BINN"],loc='upper left',ncol=2,columnspacing=0.4)
    plt.xlabel('trajectory')
    
    plt.figure(dpi=400,figsize=[4.5,4])
    ax2 = plt.gca()
    
    ax2.set_ylim(ymin=-1.4,ymax=2.5)
    plt.xticks([5000,6000,7000])
    # ax2.plot(ind_interface,fac[4*NE:,0], c='green' )
    # ax2.plot(ind_interface,f[4*NE:,0]*10,c='magenta',ls='--')
    # ax2.plot(ind_interface,fac[4*NE:,1],c='navy')
    # ax2.plot(ind_interface,f[4*NE:,1]*10,ls='--',c='gold')
    ax2.plot(ind_interface,fac[4*NE:,0] )
    ax2.plot(ind_interface,f[4*NE:,0]*10,ls='--')
    ax2.plot(ind_interface,fac[4*NE:,1],c='black')
    ax2.plot(ind_interface,f[4*NE:,1]*10,ls='--',c='red')
    plt.legend(["t$_1$,FEM","t$_1$,BINN","t$_2$,FEM","t$_2$,BINN"],loc='upper left',ncol=2,columnspacing=0.4)
    
    font = {'family': 'serif',
    'serif': 'Times New Roman',
    'weight': 'normal',
    'size': 14}
    plt.rc('font',**font)
    # plt.ylim(ymax=8)
    # plt.xlim(xmax=4300)
    plt.xlabel('trajectory')
    # plt.ylabel('u')
    
    
#---------------------------------------error    
    plt.figure(dpi=400)
    ax1=plt.gca()
    ax1.set_ylim(ymin=-0.08,ymax=0.05)
    plt.xticks([0,1000,2000,3000])
    
    # plt.title("int_p = "+str(Grids0.H.size()[1])+", S_p = "+str(Grids0.H.size()[0])+", epoch ="+str(epoch+1)+", err = "+str(err))
    ax1.plot(uac[0:3*NE,0]-u[0:3*NE,0])
    ax1.plot(uac[0:3*NE,1]-u[0:3*NE,1])
    plt.legend(["error of u$_1$","error of u$_2$"],loc='upper left',ncol=1,columnspacing=0.4)
    plt.xlabel('trajectory')
    
    plt.figure(dpi=400,figsize=[4.5,4])
    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(x_locator)
    ax2.set_ylim(ymin=-0.06,ymax=0.05)
    ax2.plot(ind_essential,fac[3*NE:4*NE,0]-f[3*NE:4*NE,0])
    ax2.plot(ind_essential,fac[3*NE:4*NE,1]-f[3*NE:4*NE,1])
    plt.legend(["error of t$_1$","error of t$_2$"],loc='upper left',ncol=1,columnspacing=0.4,bbox_to_anchor=[0.1,0,1,1])
    plt.xlabel('trajectory')
    #------------------
    plt.figure(dpi=400,figsize=[4.5,4])
    ax1=plt.gca()
    ax1.set_ylim(ymin=-0.0013,ymax=0.0025)
    x_locator = plt.MultipleLocator(NE)
    ax1.xaxis.set_major_locator(x_locator)
    
    ax1.plot(ind_interface,uac[4*NE:,0]-u[4*NE:,0])
    ax1.plot(ind_interface,uac[4*NE:,1]-u[4*NE:,1])
    plt.xlabel('trajectory')
    plt.legend(["error of u$_1$","error of u$_2$"],loc='upper left',ncol=1,columnspacing=0.4)
    
    plt.figure(dpi=400,figsize=[4.5,4])
    ax2 = plt.gca()
    ax2.set_ylim(ymin=-0.0062,ymax=0.0078)
    ax2.xaxis.set_major_locator(x_locator)
    ax2.plot(ind_interface,fac[4*NE:,0]-f[4*NE:,0]*10)
    ax2.plot(ind_interface,fac[4*NE:,1]-f[4*NE:,1]*10)
    # plt.legend(["error of u$_1$","error of u$_2$","error of t$_1$","error of t$_2$"],loc='upper left',ncol=1)
    plt.legend(["error of t$_1$","error of t$_2$"],loc='upper left',ncol=1,columnspacing=0.4)
   
    plt.xlabel('trajectory')
    plt.show()   

def rect1(x,y,b,h,NE):
     lines  = np.zeros([4,2,2])
     lines[0,:,:] = [[x-b/2,y-h/2],[x+b/2,y-h/2]]
     lines[1,:,:] = [[x+b/2,y-h/2],[x+b/2,y+h/2]]
     lines[2,:,:] = [[x+b/2,y+h/2],[x-b/2,y+h/2]]
     lines[3,:,:] = [[x-b/2,y+h/2],[x-b/2,y-h/2]]
     
     # BCtype0 = [1,1,1,1,1,1,1,1];
     # BCtype0 = [0,1,1,1,0,1,1,1];
     # BCvalue = [np.ones([Ngauss,1]),np.zeros([Ngauss,1]),11*np.ones([Ngauss,1]),np.zeros([Ngauss,1])]
     p0 = np.zeros([4*NE,2])
     # p1 = np.zeros([4*NE,2])
     p0[0:NE,0]=np.linspace(lines[0,0,0],lines[0,1,0],NE+1,endpoint=False)[1:]
     p0[NE:2*NE,0]=np.linspace(lines[1,0,0],lines[1,1,0],NE+1,endpoint=False)[1:]
     p0[2*NE:3*NE,0]=np.linspace(lines[2,0,0],lines[2,1,0],NE+1,endpoint=False)[1:]
     p0[3*NE:4*NE,0]=np.linspace(lines[3,0,0],lines[3,1,0],NE+1,endpoint=False)[1:]
     
     p0[0:NE,1]=np.linspace(lines[0,0,1],lines[0,1,1],NE+1,endpoint=False)[1:]
     p0[NE:2*NE,1]=np.linspace(lines[1,0,1],lines[1,1,1],NE+1,endpoint=False)[1:]
     p0[2*NE:3*NE,1]=np.linspace(lines[2,0,1],lines[2,1,1],NE+1,endpoint=False)[1:]
     p0[3*NE:4*NE,1]=np.linspace(lines[3,0,1],lines[3,1,1],NE+1,endpoint=False)[1:]
     # p0[-1,:] = p0[0,:]
     # p1[0:4*NE-1,:] = p0[1:4*NE,:]
     # p1[4*NE-1,:] =p0[0,:]
     # rE = p0[1:]-p0[0:-1]
     # LE = np.linalg.norm(rE,axis=1)
     norm = np.empty(p0.shape)
     norm[0:NE,0] = 0
     norm[0:NE,1] = -1
     norm[NE:2*NE,0] = 1
     norm[NE:2*NE,1] = 0
     norm[2*NE:3*NE,0] = 0
     norm[2*NE:3*NE,1] = 1
     norm[3*NE:4*NE,0] = -1
     norm[3*NE:4*NE,1] = 0
     return p0,norm
 
def circle1(x,y,r,NE):
    p2 = -2*np.pi
    p1 = 0
    theta = np.linspace(p1,p2,NE+1,endpoint=True)
    sourcetheta = (theta[0:-1]+theta[1:NE+1])/2
    
    source = np.zeros([NE,2])
    source[:,0] = r*np.cos(sourcetheta)+x
    source[:,1] = r*np.sin(sourcetheta)+y
    source_norm = np.array([np.cos(sourcetheta),np.sin(sourcetheta)]).T*np.sign(p2-p1)
    return source,source_norm

def read_interface(norm2):
    def quchong(u):
        m_index=[]
        ubar=[]
        i=0
        while i <u.shape[0]:
            if i<u.shape[0]-1:
                if abs(u[i,0]-u[i+1,0])<1e-8:
                    m_index.append(i)
                    ubar.append(min(u[i,1],u[i+1,1]))
                    i+=1
                else:
                    ubar.append(u[i,1])
            else:
                ubar.append(u[i,1])
            i+=1
        
        return np.array(ubar)
    iS11_inner = np.load('inclusion_BC_IS11_inner.npy')
    iS22_inner = np.load('inclusion_BC_IS22_inner.npy')
    iS12_inner = np.load('inclusion_BC_IS12_inner.npy')
    iS11_outer = np.load('inclusion_BC_IS11_outer.npy')
    iS22_outer = np.load('inclusion_BC_IS22_outer.npy')
    iS12_outer = np.load('inclusion_BC_IS12_outer.npy')

    iS11_inner = quchong(iS11_inner)
    iS22_inner = quchong(iS22_inner)
    iS12_inner = quchong(iS12_inner)
    iS11_outer = quchong(iS11_outer)
    iS22_outer = quchong(iS22_outer)
    iS12_outer = quchong(iS12_outer)    
    
    IF1_inner = iS11_inner*norm2[:,0]+iS12_inner*norm2[:,1]
    IF2_inner = iS12_inner*norm2[:,0]+iS22_inner*norm2[:,1]
    IF1_outer = iS11_outer*norm2[:,0]+iS12_outer*norm2[:,1]
    IF2_outer = iS12_outer*norm2[:,0]+iS22_outer*norm2[:,1]
    plt.figure()
    plt.plot(IF1_inner)
    plt.plot(IF1_outer)
    plt.plot((IF1_outer+IF1_inner)/2)
    plt.figure()
    plt.plot(IF2_inner)
    plt.plot(IF2_outer)
    plt.plot((IF2_outer+IF2_inner)/2)
    return (IF1_outer+IF1_inner)/2,(IF2_outer+IF2_inner)/2
    
    
    
    