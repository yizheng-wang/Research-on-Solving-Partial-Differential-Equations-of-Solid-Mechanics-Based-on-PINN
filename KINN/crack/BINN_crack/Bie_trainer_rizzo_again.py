# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:24:29 2022

@author: yludragon
"""
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
MSE = torch.nn.MSELoss(reduction='mean')
def Batch_loader(N1,batchsize):
    ListN=list(range(N1))
    random.shuffle(ListN)
    ii = 0
    i=0
    '''
    '''
    while i+batchsize<N1:
        Ibatch=ListN[i:i+batchsize]
        i=i+batchsize
        
def Batch_BC(N1,Net1,Grids1,B1,optimizer1):
    ListN=list(range(N1))
    random.shuffle(ListN)
    i=0
    '''
    '''
    while i+B1<N1+1:
        loss_BC=0
        Ibatch=ListN[i:i+B1]
        i=i+B1
        optimizer1.zero_grad()
        for J in Ibatch:                
            Grids1[J].update_func(Net1)
            loss_BC +=Grids1[J].BC_loss()
    
        loss_BC/=B1
        
        loss_BC.backward()
        optimizer1.step()
    return loss_BC

def Batch_BIE(N1,NG1,Net1,Source1,Grids1,B1,optimizer1):
    ListN=list(range(N1))
    random.shuffle(ListN)
    i=0
    while i+B1<=N1+1:
        loss_BIE_M = torch.zeros([B1],dtype = torch.float32)
        Ibatch=ListN[i:i+B1]
        i=i+B1
        optimizer1.zero_grad()
       
        for J in range(NG1):                
            Grids1[J].update_func(Net1)
            
        ii=0    
        for J in Ibatch:
            for K in range(NG1):
                Grids1[K].update_fund(Source1[J])
                SF,SDF = Grids1[K].integral_func()              
                loss_BIE_M[ii]+=SF-SDF
            ii+=1
                
        loss_BIE=torch.dot(loss_BIE_M,loss_BIE_M)
        loss_BIE/=B1
        loss_BIE.backward()
        optimizer1.step()
    return loss_BIE

def Batch_BIE_rizzo(NG1,Net1,Grids1,B1,optimizer1):
    ListN=list(range(NG1))
    random.shuffle(ListN)
    i=0
    while i+B1<=NG1+1:
        loss_BIE_M = torch.zeros([B1],dtype = torch.float32)
        Ibatch=ListN[i:i+B1]
        i=i+B1
        optimizer1.zero_grad()
       
        for J in range(NG1):                
            Grids1[J].update_func(Net1)
            
        ii=0    
        for J in Ibatch:
            for K in range(NG1):
                if K!=J:
                    Grids1[K].update_fund(Grids1[J].Source)
                    SF,SDF= Grids1[K].integral_func()  
                else:
                    # Grids0[K].update_fund(Grids0[J].Source)#非常重要，但是为什么？这句话没起作用啊
                    SF,SDF = Grids1[K].integral_func_single(Net1)  
                loss_BIE_M[ii]+=SF-SDF
            ii+=1
                
        loss_BIE=torch.dot(loss_BIE_M,loss_BIE_M)
        loss_BIE/=B1
        loss_BIE.backward()
        optimizer1.step()
    return loss_BIE

def Batch_BIE_rizzo_HB(NG1,H,G,Net1,Grids1,B1,optimizer1,Ngauss):
      MSE = torch.nn.MSELoss()
      DOF = H.size()[1]
      ListN=list(range(NG1))
      random.shuffle(ListN)
      i=0
      while i+B1<=NG1+1:
          intdf = 0
          df = 0
          df = torch.zeros([DOF],dtype = torch.float32)
          f = torch.zeros([DOF],dtype = torch.float32)
          intCPV = torch.zeros([NG1],dtype = torch.float32)
          intlog = torch.zeros([NG1],dtype = torch.float32)
          # loss_BIE_M = torch.zeros([B1],dtype = torch.float32)
          Ibatch=ListN[i:i+B1]
          i=i+B1
          optimizer1.zero_grad()
          reg=0
          for J in range(NG1):                
              Grids1[J].update_func(Net1)
              intlog[J],intCPV[J]=Grids1[J].return_func_single(Net1)
              
              
              if Grids1[J].type==0:
                  f[J*Ngauss:(J+1)*(Ngauss)] = Grids1[J].value.view(-1)
                  df[J*Ngauss:(J+1)*(Ngauss)] = Grids1[J].df.view(-1)
                  reg+=torch.var(Grids1[J].df.view(-1))
              elif Grids1[J].type==1:
                  f[J*Ngauss:(J+1)*(Ngauss)] = Grids1[J].f.view(-1)
                  df[J*Ngauss:(J+1)*(Ngauss)] = Grids1[J].value.view(-1)
                  reg+=torch.var(Grids1[J].f.view(-1))
          intdf=torch.mv(H[Ibatch,:],f)+intCPV[Ibatch]
          intf=torch.mv(G[Ibatch,:],df)+intlog[Ibatch]
          
          # for J in Ibatch:
          #     for K in range(NG1):
          #         if K!=J:
          #             Grids1[K].update_fund(Grids1[J].Source)
          #             SF,SDF= Grids1[K].integral_func()  
          #         else:
          #             # Grids0[K].update_fund(Grids0[J].Source)#非常重要，但是为什么？这句话没起作用啊
          #             SF,SDF = Grids1[K].integral_func_single(Net1)  
          #         loss_BIE_M[ii]+=SF-SDF
          #     ii+=1
                  
          loss_BIE=MSE(intf,intdf)+reg/NG1
         
          loss_BIE.backward()
          optimizer1.step()
      return loss_BIE
  
def Batch_BIE_NO_rizzo(NG1,Net1,Grids1,B1,optimizer1)  :
    MSE = torch.nn.MSELoss()
   
    ListN=list(range(NG1))
    random.shuffle(ListN)
    i=0
    while i+B1<=NG1+1:
        intdf = 0
        df = 0
        # loss_BIE_M = torch.zeros([B1],dtype = torch.float32)
        Ibatch=ListN[i:i+B1]
        i=i+B1
        optimizer1.zero_grad()
        # reg=0
        Grids1.update_func(Net1)
        intf,intdf = Grids1.integral_func(Ibatch)
        loss_BIE=MSE(intf,intdf)
       
        loss_BIE.backward()
        optimizer1.step()
    return loss_BIE


def Batch_full_rizzo(NG1,Net1,Grids1,B1,optimizer1):
    ListN=list(range(NG1))
    random.shuffle(ListN)
    i=0
    while i+B1<=NG1+1:
        loss_BIE_M = torch.zeros([B1],dtype = torch.float32)
        Ibatch=ListN[i:i+B1]
        i=i+B1
        optimizer1.zero_grad()
       
        for J in range(NG1):                
            Grids1[J].update_func(Net1)
            
        ii=0    
        for J in Ibatch:
            for K in range(NG1):
                if K!=J:
                    Grids1[K].update_fund(Grids1[J].Source)
                    SF,SDF= Grids1[K].integral_func()  
                else:
                    # Grids0[K].update_fund(Grids0[J].Source)#非常重要，但是为什么？这句话没起作用啊
                    SF,SDF = Grids1[K].integral_func_single(Net1)  
                loss_BIE_M[ii]+=SF-SDF
            ii+=1
                
        loss_BIE=torch.dot(loss_BIE_M,loss_BIE_M)
        loss_BIE/=B1
        loss_BIE.backward()
        optimizer1.step()
    return loss_BIE

def Batch_BIE_Node(N1,Net1,Source1,Grids1,B1,optimizer1):
    ListN=list(range(N1))
    random.shuffle(ListN)
    i=0
    while i+B1<=N1+1:
        # loss_BIE_M = torch.zeros([B1],dtype = torch.float32)
        loss_BIE = 0
        Ibatch=ListN[i:i+B1]
        i=i+B1
        optimizer1.zero_grad()       
        Grids1.update_func(Net1)
        loss_BIE = Grids1.BIE_loss_batch(Ibatch)      
        # for J in Ibatch:           
            # Grids1.update_fund(Source1[J])
            # loss_BIE += (Grids1.BIE_loss())**2   
            
        # loss_BIE/=B1
        loss_BIE.backward()
        optimizer1.step()
    return loss_BIE

def err_BIE_Node(NS,Net1,Source1,Grids1):
    Grids1.update_func(Net1)
    loss_BIE = torch.zeros(NS)
    for J in range(NS):
        Grids1.update_fund(Source1[J])
        loss_BIE[J] = Grids1.BIE_loss()
    return loss_BIE  

def Batch_BIEBC(N1,NG1,Net1,Source1,Grids1,B1,optimizer1):
    ListN=list(range(N1))
    random.shuffle(ListN)
    i=0
    while i+B1<=N1+1:
        loss_BIE_M = torch.zeros([B1],dtype = torch.float32)
        Ibatch=ListN[i:i+B1]
        i=i+B1
        optimizer1.zero_grad()
        loss_BC = 0

        for J in range(NG1):                
            Grids1[J].update_func(Net1)
            loss_BC +=Grids1[J].BC_loss()
        
        ii=0    
        for J in Ibatch:
            for K in range(NG1):
                Grids1[K].update_fund(Source1[J])
                SF,SDF = Grids1[K].integral_func()              
                loss_BIE_M[ii]+=SF-SDF
            ii+=1
                
        loss_BIE=torch.dot(loss_BIE_M,loss_BIE_M)
        loss_BIE/=B1
        loss_BC/=NG1
        loss = loss_BIE+loss_BC
        loss.backward()
        optimizer1.step()
    return loss

#完整的损失函数
def train_full(Source,Grids0,Net1,device,workspace,Nepoch,lr0,Netinvname):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//5), gamma=0.2)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    ap=1.0
    NG = len(Grids0)
    NS = Source.shape[0]
    Net1.train(mode=True)
    Nbatch = NS
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
                
    for epoch in range(Nepoch):
        loss_BIE_M = torch.zeros([Nbatch],dtype = torch.float32)
        loss_BC = 0
        loss = 0
        optimizer.zero_grad()
        for J in range(NG):                
            Grids0[J].update_func(Net1)
            loss_BC += Grids0[J].BC_loss()
        for J in range(NS):
            for K in range(NG):
                Grids0[K].update_fund(Source[J])
                SF,SDF = Grids0[K].integral_func()              
                loss_BIE_M[J]+=SF-SDF
                
        loss_BIE=torch.dot(loss_BIE_M,loss_BIE_M)
        loss_BIE/=Nbatch
        loss_BC/=NG
        loss = loss_BIE+loss_BC
        loss.backward()
        optimizer.step()
        
        print('[%d] loss: %.6f' %
              (epoch + 1, loss))
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror

#完整的损失函数,合并训练，对Source采用batch策略
def train_full_bb(Source,Grids0,Net1,device,workspace,Nepoch,lr0,Netinvname):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//5), gamma=0.333)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    ap=1.0
    NG = len(Grids0)
    NS = Source.shape[0]
    Net1.train(mode=True)
    Nbatch = NS
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
                
    for epoch in range(Nepoch):
        loss_BIE_M = torch.zeros([Nbatch],dtype = torch.float32)
        loss_BC = 0
        loss = 0
        
        optimizer.zero_grad()
        for J in range(NG):                
            Grids0[J].update_func(Net1)
            loss_BC += Grids0[J].BC_loss()
        loss_BC/=NG
        loss_BC.backward()
        optimizer.step()
        
        optimizer.zero_grad()
        for J in range(NG):                
            Grids0[J].update_func(Net1)
        for J in range(NS):
            for K in range(NG):
                Grids0[K].update_fund(Source[J])
                SF,SDF = Grids0[K].integral_func()              
                loss_BIE_M[J]+=SF-SDF
                
        loss_BIE=torch.dot(loss_BIE_M,loss_BIE_M)
        loss_BIE/=Nbatch
        loss_BIE.backward()        
        optimizer.step()
        
        
        loss = loss_BIE+loss_BC
        print('[%d] loss: %.6f' %
              (epoch + 1, loss))
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror

#完整的损失函数,BC与BIE交替训练
def train_full_alternate(Source,Grids0,Net1,device,workspace,Nepoch,lr0,Netinvname):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//3), gamma=0.333)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    ap=1.0
    NG = len(Grids0)
    NS = Source.shape[0]
    Net1.train(mode=True)
    B_BIE = min(32,NS)
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
                
    for epoch in range(Nepoch):
        loss = Batch_BIE(NS,NG,Net1,Source,Grids0,B_BIE,optimizer)
        
        print('[%d] loss: %.6f' %
              (epoch + 1, loss))
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror

#完整的损失函数,BC与BIE交替训练,均采用batch wise策略
def train_full_alternate_bb(Source,Grids0,Net1,device,workspace,Nepoch,lr0,Netinvname):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//3), gamma=0.333)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    
    ap=1.0
    NG = len(Grids0)
    NS = Source.shape[0]
    B_BC = min(64, NG)
    B_BIE = min(32,NS)
    Net1.train(mode=True)
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
                
    for epoch in range(Nepoch):
        loss_BC = Batch_BC(NG,Net1,Grids0,B_BC,optimizer)
        loss_BIE = Batch_BIE(NS,NG,Net1,Source,Grids0,B_BIE,optimizer)
        # loss_BIE_M = torch.zeros([Nbatch],dtype = torch.float32)
        # loss_BC = 0
        # loss = 0
        
        # optimizer.zero_grad()
        # for J in range(NG):                
        #     Grids0[J].update_func(Net1)
        #     loss_BC += Grids0[J].BC_loss()
        # loss_BC/=NG
        # loss_BC.backward()
        # optimizer.step()
        
        # optimizer.zero_grad()
        # for J in range(NG):                
        #     Grids0[J].update_func(Net1)
        # for J in range(NS):
        #     for K in range(NG):
        #         Grids0[K].update_fund(Source[J])
        #         SF,SDF = Grids0[K].integral_func()              
        #         loss_BIE_M[J]+=SF-SDF
                
        # loss_BIE=torch.dot(loss_BIE_M,loss_BIE_M)
        # loss_BIE/=Nbatch
        # loss_BIE.backward()        
        # optimizer.step()
        loss = loss_BIE+loss_BC
        print('[%d] loss: %.6f' %
              (epoch + 1, loss))
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror

def train_full_rizzo_b(Grids0,Net1,device,workspace,Nepoch,lr0,Netinvname):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//5), gamma=0.2)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    ap=1.0
    NG = len(Grids0)
    Net1.train(mode=True)
    batchsize = min(32,NG)
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
                
    for epoch in range(Nepoch):  
        loss = Batch_full_rizzo(NG,Net1,Grids0,batchsize,optimizer)  
       
        print('[%d] loss: %.6f' %
              (epoch + 1, loss))
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror
#只训练给定的边界条件
def train_datadriven(Source,Grids0,Net1,device,workspace,Nepoch,lr0,Netinvname):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//5), gamma=0.2)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    ap=1.0
    NG = len(Grids0)
    NS = Source.shape[0]
    Net1.train(mode=True)
    Nbatch = NS
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
        
    for epoch in range(Nepoch):
       
        loss_BC = 0
        loss = 0
        optimizer.zero_grad()
        for J in range(NG):                
            Grids0[J].update_func(Net1)
            loss_BC +=Grids0[J].BC_loss()
        
                
        
        loss_BC/=NG
        loss = loss_BC
        loss.backward()
        optimizer.step()
        
        print('[%d] loss: %.6f' %
              (epoch + 1, loss))
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror

#data_driven 的batch版本,对边界点进行batch操作
def train_datadriven_b(Source,Grids0,Net1,device,workspace,Nepoch,lr0,Netinvname):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//5), gamma=0.2)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    ap=1.0
    NG = len(Grids0)
    batchsize = min(32,NG)
    Net1.train(mode=True)

    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
        
    for epoch in range(Nepoch):
        loss = Batch_BC(NG,Net1,Grids0,batchsize,optimizer)    
        print('[%d] loss: %.6f' %
                  (epoch + 1, loss))
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()          
        
        # ListN=list(range(NG))
        # random.shuffle(ListN)
        # ii = 0
        # i=0
        # '''
        # '''
        # while i+batchsize<NG:
        #     Ibatch=ListN[i:i+batchsize]
        #     i=i+batchsize
        #     optimizer.zero_grad()
        #     loss_BC = 0
        #     for J in Ibatch:                
        #         Grids0[J].update_func(Net1)
        #         loss_BC +=Grids0[J].BC_loss()
        
        #     loss_BC/=batchsize
        #     loss = loss_BC
        #     loss.backward()
        #     optimizer.step()
        
        # print('[%d] loss: %.3f' %
        #       (epoch + 1, loss))
        # Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        # scheduler.step()
    print('Finished Training')
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror

def train_datadriven_Node0(Nodes0,value0,type0,norm0,Net1,device,workspace,Nepoch,lr0,Netinvname):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//5), gamma=0.2)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    ap=1.0
    NN = Nodes0.shape[0]
    Net1.train(mode=True)
    Nbatch = NN
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
        
    for epoch in range(Nepoch):
       
        loss_BC = 0
        loss = 0
        optimizer.zero_grad()
        value = Net1(torch.tensor(Nodes0,dtype = torch.float32))
        loss_BC = MSE(value,torch.tensor(value0,dtype = torch.float32))              
        loss = loss_BC
        loss.backward()
        optimizer.step()
        
        print('[%d] loss: %.6f' %
              (epoch + 1, loss))
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror

def train_datadriven_Node(Nodes0,Net1,device,workspace,Nepoch,lr0,Netinvname):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//5), gamma=0.2)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    ap=1.0
    NN = Nodes0.DOF
    Net1.train(mode=True)
    Nbatch = NN
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
        
    for epoch in range(Nepoch):
       
        loss_BC = 0
        loss = 0
        optimizer.zero_grad()
        # value = Net1(torch.tensor(Nodes0,dtype = torch.float32))
        Nodes0.update_func(Net1)
        loss_BC = Nodes0.BC_loss()              
        loss = loss_BC
        loss.backward()
        optimizer.step()
        
        print('[%d] loss: %.6f' %
              (epoch + 1, loss))
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror       

#只训练边界积分方程
def train_partitial(Source,Grids0,Net1,device,workspace,Nepoch,lr0,Netinvname):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//5), gamma=0.2)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    ap=1.0
    NG = len(Grids0)
    NS = Source.shape[0]
    Net1.train(mode=True)
    Nbatch = NS
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
                
    for epoch in range(Nepoch):
        loss_BIE_M = torch.zeros([Nbatch],dtype = torch.float32)
       
        loss = 0
        optimizer.zero_grad()
        for J in range(NG):                
            Grids0[J].update_func(Net1)
            
        for J in range(NS):
            for K in range(NG):
                Grids0[K].update_fund(Source[J])
                SF,SDF = Grids0[K].integral_func()              
                loss_BIE_M[J]+=SF-SDF
                
        loss_BIE=torch.dot(loss_BIE_M,loss_BIE_M)
        loss_BIE/=Nbatch
        
        loss = loss_BIE
        loss.backward()
        optimizer.step()
        
        print('[%d] loss: %.6f' %
              (epoch + 1, loss))
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror

def train_partitial_rizzo(Grids0,Net1,device,workspace,Nepoch,lr0,Netinvname):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//2), gamma=0.5)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    ap=1.0
    NG = len(Grids0)
    Net1.train(mode=True)
    Nbatch = NG
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
                
    for epoch in range(Nepoch):
        loss_BIE_M = torch.zeros([Nbatch],dtype = torch.float32)
        # loss_BIE_M1 = torch.zeros([Nbatch],dtype = torch.float32)
       
        loss = 0
        optimizer.zero_grad()
        for J in range(NG):                
            Grids0[J].update_func(Net1)
            
        for J in range(NG):
            for K in range(NG):
                if K!=J:
                    Grids0[K].update_fund(Grids0[J].Source)
                    SF,SDF= Grids0[K].integral_func()  
                else:
                    # Grids0[K].update_fund(Grids0[J].Source)#非常重要，但是为什么？这句话没起作用啊
                    SF,SDF = Grids0[K].integral_func_single(Net1)  
                loss_BIE_M[J]+=SF-SDF
                # print(SF)
                # print(SDF)
                # loss_BIE_M1[J]+=SF1-SDF
                
        loss_BIE=torch.dot(loss_BIE_M,loss_BIE_M)
        loss_BIE/=Nbatch
        
        loss = loss_BIE
        loss.backward()
        optimizer.step()
        
        print('[%d] loss: %.6f' %
              (epoch + 1, loss))
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror
# 只训练边界积分方程,batchwise版本
def train_partitial_rizzo_b(Grids0,Net1,device,workspace,Nepoch,lr0,Netinvname):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//5), gamma=0.2)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    ap=1.0
    NG = len(Grids0)
    Net1.train(mode=True)
    batchsize = min(80,NG)
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
                
    for epoch in range(Nepoch):  
        loss = Batch_BIE_rizzo(NG,Net1,Grids0,batchsize,optimizer)     
        print('[%d] loss: %.6f' %
              (epoch + 1, loss))
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror
def train_partitial_rizzo_HB(Grids0,H,G,Net1,device,workspace,Nepoch,lr0,Netinvname):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//5), gamma=0.2)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    ap=1.0
    NG = len(Grids0)
    Net1.train(mode=True)
    batchsize = min(40,NG)
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
                
    for epoch in range(Nepoch):  
        loss = Batch_BIE_rizzo_HB(NG,H,G,Net1,Grids0,batchsize,optimizer,10)     
        print('[%d] loss: %.6f' %
              (epoch + 1, loss))
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror

def test_partitial_rizzo_HB(Grids0,H,G,Net1):
    #先计算基本解的值，这部分在训练过程中一直保持不变
    Ngauss=10
    NG = len(Grids0)
    Net1.eval()
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
    DOF = H.size()[1]
    intdf = 0
    df = 0
    df = torch.zeros([DOF],dtype = torch.float32)
    f = torch.zeros([DOF],dtype = torch.float32)
    intCPV = torch.zeros([NG],dtype = torch.float32)
    intlog = torch.zeros([NG],dtype = torch.float32)
    # loss_BIE_M = torch.zeros([B1],dtype = torch.float32)
   
    for J in range(NG):                
        Grids0[J].update_func(Net1)
        intlog[J],intCPV[J]=Grids0[J].return_func_single(Net1)
        
        
        if Grids0[J].type==0:
            f[J*Ngauss:(J+1)*(Ngauss)] = Grids0[J].value.view(-1)
            df[J*Ngauss:(J+1)*(Ngauss)] = Grids0[J].df.view(-1)
            
        elif Grids0[J].type==1:
            f[J*Ngauss:(J+1)*(Ngauss)] = Grids0[J].f.view(-1)
            df[J*Ngauss:(J+1)*(Ngauss)] = Grids0[J].value.view(-1)
            
    intdf=(torch.mv(H,f)+intCPV).detach().numpy()
    intf=(torch.mv(G,df)+intlog).detach().numpy()        
  
    
    return intf,intdf

def train_partitial_NO_rizzo(Source,Grids0,Net1,device,workspace,Nepoch,lr0,Netinvname):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//3), gamma=0.3)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    NG = Source.shape[0]
    Net1.train(mode=True)
    # batchsize = min(40,NG)
    batchsize = NG
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
                
    for epoch in range(Nepoch):  
        loss = Batch_BIE_NO_rizzo(NG,Net1,Grids0,batchsize,optimizer)     
        print('[%d] loss: %.6f' %
              (epoch + 1, loss))
        if epoch%500==499:
            Grids0.update_func(Net1,alpha=0)
            draw(Grids0,0,epoch)
                
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror

def train_partitial_rizzo_again(Grids0,Net1,device,workspace,Nepoch,lr0):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch), gamma=0.3)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    
    # batchsize = min(40,NG)
    Xerror=np.empty([0],dtype=float)
    inner_err = np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
                
    for epoch in range(Nepoch):  
        Net1.train(mode=True)
        optimizer.zero_grad()
        Grids0.update_func(Net1)
        
        loss=Grids0.update_loss(Net1)
        loss.backward()
        optimizer.step()
        
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        error = plot_inner(Grids0,Net1,100,area = np.array([0,0.5,2,1]))   
        
        if epoch%10==0:
            print(' epoch : %i, the loss : %f ,  error: %f' % \
                  (epoch, loss.data, error))
            
        if epoch%50==49:
            Grids0.update_func(Net1)
            plt.figure()
            plt.semilogy(inner_err)
            draw(Grids0,epoch)
                

        inner_err = np.append(inner_err ,error)
        # tr,te=Derror_during_training(Grids0,Net1)
        # trainerr=np.append(trainerr,tr.cpu().detach().numpy())
        # testerr=np.append(testerr,te.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    return Xerror,inner_err

def train_partitial_rizzo_again_efficient_mode(Grids0,Net1,device,workspace,Nepoch,lr0):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch), gamma=0.3)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    
    # batchsize = min(40,NG)
    Xerror=np.empty([0],dtype=float)
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
    start = time.time()
                
    for epoch in range(Nepoch):  
        Net1.train(mode=True)
        optimizer.zero_grad()
        Grids0.update_func(Net1)
        
        loss=Grids0.update_loss(Net1)
        loss.backward()
        optimizer.step()
        
        # loss = Batch_BIE_NO_rizzo(NG,Net1,Grids0,batchsize,optimizer)     
        
        if epoch%5==4:
            print('[%d] loss: %.4e' %
                  (epoch + 1, loss))
        if epoch%1000==4:
            end = time.time()
            consume_time = end - start
            print('time: %.4e' %
                  (consume_time))        
        # if epoch%50==49:
        #     Grids0.update_func(Net1)
        #     plt.figure()
        #     plt.semilogy(inner_err)
        #     draw(Grids0,epoch)
                
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        # error = plot_inner(Grids0,Net1,100,area = np.array([0,0.5,2,1]))
        # inner_err = np.append(inner_err ,error)
        # tr,te=Derror_during_training(Grids0,Net1)
        # trainerr=np.append(trainerr,tr.cpu().detach().numpy())
        # testerr=np.append(testerr,te.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    return Xerror
def Derror_during_training(Grids0,Net1):
    Net1.eval()
    _,DUtr = Grids0.update_func_any(Net1,Grids0.GP,Grids0.norm)
    DUtr=DUtr
    DUactr = Grids0.func(Grids0.GP,Grids0.norm,bctype = 1,para=1).reshape([-1])
    error_train = np.linalg.norm(DUtr-DUactr)/np.linalg.norm(DUactr)
    
    _,DU = Grids0.update_func_any(Net1,Grids0.testpoint,Grids0.testpoint_norm)
    DU=DU
    DUac = Grids0.func(Grids0.testpoint,norm=Grids0.testpoint_norm,bctype = 1,para=1).reshape([-1])
    error_test = np.linalg.norm(DU-DUac)/np.linalg.norm(DUac)
    
    return error_train,error_test    
def test_partitial_NO_rizzo(Grids0,Net1):
    #先计算基本解的值，这部分在训练过程中一直保持不变
    Grids0.update_func(Net1)
    
    return Grids0.f, Grids0.df
    
def fundamental_para(R,Norm):
    LR = np.linalg.norm(R,axis=-1)
    fs = -np.log(LR)/2/np.pi
    dfs = -np.sum(R*Norm,axis=-1)/2/np.pi/LR/LR
    fs = torch.tensor(fs,dtype=torch.float32).reshape([-1,1])
    dfs = torch.tensor(dfs,dtype=torch.float32).reshape([-1,1])
    return fs,dfs    
def draw(Grids0,epoch):

        
        predict = Grids0.f.detach().cpu().numpy()
        predict[Grids0.ucol_index] = Grids0.df[Grids0.ucol_index].detach().cpu().numpy()
        err=np.linalg.norm(Grids0.solution-predict)/np.linalg.norm(Grids0.solution)
        plt.figure()
        plt.title("int_p = "+str(Grids0.H.size()[1])+", S_p = "+str(Grids0.H.size()[0])+", epoch ="+str(epoch+1)+", err = "+str(err))
        plt.plot(Grids0.solution)
        plt.plot(predict)
        plt.legend(["accurate","train"])
        
        plt.figure()
        plt.title("int_p = "+str(Grids0.H.size()[1])+", S_p = "+str(Grids0.H.size()[0])+", epoch ="+str(epoch+1)+", err = "+str(err))
        
        plt.plot(Grids0.solution-predict)
        plt.legend(["train error"])
        plt.show()

def get_inner_error(Grids0,Net,N,area = np.array([-1,0,1,1])):
    Net.eval()
    x=np.zeros([N*N,2])
    x0=area[0]-area[2]/2+0.03
    x1=area[0]+area[2]/2-0.03
    y0 = area[1]-area[3]/2+0.03
    y1 = area[1]+area[3]/2-0.03
    x[:,0]=np.repeat(np.linspace(x0,x1,N),N)
    x[:,1]=np.tile(np.linspace(y0,y1,N),N)
    u_pred=Grids0.inner(Net,x).cpu().numpy()
    u_exact = Grids0.func(x).reshape([-1])
    error = np.abs(u_exact-u_pred)
    error_t = np.linalg.norm(error)/np.linalg.norm(u_exact)
    
    return error_t
def plot_inner(Grids0,Net,N,area=np.array([-1,0,1,1]),device = 'cpu'):
    
    x=np.zeros([N*N,2])
    
    #确定绘图范围
    #三型裂纹
    x0=area[0]-area[2]/2+0.03
    x1=area[0]+area[2]/2-0.03
    y0 = area[1]-area[3]/2+0.03
    y1 = area[1]+area[3]/2-0.03
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
    x[:,0]=np.repeat(np.linspace(x0,x1,N),N)
    x[:,1]=np.tile(np.linspace(y0,y1,N),N)
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
    
    U=Grids0.inner(Net,x).cpu().numpy()
    Uac = Grids0.func(x).reshape([-1])

    fvalue = Uac-U
    err = np.linalg.norm(fvalue)/np.linalg.norm(Uac)
    return err
    