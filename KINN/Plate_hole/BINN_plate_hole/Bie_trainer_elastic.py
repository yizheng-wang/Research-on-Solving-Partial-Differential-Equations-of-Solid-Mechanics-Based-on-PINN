# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 22:38:19 2022

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

def draw(Grids0,epoch=0):

        # plt.legend (loc='lower right', fontsize=40)
        
        ax=plt.gca()
        x_locator = plt.MultipleLocator(0.5)
        ax.xaxis.set_major_locator(x_locator)
        
        predict = Grids0.df.detach().cpu().numpy()
        predict[Grids0.Nu_index] = Grids0.f[Grids0.Nu_index].detach().cpu().numpy()
        err = -1
        
        NNODE = predict.shape[0]//2
        if np.linalg.norm(Grids0.solution)!=0:
            err=np.linalg.norm(Grids0.solution-predict)/np.linalg.norm(Grids0.solution)
        plt.figure(dpi=400)
        # plt.title("int_p = "+str(Grids0.H.size()[1])+", S_p = "+str(Grids0.H.size()[0])+", epoch ="+str(epoch+1)+", err = "+str(err))
        plt.plot(Grids0.solution[0:NNODE])
        plt.plot(predict[0:NNODE],ls='--')
        plt.plot(Grids0.solution[NNODE:],c='black')
        plt.plot(predict[NNODE:],ls='--',c='red')

        font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 14}
        plt.rc('font',**font)
        # plt.ylim(ymin=-0.45)
        # plt.xlabel('x')
        # plt.ylabel('u')
        plt.legend(["u$_1$,exact","u$_1$,prediction","u$_2$,exact","u$_2$,prediction"],loc='upper right')
        
        
        
        plt.figure(dpi=400)
        # plt.title("int_p = "+str(Grids0.H.size()[1])+", S_p = "+str(Grids0.H.size()[0])+", epoch ="+str(epoch+1)+", err = "+str(err))
        
        plt.plot(Grids0.solution[0:NNODE]-predict[0:NNODE])
        plt.plot(Grids0.solution[NNODE:]-predict[NNODE:])
        plt.legend(["u$_1$,error","u$_2$,error"])
        plt.show()
        
def train_partitial_rizzo_again(Grids0,Net1,device,workspace,Nepoch,lr0,Netinvname):
    T1 =time.perf_counter()

    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//3), gamma=0.3)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    Net1.train(mode=True)
    # batchsize = min(40,NG)
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
                
    for epoch in range(Nepoch):  
        optimizer.zero_grad()
        Grids0.update_func(Net1)
        loss=Grids0.update_loss(Net1)
        loss.backward()
        optimizer.step()
        # loss = Batch_BIE_NO_rizzo(NG,Net1,Grids0,batchsize,optimizer)    
        if epoch%10==0:
            print('[%d] loss: %.4e' %
                  (epoch + 1, loss))
        
        if epoch%200==199:
            Grids0.update_func(Net1)
            # draw(Grids0,epoch)
                
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()
    T2 =time.perf_counter()    
     
    print('Finished Training, training time: %s second' % ((T2 - T1)))
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror

def train_hete(Grids0,Grids1,Net1,device,workspace,Nepoch,lr0,Netinvname,beta=10):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//3), gamma=0.3)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    Net1.train(mode=True)
    # batchsize = min(40,NG)
    Xerror1=np.empty([0],dtype=float)
    Xerror2=np.empty([0],dtype=float)
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
    # beta=10
    T1 = time.perf_counter()
    for epoch in range(Nepoch):  
        optimizer.zero_grad()
        Grids0.update_func(Net1)
        Grids1.update_func(Net1)
        loss1 = Grids0.update_loss(Net1)
        loss2 = Grids1.update_loss(Net1)
        loss=loss1+loss2*beta
        loss.backward()
        optimizer.step()
        # loss = Batch_BIE_NO_rizzo(NG,Net1,Grids0,batchsize,optimizer)     
        
        if epoch%5000==4999:
            print('[%d] loss: %.6f =  %.6f +  %.6f' %
                  (epoch + 1, loss,loss1,loss2*beta))
            # Grids0.update_func(Net1)
            # draw(Grids0,epoch)
            # Grids1.update_func(Net1)
            # draw(Grids1,epoch)
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        Xerror1=np.append(Xerror1,loss1.cpu().detach().numpy())
        Xerror2=np.append(Xerror2,loss2.cpu().detach().numpy())
        scheduler.step()
    T2 = time.perf_counter()
    print('Finished Training, 程序运行时间:%s毫秒' % ((T2 - T1)*1000))
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror,Xerror1,Xerror2

def train_couple(Grids0,Grids1,Net1,device,workspace,Nepoch,lr0,Netinvname):
    optimizer = torch.optim.Adam(Net1.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(Nepoch//3), gamma=0.3)
    #先计算基本解的值，这部分在训练过程中一直保持不变
    Net1.train(mode=True)
    # batchsize = min(40,NG)
    Xerror=np.empty([0],dtype=float)
    # for I in range(NS):
    #         for J in range(NG):
    #             Grids0[J].update_fund(Source[I])
                
    for epoch in range(Nepoch):  
        optimizer.zero_grad()
        Grids0.update_func(Net1)
        loss=Grids0.update_loss(Net1)
        loss.backward()
        optimizer.step()
        # loss = Batch_BIE_NO_rizzo(NG,Net1,Grids0,batchsize,optimizer)     
        print('[%d] loss: %.6f' %
              (epoch + 1, loss))
        if epoch%500==499:
            Grids0.update_func(Net1)
            draw(Grids0,epoch)
                
        Xerror=np.append(Xerror,loss.cpu().detach().numpy())
        scheduler.step()
    print('Finished Training')
    torch.save(Net1.state_dict(), Netinvname)
    return Xerror

