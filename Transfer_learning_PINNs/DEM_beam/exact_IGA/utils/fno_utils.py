#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Functions for the Fourier Neural Operator
Adapted from https://github.com/neuraloperator/neuraloperator/blob/master/utilities3.py
'''
import torch
import torch.nn.functional as F
import numpy as np
import operator
from functools import reduce
from timeit import default_timer

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c

def du_FD(U, device):
    '''
    Computes the first derivative of U using the finite difference
    stencil [-1/2, 0, 1/2] in the interior, the left edge stencil
    [-1, 1], and the right edge stencil [-1, 1]

    Parameters
    ----------
    U : (tensor of dimension batch_sizex1xnum_nodes)
        function u(X) evaluated at equally spaced nodes in each row
   
    device : (device type)
        cuda (for GPU ) or cpu
   
    
    Returns
    -------
    F_fd : (tensor of dimension batch_sizex1x(num_nodes+2))
        function f(x) = u'(x) evaluated at the nodes and endpoints of the interval

    '''
    num_nodes = U.shape[2]
    filters = torch.tensor([-1/2, 0., 1/2]).unsqueeze(0).unsqueeze(0).to(device)
    F_fd = F.conv1d(U, filters, padding=0, groups=1)
    left_col =  -1*U[:,:,0:1]+1*U[:,:,1:2]
    right_col = -1*U[:, :, -2:-1] + 1*U[:, :, -1:]
    F_fd = torch.cat((left_col, F_fd, right_col), axis=2)
    F_fd *= (num_nodes-1)
    return F_fd

def d2u_FD(U, device):
    '''
    Computes the (negative) second derivative of U using the finite difference
    stencil [-1, 2, -1]

    Parameters
    ----------
    U : (tensor of dimension batch_sizex1xnum_nodes)
        function u(X) evaluated at equally spaced nodes in each row
   
    device : (device type)
        cuda (for GPU ) or cpu
   
    
    Returns
    -------
    F_fd : (tensor of same dimension as U, padded with zeros)
        function f(x) = -u''(x) evaluated at the nodes

    '''
    num_nodes = U.shape[2]
    filters = torch.tensor([-1., 2., -1.]).unsqueeze(0).unsqueeze(0).to(device)
    F_fd = F.conv1d(U, filters, padding=0, groups=1)
    F_fd *= (num_nodes+1)**2
        
    return F_fd

def Laplacian(U, device):
    '''
    Computes the (negative) laplacian of U using the finite difference
    stencil [ 0, -1,  0
             -1,  4, -1
              0, -1,  0]

    Parameters
    ----------
    U : (tensor of dimension batch_size x 1 x num_nodes x num_nodes)
   
    device : (device type)
        cuda (for GPU ) or cpu
   
    
    Returns
    -------
    F_fd : (tensor of same dimension as U, padded with zeros)
        function f(x) = -\nabla^2 u(x, y) evaluated at the nodes

    '''
    num_nodes = U.shape[2]
    filters = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]).unsqueeze(0).unsqueeze(0).to(device)
    F_fd = F.conv2d(U, filters, groups=1)
    F_fd *= (num_nodes-1)**2
        
    return F_fd

def diff_x(U, device):
    num_nodes = U.shape[3]
    filter_dx = torch.tensor([[-1/2, 0., 1/2]]).unsqueeze(0).unsqueeze(0).to(device)
    dudx = F.conv2d(U, filter_dx)
    dudx_left = -1*U[:, :, :, 0:1] + 1*U[:, :, :, 1:2]
    dudx_right = -1*U[:, :, :, -2:-1] + 1*U[:, :, :, -1:]
    dudx = torch.cat((dudx_left, dudx, dudx_right), dim=3)    
    dudx *= num_nodes-1    
    return dudx

def diff_y(U, device):
    num_nodes = U.shape[2]
    filter_dy = torch.tensor([[-1/2], [0.], [1/2]]).unsqueeze(0).unsqueeze(0).to(device)
    dudy = F.conv2d(U, filter_dy)
    dudy_bottom = -1*U[:, :, 0:1, :] + 1*U[:, :, 1:2, :] 
    dudy_top = -1*U[:, :, -2:-1, :] + 1*U[:, :, -1:, :]
    dudy = torch.cat((dudy_bottom, dudy, dudy_top), dim=2)    
    dudy *= num_nodes-1    
    return dudy

def diff_x_1st(U, device, data_type=torch.float32):
    filter_dx = torch.tensor([[-1., 1]], dtype=data_type).unsqueeze(0).unsqueeze(0).to(device)
    dudx = F.conv2d(U, filter_dx)
    dudx_right = -1*U[:, :, :, -2:-1] + 1*U[:, :, :, -1:]
    dudx = torch.cat((dudx, dudx_right), dim=3)    
    return dudx

def diff_y_1st(U, device, data_type=torch.float32):
    filter_dy = torch.tensor([[-1], [1]], dtype=data_type).unsqueeze(0).unsqueeze(0).to(device)
    dudy = F.conv2d(U, filter_dy)
    dudy_top = -1*U[:, :, -2:-1, :] + 1*U[:, :, -1:, :]
    dudy = torch.cat((dudy, dudy_top), dim=2)    
    return dudy

def train_fno(model, train_loader, test_loader, loss_func, optimizer, scheduler, 
              device, epochs, batch_size):
    model.train()
    ntrain = len(train_loader)
    ntest = len(test_loader)
    train_mse_log = np.zeros(epochs)
    train_l2_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)
    for ep in range(epochs):    
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x1, x2, y in train_loader:
            
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x1, x2)
            
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2 = loss_func(out.view(batch_size, -1), y.view(batch_size, -1))/len(x1)
            l2.backward() # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_mse)
        else:
            scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x1, x2, y in test_loader:
                
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                out = model(x1, x2)
                test_l2 += loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item()/len(x1)

        train_mse /= ntrain
        train_l2 /= ntrain
        test_l2 /= ntest
        
        train_mse_log[ep] = train_mse
        train_l2_log[ep] = train_l2
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % 5 == 0:
            print("ep,      t2-t1,      train_mse,      train_l2,       test_l2")
            print(ep, t2-t1, train_mse, train_l2, test_l2)
            
    return train_mse_log, train_l2_log, test_l2_log

def train_fno_domain(model, train_loader, test_loader, loss_func, optimizer, scheduler, 
              device, epochs, batch_size):
    model.train()
    ntrain = len(train_loader)
    ntest = len(test_loader)
    train_mse_log = np.zeros(epochs)
    train_l2_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)
    for ep in range(epochs):    
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x1, x2, x3, y in train_loader:
            
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x1, x2, x3)
            
            #mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='sum')/len(x1)
            l2 = loss_func(out.view(batch_size, -1), y.view(batch_size, -1))/len(x1)
            l2.backward() # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_mse)
        else:
            scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x1, x2, x3, y in test_loader:
                
                x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
                out = model(x1, x2, x3)
                test_l2 += loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item()/len(x1)

        train_mse /= ntrain
        train_l2 /= ntrain
        test_l2 /= ntest
        
        train_mse_log[ep] = train_mse
        train_l2_log[ep] = train_l2
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % 5 == 0:
            print("ep,      t2-t1,      train_mse,      train_l2,       test_l2")
            print(ep, t2-t1, train_mse, train_l2, test_l2)
            
    return train_mse_log, train_l2_log, test_l2_log

def train_pino(model, train_loader, test_loader, loss_func, optimizer, scheduler, 
              device, epochs, batch_size):
    model.train()
    ntrain = len(train_loader)
    ntest = len(test_loader)
    train_mse_log = np.zeros(epochs)
    train_l2_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)
    for ep in range(epochs):    
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x in train_loader:
            
            x = x[0].to(device)
            optimizer.zero_grad()
            u_pred = model(x)
            u_pred = F.pad(u_pred, (0,0,1,1), mode='constant', value=0)
            f_pred = d2u_FD(u_pred.permute(0,2,1), device).permute(0,2,1)
            
            mse = F.mse_loss(x.view(batch_size, -1), f_pred.view(batch_size, -1), reduction='mean')/len(x)
            l2 = loss_func(x.view(batch_size, -1), f_pred.view(batch_size, -1))/len(x)
            l2.backward() # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_mse)
        else:
            scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x in test_loader:
                
                x =x[0].to(device)
                u_pred = model(x)
                u_pred = F.pad(u_pred, (0,0,1,1), mode='constant', value=0)
                f_pred = d2u_FD(u_pred.permute(0,2,1), device).permute(0,2,1)
                test_l2 += loss_func(x.view(batch_size, -1), f_pred.view(batch_size, -1)).item()/len(x)

        train_mse /= ntrain
        train_l2 /= ntrain
        test_l2 /= ntest
        
        train_mse_log[ep] = train_mse
        train_l2_log[ep] = train_l2
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % 50 == 0:
            print(ep, t2-t1, train_mse, train_l2, test_l2)
            
    return train_mse_log, train_l2_log, test_l2_log

def train_pino_dem(model, train_loader, test_loader, loss_func, optimizer, scheduler, 
              device, epochs, batch_size):
    model.train()
    ntrain = len(train_loader)
    ntest = len(test_loader)
    train_l2_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)
    for ep in range(epochs):    
        t1 = default_timer()
        train_l2 = 0
        for x in train_loader:
            
            x = x[0].to(device)
            mask = torch.ones_like(x)
            optimizer.zero_grad()
            u_pred = model(x)
            u_pred = F.pad(u_pred, (0,0,1,1), mode='constant', value=0)

            du_pred = du_FD(u_pred.permute(0,2,1), device).permute(0,2,1)
                        
            x = F.pad(x, (0,0,1,1), mode='constant', value = 0)
            mask = F.pad(mask, (0,0,1,1), mode='constant', value=0.5)
            #integrand = 1/2*du_pred.view(batch_size, -1)**2-u_pred.view(batch_size, -1)*x.view(batch_size, -1)
            integrand = 1/2*du_pred**2-u_pred*x
            
            #u_pred = 1/2*(u_pred[:,:-1,:] + u_pred[:,1:,:])
            l2 = torch.mean(integrand*mask)/len(x)
                        
            #l2 = loss_func(x.view(batch_size, -1), f_pred.view(batch_size, -1))
            l2.backward() # use the l2 relative loss

            optimizer.step()
            train_l2 += l2.item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(l2)
        else:
            scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x in test_loader:
                
                x =x[0].to(device)
                u_pred = model(x)
                u_pred = F.pad(u_pred, (0,0,1,1), mode='constant', value=0)
                f_pred = d2u_FD(u_pred.permute(0,2,1), device).permute(0,2,1)
                test_l2 += loss_func(x.reshape(batch_size, -1), f_pred.reshape(batch_size, -1)).item()/len(x)

        train_l2 /= ntrain
        test_l2 /= ntest
        
        train_l2_log[ep] = train_l2
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % 50 == 0:
            print(ep, t2-t1, train_l2, test_l2)
            
    return train_l2_log, test_l2_log

def train_pino_2D(model, train_loader, test_loader, loss_func, optimizer, scheduler, 
              device, epochs, step_size, batch_size):
    model.train()
    ntrain = len(train_loader)
    ntest = len(test_loader)
    train_mse_log = np.zeros(epochs)
    train_l2_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)
    for ep in range(epochs):    
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x in train_loader:
            x = x[0].to(device)
            optimizer.zero_grad()
            u_pred = model(x)
            u_pred = F.pad(u_pred, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
            f_pred = Laplacian(u_pred.permute(0,3,1,2), device).permute(0,2,3,1)
            
            mse = F.mse_loss(x.view(batch_size, -1), f_pred.view(batch_size, -1), reduction='mean')/len(x)
            l2 = loss_func(x.view(batch_size, -1), f_pred.view(batch_size, -1))/len(x)
                        
            l2.backward() # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_mse)
        else:
            scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x in test_loader:
                x =x[0].to(device)
                u_pred = model(x)
                u_pred = F.pad(u_pred, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
                f_pred = Laplacian(u_pred.permute(0,3,1,2), device).permute(0,2,3,1)
                test_l2 += loss_func(x.view(batch_size, -1), f_pred.view(batch_size, -1)).item()/len(x)

        train_mse /= ntrain
        train_l2 /= ntrain
        test_l2 /= ntest
        
        train_mse_log[ep] = train_mse
        train_l2_log[ep] = train_l2
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % step_size == 0:
            print(ep, t2-t1, train_mse, train_l2, test_l2)
            
    return train_mse_log, train_l2_log, test_l2_log

def train_pino_dem_2D(model, train_loader, test_loader, loss_func, optimizer, scheduler, 
              device, epochs, step_size, batch_size):
    model.train()
    ntrain = len(train_loader)
    ntest = len(test_loader)
    train_l2_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)
    for ep in range(epochs):    
        t1 = default_timer()
        train_l2 = 0
        for x in train_loader:
            x = x[0].to(device)
            mask = torch.ones_like(x)
            optimizer.zero_grad()
            u_pred = model(x)
            u_pred = F.pad(u_pred, (0, 0, 1, 1, 1, 1), mode='constant', value=0)

            #du_pred_x = diff_x(u_pred.permute(0,3,1,2), device).permute(0,2,3,1)
            #du_pred_y = diff_y(u_pred.permute(0,3,1,2), device).permute(0,2,3,1)
            
            dUdx = diff_x_1st(u_pred.permute(0,3,1,2), device).permute(0,2,3,1)
            dUdy = diff_y_1st(u_pred.permute(0,3,1,2), device).permute(0,2,3,1)
            
            x = F.pad(x, (0, 0, 1, 1, 1, 1), mode='replicate')
            mask = F.pad(mask, (0, 0, 1, 1, 1, 1), mode='constant', value=0.5)
            mask[:, 0, 0, 0] = 0.25
            mask[:, 0, -1, 0] = 0.25
            mask[:, -1, 0, 0] = 0.25
            mask[:, -1, -1, 0] = 0.25
                        
            #integrand = 1/2*(du_pred_x**2 + du_pred_y**2)-u_pred*x
            num_nodes_x = u_pred.shape[2]
            num_nodes_y = u_pred.shape[1]

            loss_grad = 1/2*torch.sum(2/3*dUdx[:, 0:-1, 0:-1, :]**2 + 2/3*dUdx[:, 1:,0:-1, :]**2 + dUdy[:, 0:-1,0:-1, :]**2 - \
                                  1/3*dUdx[:, 0:-1, 0:-1, :]*dUdx[:, 1:,0:-1, :] - dUdx[:, 0:-1, 0:-1, :]*dUdy[:, 0:-1,0:-1, :] + \
                                      dUdx[:, 1:,0:-1, :]*dUdy[:, 0:-1,0:-1, :])/len(x)
            loss_int = torch.sum((x*u_pred)*mask)/len(x)/((num_nodes_x-1)*(num_nodes_y-1))
            l2 = loss_grad - loss_int
            #l2 = torch.mean(integrand*mask)/len(x)
            l2.backward() 

            optimizer.step()
            train_l2 += l2.item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(l2)
        else:
            scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x in test_loader:
                x =x[0].to(device)
                u_pred = model(x)
                u_pred = F.pad(u_pred, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
                f_pred = Laplacian(u_pred.permute(0,3,1,2), device).permute(0,2,3,1)
                test_l2 += loss_func(x.view(batch_size, -1), f_pred.view(batch_size, -1)).item()/len(x)

        train_l2 /= ntrain
        test_l2 /= ntest
        train_l2_log[ep] = train_l2
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % step_size == 0:
            print(ep, t2-t1, train_l2, test_l2)
            
    return train_l2_log, test_l2_log

def train_pino_darcy(model, train_loader, test_loader, loss_func, optimizer, scheduler, 
              device, epochs, step_size, batch_size):
    model.train()
    ntrain = len(train_loader)
    ntest = len(test_loader)
    train_mse_log = np.zeros(epochs)
    train_l2_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)
    for ep in range(epochs):    
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x in train_loader:
            x = x[0].to(device)
            optimizer.zero_grad()
            u_pred = model(x)
            
            u_pred = F.pad(u_pred, (0, 0, -1, -1, -1, -1), mode='constant', value=0)
            u_pred = F.pad(u_pred, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
            #a = F.pad(x, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
            a = x
            dUdx = diff_x(u_pred.permute(0,3,1,2), device).permute(0,2,3,1)
            dUdy = diff_y(u_pred.permute(0,3,1,2), device).permute(0,2,3,1)
            
            a_dUdx = a*dUdx
            a_dUdy = a*dUdy
            
            f_pred_1 = diff_x(a_dUdx.permute(0,3,1,2), device).permute(0,2,3,1)
            f_pred_2 = diff_y(a_dUdy.permute(0,3,1,2), device).permute(0,2,3,1)
            
            f_pred = f_pred_1 + f_pred_2
            
            #f_pred = F.pad(f_pred, (0, 0, -1, -1, -1, -1), mode='constant', value=0)
            
            f_ground = torch.ones_like(x)
            
            mse = F.mse_loss(f_ground.view(batch_size, -1), f_pred.view(batch_size, -1), reduction='mean')/len(x)
            l2 = loss_func(f_ground.view(batch_size, -1), f_pred.view(batch_size, -1))/len(x)
                        
            l2.backward() # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_mse)
        else:
            scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x in test_loader:
                x =x[0].to(device)
                u_pred = model(x)
                u_pred = F.pad(u_pred, (0, 0, -1, -1, -1, -1), mode='constant', value=0)
                u_pred = F.pad(u_pred, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
                
                #a = F.pad(x, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
                a = x
                
                dUdx = diff_x(u_pred.permute(0,3,1,2), device).permute(0,2,3,1)
                dUdy = diff_y(u_pred.permute(0,3,1,2), device).permute(0,2,3,1)
                
                a_dUdx = a*dUdx
                a_dUdy = a*dUdy
                
                f_pred_1 = diff_x(a_dUdx.permute(0,3,1,2), device).permute(0,2,3,1)
                f_pred_2 = diff_y(a_dUdy.permute(0,3,1,2), device).permute(0,2,3,1)
                
                f_pred = f_pred_1 + f_pred_2
                
                #f_pred = F.pad(f_pred, (0, 0, -1, -1, -1, -1), mode='constant', value=0)
                
                f_ground = torch.ones_like(x)
                
                test_l2 += loss_func(f_ground.view(batch_size, -1), f_pred.view(batch_size, -1)).item()/len(x)

        train_mse /= ntrain
        train_l2 /= ntrain
        test_l2 /= ntest
        
        train_mse_log[ep] = train_mse
        train_l2_log[ep] = train_l2
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % step_size == 0:
            print(ep, t2-t1, train_mse, train_l2, test_l2)
            
    return train_mse_log, train_l2_log, test_l2_log

