# 这里用里兹的方法研究裂纹问题，两个特解网络，不用可能位移场，突出
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
from itertools import chain

torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
penalty = 3000
delta = 0.0
train_p = 0
nepoch_u0 = 501 
a = 1

def interface1(Ni1): # 内部的边界点,由于交界面附近的梯度很大，所以多配一些点
    '''
     生成裂纹尖端上半圆的点，为了多分配点
    '''
    
    theta = np.pi*np.random.rand(Ni1) # 角度0到pi
    rp = delta*np.random.rand(Ni1) # 半径0到delta
    x = np.cos(theta) * rp
    y = np.sin(theta) * rp
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi,  requires_grad=True, device='cuda')
    return xi
def interface2(Ni1): # 外部的边界点
    '''
     生成裂纹尖端下半圆的点，为了多分配点
    '''
    
    theta = -np.pi*np.random.rand(Ni1) # 角度0到-pi
    rp = delta*np.random.rand(Ni1) # 半径0到delta
    x = np.cos(theta) * rp
    y = np.sin(theta) * rp
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi,  requires_grad=True, device='cuda')
    return xi
def interface(Ni):
    '''
     生成交界面的随机点
    '''
    re = np.random.rand(Ni)
    theta = 0
    x = np.cos(theta) * re
    y = np.sin(theta) * re
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi,  requires_grad=True, device='cuda')
    return xi
def train_data(Nb, Nf):
    '''
    生成强制边界点，四周以及裂纹处
    生成上下的内部点
    '''
    

    xu = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  np.ones([int(Nb/4),1])])
    xd = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  -np.ones([int(Nb/4),1])])
    xl = np.hstack([-np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1])
    xr = np.hstack([np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1])
    xcrack = np.hstack([-np.random.rand(int(Nb/4),1), np.zeros([int(Nb/4),1])])
    # 随机撒点时候，边界没有角点，这里要加上角点
    xc1 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]]) # 边界点效果不好，增加训练点
    xc2 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]])
    xc3 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]])
    Xb = np.concatenate((xu, xd, xl, xr, xcrack, xc1, xc2, xc3)) # 上下左右四个边界组装起来

    Xb = torch.tensor(Xb, device='cuda') # 转化成tensor
    
    Xf = torch.rand(Nf, 2)*2-1
    

    Xf1 = Xf[(Xf[:, 1]>0) & (torch.norm(Xf, dim=1)>=delta)] # 上区域点，去除内部多配的点
    Xf2 = Xf[(Xf[:, 1]<0) & (torch.norm(Xf, dim=1)>=delta)]  # 下区域点，去除内部多配的点
    
    Xf1 = torch.tensor(Xf1, requires_grad=True, device='cuda')
    Xf2 = torch.tensor(Xf2, requires_grad=True, device='cuda')
    
    return Xb, Xf1, Xf2

# for particular solution 
class particular(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(particular, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)
        
        #self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        

        self.a1 = torch.Tensor([0.1]).cuda()
        self.a2 = torch.Tensor([0.1])
        self.a3 = torch.Tensor([0.1])
        self.n = 1/self.a1.data.cuda()


        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+D_out)))
        
        
        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        y1 = torch.tanh(self.n*self.a1*self.linear1(x))
        y2 = torch.tanh(self.n*self.a1*self.linear2(y1))
        y3 = torch.tanh(self.n*self.a1*self.linear3(y2))
        y = self.n*self.a1*self.linear4(y3)
        return y


    
def pred(xy):
    '''
    

    Parameters
    ----------
    Nb : int
        the  number of boundary point.
    Nf : int
        the  number of internal point.

    Returns
    -------
    Xb : tensor
        The boundary points coordinates.
    Xf1 : tensor
        interior hole.
    Xf2 : tensor
        exterior region.
    '''
    pred = torch.zeros((len(xy), 1), device = 'cuda')
    # 上区域的预测，由于要用到不同的特解网络
    pred[(xy[:, 1]>0) ] = model_p1(xy[xy[:, 1]>0])
    # 下区域的预测，由于要用到不同的特解网络
    pred[xy[:, 1]<0] = model_p2(xy[xy[:, 1]<0])
    # 裂纹右端的预测，用到不同的特解网络，所以去了平局值，即是分片的交界面, 下面最好不要用，因为如果用下面的代码，就要用到裂纹处的判断，然而在裂纹处的判断是会出现paradox的
    pred[(xy[:, 1]==0) & (xy[:, 0]>0)] = (model_p1(xy[(xy[:, 1]==0) & (xy[:, 0]>0)]) + model_p2(xy[(xy[:, 1]==0) & (xy[:, 0]>0)]))/2 

    return pred    
    
def evaluate():
    N_test = 100
    x = np.linspace(-1, 1, N_test)
    y = np.linspace(-1, 1, N_test)
    x, y = np.meshgrid(x, y)
    xy_test = np.stack((x.flatten(), y.flatten()),1)
    xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction
    
    u_pred = pred(xy_test)
    u_pred = u_pred.data
    u_pred = u_pred.reshape(N_test, N_test)
    

    u_exact = np.zeros(x.shape)
    u_exact[y>0] = np.sqrt(x[y>0]**2+y[y>0]**2)*np.sqrt((1-x[y>0]/np.sqrt(x[y>0]**2+y[y>0]**2))/2)
    u_exact[y<0] = -np.sqrt(x[y<0]**2+y[y<0]**2)*np.sqrt((1-x[y<0]/np.sqrt(x[y<0]**2+y[y<0]**2))/2)

    u_exact = u_exact.reshape(x.shape)
    u_exact = torch.tensor(u_exact)
    error = torch.abs(u_pred - u_exact) # get the error in every points
    error_t = torch.norm(error)/torch.norm(u_exact) # get the total relative L2 error
    return error_t

# learning the homogenous network

model_p1 = particular(2, 200, 1).cuda()
model_p2 = particular(2, 200, 1).cuda()
criterion = torch.nn.MSELoss()
optim_h = torch.optim.Adam(params=chain(model_p1.parameters(), model_p2.parameters()), lr= 0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_h, milestones=[1000, 3000, 5000], gamma = 0.1)
loss_array = []
error_array = []
loss1_array = []
loss2_array = []  
lossii1_array = []  
lossii2_array = []   
lossi_array = []
nepoch_u0 = int(nepoch_u0)
eigvalues = []
start = time.time()
for epoch in range(nepoch_u0):
    if epoch ==1000:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0:        
        Xb, Xf1, Xf2 = train_data(256, 4096)
        Xi1 = interface1(5000)
        Xi2 = interface2(5000)
        Xi = interface(1000)
        
        Xb1 = Xb[Xb[:, 1]>=0] # 上边界点
        Xb2 = Xb[Xb[:, 1]<=0] # 下边界点
        target_b1 = torch.sqrt(Xb1[:,0]**2+Xb1[:, 1]**2)*torch.sqrt((1-Xb1[:,0]/torch.sqrt(Xb1[:,0]**2+Xb1[:,1]**2))/2)
        target_b1 = target_b1.unsqueeze(1)
        target_b2 = -torch.sqrt(Xb2[:,0]**2+Xb2[:, 1]**2)*torch.sqrt((1-Xb2[:,0]/torch.sqrt(Xb2[:,0]**2+Xb2[:,1]**2))/2)
        target_b2 = target_b2.unsqueeze(1)
    def closure():  


        
        u_pred1 = model_p1(Xf1)  
        u_pred2 = model_p2(Xf2) # 获得特解


        # 算一算积分值是多少，解析积分是0.8814
        du1dxy = grad(u_pred1, Xf1, torch.ones(Xf1.size()[0], 1).cuda(), create_graph=True)[0]
        du1dx = du1dxy[:, 0].unsqueeze(1)
        du1dy = du1dxy[:, 1].unsqueeze(1)

        du2dxy = grad(u_pred2, Xf2, torch.ones(Xf2.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du2dx = du2dxy[:, 0].unsqueeze(1)
        du2dy = du2dxy[:, 1].unsqueeze(1)
        
        J1 = (0.5 * torch.sum(du1dx**2 + du1dy**2)) * (2- 0.5*(delta)**2*np.pi)/len(Xf1) 
        J2 = (0.5 * torch.sum(du2dx**2 + du2dy**2)) * (2- 0.5*(delta)**2*np.pi)/len(Xf2) 
        
        u_i1 = model_p1(Xi1)  # 分别得到两个特解网络在内区域加密点的预测
        u_i2 = model_p2(Xi2)       
        
        du1dxyi = grad(u_i1, Xi1, torch.ones(Xi1.size()[0], 1).cuda(), create_graph=True)[0]
        du1dxi = du1dxyi[:, 0].unsqueeze(1)
        du1dyi = du1dxyi[:, 1].unsqueeze(1)

        du2dxyi = grad(u_i2, Xi2, torch.ones(Xi2.size()[0], 1).cuda(), create_graph=True)[0]
        du2dxi = du2dxyi[:, 0].unsqueeze(1)
        du2dyi = du2dxyi[:, 1].unsqueeze(1)  
        
        Jie1 = (0.5 * torch.sum(du1dxi**2 + du1dyi**2)) * 0.5 * (delta**2)*np.pi/len(Xi1) 
        Jie2 = (0.5 * torch.sum(du2dxi**2 + du2dyi**2)) * 0.5 * (delta**2)*np.pi/len(Xi2) 

        if delta != 0:
            J = J1 + J2 + Jie1 + Jie2
        if delta ==0:
            J = J1 + J2
        
        # 罚函数项，上下两个网络的本质边界的损失
        pred_b1 = model_p1(Xb1) # predict the boundary condition
        loss_b1 = criterion(pred_b1, target_b1)  
        pred_b2 = model_p2(Xb2) # predict the boundary condition
        loss_b2 = criterion(pred_b2, target_b2) 
        loss_b = loss_b1 + loss_b2# 边界的损失函数
        # 求交界面的损失函数，因为这里是两个特解网络
        pred_bi1 = model_p1(Xi) # predict the boundary condition
        pred_bi2 = model_p2(Xi) # predict the boundary condition
        loss_bi = criterion(pred_bi1, pred_bi2)  
        
        loss = J + penalty * (loss_b + loss_bi)
        
        error_t = evaluate()
        optim_h.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        error_array.append(error_t.data.cpu())
        loss1_array.append(J1.data.cpu())
        loss2_array.append(J2.data.cpu())
        lossii1_array.append(Jie1.data.cpu())
        lossii2_array.append(Jie2.data.cpu())
        if epoch%10==0:
            print(' epoch : %i, the loss : %f ,  J1 : %f , J2 : %f , Jie1 : %f, Jie2 : %f, Jb: %f, Ji: %f' % (epoch, loss.data, J1.data, J2.data, Jie1.data, Jie2.data, loss_b.data, loss_bi.data))
        return loss
    optim_h.step(closure)
    scheduler.step()
    
    
    
N_test = 100
x = np.linspace(-1, 1, N_test)
y = np.linspace(-1, 1, N_test)
x, y = np.meshgrid(x, y)
xy_test = np.stack((x.flatten(), y.flatten()),1)
xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction

u_pred = pred(xy_test)
u_pred = u_pred.data
u_pred = u_pred.reshape(N_test, N_test).cpu()

u_exact = np.zeros(x.shape)
u_exact[y>0] = np.sqrt(x[y>0]**2+y[y>0]**2)*np.sqrt((1-x[y>0]/np.sqrt(x[y>0]**2+y[y>0]**2))/2)
u_exact[y<0] = -np.sqrt(x[y<0]**2+y[y<0]**2)*np.sqrt((1-x[y<0]/np.sqrt(x[y<0]**2+y[y<0]**2))/2)
u_exact = torch.from_numpy(u_exact)
error = torch.abs(u_pred - u_exact) # get the error in every points
error_t = torch.norm(error)/torch.norm(u_exact) # get the total relative L2 error

# plot the prediction solution
fig = plt.figure(figsize=(20, 20))

plt.subplot(3, 3, 1)
Xb, Xf1, Xf2 = train_data(256, 4096)
Xi1 = interface1(1000)
Xi2 = interface2(1000)
plt.scatter(Xb.detach().cpu().numpy()[:, 0], Xb.detach().cpu().numpy()[:, 1], c='r', s = 0.1)
plt.scatter(Xf1.detach().cpu().numpy()[:, 0], Xf1.detach().cpu().numpy()[:, 1], c='b', s = 0.1)
plt.scatter(Xf2.detach().cpu().numpy()[:, 0], Xf2.detach().cpu().numpy()[:, 1], c='g', s = 0.1)
plt.scatter(Xi1.detach().cpu().numpy()[:, 0], Xi1.detach().cpu().numpy()[:, 1], c='m', s = 0.1)
plt.scatter(Xi2.detach().cpu().numpy()[:, 0], Xi2.detach().cpu().numpy()[:, 1], c='y', s = 0.1)

plt.subplot(3, 3, 2)
h2 = plt.contourf(x, y, u_pred.detach().numpy(), levels=100 ,cmap = 'jet')
plt.title('penalty prediction') 
plt.colorbar(h2)

plt.subplot(3, 3, 3)
h3 = plt.contourf(x, y, u_exact, levels=100 ,cmap = 'jet')
plt.title('exact solution') 
plt.colorbar(h3)

plt.subplot(3, 3, 4)
h4 = plt.contourf(x, y, error.detach().numpy(), levels=100 ,cmap = 'jet')
plt.title('absolute error') 
plt.colorbar(h4)

plt.subplot(3, 3, 5)
loss_array = np.array(loss_array)
loss_array = loss_array[loss_array<50]
plt.yscale('log')
plt.plot(loss_array)
plt.xlabel('the iteration')
plt.ylabel('loss')
plt.title('loss evolution')
 
plt.subplot(3, 3, 6)
error_array = np.array(error_array)
error_array = error_array[error_array<1]
plt.yscale('log')
plt.plot(error_array)
plt.xlabel('the iteration')
plt.ylabel('error')
plt.title('relative total error evolution') 

plt.subplot(3, 3, 7)
interx = torch.linspace(0, 1, N_test)[1:-1] # 获得interface的测试点
intery = torch.zeros(N_test)[1:-1]
inter = torch.stack((interx, intery),1)
inter = inter.requires_grad_(True).cuda() # 可以求导并且放入cuda中
pred_inter = pred(inter)
dudxyi = grad(pred_inter, inter, torch.ones(inter.size()[0], 1).cuda())[0]
dudyi = dudxyi[:, 1].unsqueeze(1) # 获得了交界面奇异应变
dudyi_e = 0.5/torch.sqrt(torch.norm(inter, dim=1))# 获得交界面的应变的精确值 
plt.plot(interx.cpu(), dudyi_e.detach().cpu(), label = 'exact')
plt.plot(interx.cpu(), dudyi.detach().cpu(), label = 'predict')
plt.xlabel('coordinate x')
plt.ylabel('strain e32')
plt.title('cpinn about e32 on interface')
plt.legend()



plt.subplot(3, 3, 8) # 精确解的二维频域图
fre = np.abs(np.fft.fftshift(np.fft.fft2(u_exact.cpu().numpy())))
plt.imshow(fre,                         #numpy array generating the image
            #color map used to specify colors
           interpolation='nearest'    #algorithm used to blend square colors; with 'nearest' colors will not be blended
          )
plt.colorbar()
plt.title('the frequency of the exact solution') 

plt.subplot(3, 3, 9) # 预测解的二维频域图
fre_p = np.abs(np.fft.fftshift(np.fft.fft2(u_pred.cpu().numpy())))
plt.imshow(fre_p,                         #numpy array generating the image
            #color map used to specify colors
           interpolation='nearest'    #algorithm used to blend square colors; with 'nearest' colors will not be blended
          )
plt.colorbar()
plt.title('the frequency of the pred solution') 
plt.suptitle("penalty_ritz")
plt.savefig('./picture/penalty_ritz.png')
plt.show()
print('the relative error is %f' % error_t.data)   
    
    
    