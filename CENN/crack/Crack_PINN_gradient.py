# 病态梯度下降，确定超参数的方法
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
from itertools import chain

torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
penalty = 1
delta = 0.0
train_p = 0
nepoch_u0 = 1500
a = 1

def NTK(predf1, predf2, predb1, predb2):
    # 输入强形式的内部1，内部2，和边界1，边界2
    
    paranum = sum(p.numel() for p in model_p1.parameters())-1 # 这里保证不同的分片网络结构一样，所以只需要一个就行了
    A1 = torch.zeros((len(predf1), paranum)).cuda() # 创建4个输出对参数的梯度矩阵
    A2 = torch.zeros((len(predf2), paranum)).cuda()
    A3 = torch.zeros((len(predb1), paranum)).cuda()
    A4 = torch.zeros((len(predb2), paranum)).cuda()
    for index,pred_e in enumerate(predf1):
        grad_e = grad(pred_e, model_p1.parameters(),retain_graph=True,  create_graph=True, allow_unused=True) # 获得每一个预测对参数的梯度,最后一个bias没有梯度，不知道为什么
        grad_one = torch.tensor([]).cuda() # 创建一个空的tensor
        for i in grad_e[:-1]: # 最后一个b不知道怎么回事，没有梯度
            grad_one = torch.cat((grad_one, i.flatten())) # 将每一个预测对参数的梯度预测都放入grad_one 中
        A1[index] = grad_one # 存入A中，A的行是样本个数，列是梯度个数
    K1 = torch.mm(A1, A1.T) # 矩阵的乘法组成K矩阵，这个K矩阵的获得是利用了矩阵的乘法，矩阵的乘法可以加速获得K矩阵，这里要避免使用for循环

    for index,pred_e in enumerate(predf2):
        grad_e = grad(pred_e, model_p2.parameters(),retain_graph=True,  create_graph=True, allow_unused=True) # 获得每一个预测对参数的梯度
        grad_one = torch.tensor([]).cuda() # 创建一个空的tensor
        for i in grad_e[:-1]:
            grad_one = torch.cat((grad_one, i.flatten())) # 将每一个预测对参数的梯度预测都放入grad_one 中
        A2[index] = grad_one # 存入A中，A的行是样本个数，列是梯度个数
    K2 = torch.mm(A2, A2.T) # 矩阵的乘法组成K矩阵，这个K矩阵的获得是利用了矩阵的乘法，矩阵的乘法可以加速获得K矩阵，这里要避免使用for循环

    for index,pred_e in enumerate(predb1):
        grad_e = grad(pred_e, model_p1.parameters(),retain_graph=True,  create_graph=True, allow_unused=True) # 获得每一个预测对参数的梯度
        grad_one = torch.tensor([]).cuda() # 创建一个空的tensor
        for i in grad_e[:-1]:
            grad_one = torch.cat((grad_one, i.flatten())) # 将每一个预测对参数的梯度预测都放入grad_one 中
        A3[index] = grad_one # 存入A中，A的行是样本个数，列是梯度个数
    K3 = torch.mm(A3, A3.T) # 矩阵的乘法组成K矩阵，这个K矩阵的获得是利用了矩阵的乘法，矩阵的乘法可以加速获得K矩阵，这里要避免使用for循环
    
    for index,pred_e in enumerate(predb2):
        grad_e = grad(pred_e, model_p2.parameters(),retain_graph=True,  create_graph=True, allow_unused=True) # 获得每一个预测对参数的梯度
        grad_one = torch.tensor([]).cuda() # 创建一个空的tensor
        for i in grad_e[:-1]:
            grad_one = torch.cat((grad_one, i.flatten())) # 将每一个预测对参数的梯度预测都放入grad_one 中
        A4[index] = grad_one # 存入A中，A的行是样本个数，列是梯度个数
    K4 = torch.mm(A4, A4.T) # 矩阵的乘法组成K矩阵，这个K矩阵的获得是利用了矩阵的乘法，矩阵的乘法可以加速获得K矩阵，这里要避免使用for循环
    
    tr1 = torch.trace(K1.data)
    tr2 = torch.trace(K2.data)
    tr3 = torch.trace(K3.data)    
    tr4 = torch.trace(K4.data)
    
    return tr1, tr2, tr3, tr4
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
        self.linear4 = torch.nn.Linear(H, D_out, bias=False) #  最后一层不知道为什么没有梯度，所以就不要最后一层的bias了
        
        
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
      #  torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)

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
    u_exact[y>0] = np.sqrt(np.sqrt(x[y>0]**2+y[y>0]**2))*np.sqrt((1-x[y>0]/np.sqrt(x[y>0]**2+y[y>0]**2))/2)
    u_exact[y<0] = -np.sqrt(np.sqrt(x[y<0]**2+y[y<0]**2))*np.sqrt((1-x[y<0]/np.sqrt(x[y<0]**2+y[y<0]**2))/2)

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
b1 = 1
b2 = 1
b3 = 1
b4 = 1
b5 = 1
b6 = 1
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
        target_b1 = torch.sqrt(torch.sqrt(Xb1[:,0]**2+Xb1[:, 1]**2))*torch.sqrt((1-Xb1[:,0]/torch.sqrt(Xb1[:,0]**2+Xb1[:,1]**2))/2)
        target_b1 = target_b1.unsqueeze(1)
        target_b2 = -torch.sqrt(torch.sqrt(Xb2[:,0]**2+Xb2[:, 1]**2))*torch.sqrt((1-Xb2[:,0]/torch.sqrt(Xb2[:,0]**2+Xb2[:,1]**2))/2)
        target_b2 = target_b2.unsqueeze(1)
    def closure():  
        global b1,b2,b3,b4,b5,b6
        u_pred1 = model_p1(Xf1)  
        u_pred2 = model_p2(Xf2) # 获得特解


        # 算一算积分值是多少，解析积分是0.8814
        du1dxy = grad(u_pred1, Xf1, torch.ones(Xf1.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du1dx = du1dxy[:, 0].unsqueeze(1)
        du1dy = du1dxy[:, 1].unsqueeze(1)
        du1dxxy = grad(du1dx, Xf1, torch.ones(Xf1.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du1dyxy = grad(du1dy, Xf1, torch.ones(Xf1.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du1dxx = du1dxxy[:, 0].unsqueeze(1)
        du1dyy = du1dyxy[:, 1].unsqueeze(1)        
        
        du2dxy = grad(u_pred2, Xf2, torch.ones(Xf2.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du2dx = du2dxy[:, 0].unsqueeze(1)
        du2dy = du2dxy[:, 1].unsqueeze(1)
        du2dxxy = grad(du2dx, Xf2, torch.ones(Xf2.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du2dyxy = grad(du2dy, Xf2, torch.ones(Xf2.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du2dxx = du2dxxy[:, 0].unsqueeze(1)
        du2dyy = du2dyxy[:, 1].unsqueeze(1) 
        
        J1 = torch.sum((du1dxx + du1dyy)**2)/len(du1dxx) 
        J2 = torch.sum((du2dxx + du2dyy)**2)/len(du2dxx)
        
        u_i1 = model_p1(Xi1)  # 分别得到两个特解网络在内区域加密点的预测
        u_i2 = model_p2(Xi2)       
        
        du1dxyi = grad(u_i1, Xi1, torch.ones(Xi1.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du1dxi = du1dxyi[:, 0].unsqueeze(1)
        du1dyi = du1dxyi[:, 1].unsqueeze(1)
        du1dxxyi = grad(du1dxi, Xi1, torch.ones(Xi1.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du1dyxyi = grad(du1dyi, Xi1, torch.ones(Xi1.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du1dxxi = du1dxxyi[:, 0].unsqueeze(1)
        du1dyyi = du1dyxyi[:, 1].unsqueeze(1)   

        du2dxyi = grad(u_i2, Xi2, torch.ones(Xi2.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du2dxi = du2dxyi[:, 0].unsqueeze(1)
        du2dyi = du2dxyi[:, 1].unsqueeze(1)  
        du2dxxyi = grad(du2dxi, Xi2, torch.ones(Xi2.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du2dyxyi = grad(du2dyi, Xi2, torch.ones(Xi2.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du2dxxi = du2dxxyi[:, 0].unsqueeze(1)
        du2dyyi = du2dyxyi[:, 1].unsqueeze(1)   
        
        Jie1 = torch.sum((du1dxxi + du1dyyi)**2)/len(du1dxxi) # 这里是NTK理论的损失函数
        Jie2 = torch.sum((du2dxxi + du2dyyi)**2)/len(du2dxxi) 

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
        
        # 计算交界面的位移导数，由于是y=0的交界面，所以我们的导数的损失只要两个网络在y方向的导数相同即可
        du1dxyii = grad(pred_bi1, Xi, torch.ones(Xi.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du2dxyii = grad(pred_bi2, Xi, torch.ones(Xi.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]     
        loss_bdi = criterion(du1dxyii[:, 1].unsqueeze(1), du2dxyii[:, 1].unsqueeze(1))   
        
        if epoch%100==0: # 将J1和J2的梯度最大值计算出来
            optim_h.zero_grad()
            J.backward(retain_graph=True) # 由于要分析两个梯度，所以分别对能量以及强制边界损失进行梯度下降
            weight_bias_tensor_J = torch.tensor([]).cuda()
            for name, param in model_p1.named_parameters(): # 将网络1的参数存起来,最后一层bias没有梯度，不知道为什么
               # print(name)
                #print(param.grad)
                weight_bias_tensor_J = torch.cat((weight_bias_tensor_J, param.grad.flatten().detach())) # 将权重和偏参数储存起来
            for name, param in model_p2.named_parameters(): # 将网络2的参数存起来
                #print(name)
                #print(param.grad)
                weight_bias_tensor_J = torch.cat((weight_bias_tensor_J, param.grad.flatten().detach())) # 将权重和偏参数储存起来
            # 将其中的权重W的梯度存起来，是奇数位置

            optim_h.zero_grad() # 边界1
            loss_b1.backward(retain_graph=True) # 由于要分析两个梯度，所以分别对能量以及强制边界损失进行梯度下降
            weight_bias_tensor_b1 = torch.tensor([]).cuda()
            for name, param in model_p1.named_parameters(): # 将网络1的参数存起来
               # print(name)
                #print(param.grad)
                weight_bias_tensor_b1 = torch.cat((weight_bias_tensor_b1, param.grad.flatten().detach())) # 将权重和偏参数储存起来
            for name, param in model_p2.named_parameters(): # 将网络2的参数存起来
                #print(name)
                #print(param.grad)
                weight_bias_tensor_b1 = torch.cat((weight_bias_tensor_b1, param.grad.flatten().detach())) # 将权重和偏参数储存起来
            # 将其中的权重W的梯度存起来，是奇数位置

            optim_h.zero_grad() # 边界b2
            loss_b2.backward(retain_graph=True) # 由于要分析两个梯度，所以分别对能量以及强制边界损失进行梯度下降
            weight_bias_tensor_b2 = torch.tensor([]).cuda()
            for name, param in model_p1.named_parameters(): # 将网络1的参数存起来
               # print(name)
                #print(param.grad)
                weight_bias_tensor_b2 = torch.cat((weight_bias_tensor_b2, param.grad.flatten().detach())) # 将权重和偏参数储存起来
            for name, param in model_p2.named_parameters(): # 将网络2的参数存起来
                #print(name)
                #print(param.grad)
                weight_bias_tensor_b2 = torch.cat((weight_bias_tensor_b2, param.grad.flatten().detach())) # 将权重和偏参数储存起来
            # 将其中的权重W的梯度存起来，是奇数位置

            optim_h.zero_grad()
            loss_bi.backward(retain_graph=True) # 由于要分析两个梯度，所以分别对能量以及强制边界损失进行梯度下降
            weight_bias_tensor_bi = torch.tensor([]).cuda()
            for name, param in model_p1.named_parameters(): # 将网络1的参数存起来
               # print(name)
                #print(param.grad)
                weight_bias_tensor_bi = torch.cat((weight_bias_tensor_bi, param.grad.flatten().detach())) # 将权重和偏参数储存起来
            for name, param in model_p2.named_parameters(): # 将网络2的参数存起来
                #print(name)
                #print(param.grad)
                weight_bias_tensor_bi = torch.cat((weight_bias_tensor_bi, param.grad.flatten().detach())) # 将权重和偏参数储存起来
            # 将其中的权重W的梯度存起来，是奇数位置

            optim_h.zero_grad()
            loss_bdi.backward(retain_graph=True) # 由于要分析两个梯度，所以分别对能量以及强制边界损失进行梯度下降
            weight_bias_tensor_bdi = torch.tensor([]).cuda()
            for name, param in model_p1.named_parameters(): # 将网络1的参数存起来
               # print(name)
                #print(param.grad)
                weight_bias_tensor_bdi = torch.cat((weight_bias_tensor_bdi, param.grad.flatten().detach())) # 将权重和偏参数储存起来
            for name, param in model_p2.named_parameters(): # 将网络2的参数存起来
                #print(name)
                #print(param.grad)
                weight_bias_tensor_bdi = torch.cat((weight_bias_tensor_bdi, param.grad.flatten().detach())) # 将权重和偏参数储存起来
            # 将其中的权重W的梯度存起来，是奇数位置            
            b3n = torch.abs(weight_bias_tensor_J).max()/torch.abs(weight_bias_tensor_b1).mean()
            b4n = torch.abs(weight_bias_tensor_J).max()/torch.abs(weight_bias_tensor_b2).mean()
            b5n = torch.abs(weight_bias_tensor_J).max()/torch.abs(weight_bias_tensor_bi).mean()
            b6n = torch.abs(weight_bias_tensor_J).max()/torch.abs(weight_bias_tensor_bdi).mean()
            # 用移动平均
            alfa = 0.9
            b3 = (1-alfa) * b3 + alfa * b3n
            b4 = (1-alfa) * b4 + alfa * b4n
            b5 = (1-alfa) * b5 + alfa * b5n
            b6 = (1-alfa) * b6 + alfa * b6n
        
        
        
        loss = b1 * J1 + b2 * J2 + b3 * loss_b1 + b4 * loss_b2  + b5 * loss_bi+ b6 * loss_bdi
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
            print(' epoch : %i, the loss : %f ,  J1 : %f , J2 : %f , Jie1 : %f, Jie2 : %f, Jb: %f, Ji: %f' % \
                  (epoch, loss.data, J1.data, J2.data, Jie1.data, Jie2.data, loss_b.data, (loss_bi+loss_bdi).data))
                
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
u_exact[y>0] = np.sqrt(np.sqrt(x[y>0]**2+y[y>0]**2))*np.sqrt((1-x[y>0]/np.sqrt(x[y>0]**2+y[y>0]**2))/2)
u_exact[y<0] = -np.sqrt(np.sqrt(x[y<0]**2+y[y<0]**2))*np.sqrt((1-x[y<0]/np.sqrt(x[y<0]**2+y[y<0]**2))/2)
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
plt.savefig('./picture/cpinn_ntk.png')
plt.show()
print('the relative error is %f' % error_t.data)   
    
    
    