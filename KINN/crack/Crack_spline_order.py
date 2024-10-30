# 这里我们测试一个KINN的scaling law
# 即是运行CPINN-KINN多次， 网络结构【2，5，1】， 【2，5，5,1】， 【2，5，5,5,1】,每一个网络结构我们试验不同的grid_size： 5， 10， 30

# 这里用里兹的方法研究裂纹问题，两个特解网络，不用可能位移场，突出
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
from itertools import chain
import sys
sys.path.append("..") 
from kan_efficiency import *

def setup_seed(seed):
# random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2024)

penalty = 1
delta = 0.0
train_p = 0
nepoch_u0 = 2500
a = 1

b1 = 1
b2 = 1
b3 = 50
b4 = 50
b5 = 10

def interface(Ni):
    '''
     生成交界面的随机点
    '''
    re = np.random.rand(Ni)
    theta = 0
    x = np.cos(theta) * re
    y = np.sin(theta) * re
    xi = np.stack([x, y], 1)
    xi = xi.astype(np.float32)
    xi = torch.tensor(xi,  requires_grad=True, device='cuda')
    return xi
def train_data(Nb, Nf):
    '''
    生成强制边界点，四周以及裂纹处
    生成上下的内部点
    '''
    

    xu = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  np.ones([int(Nb/4),1])]).astype(np.float32)
    xd = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  -np.ones([int(Nb/4),1])]).astype(np.float32)
    xl = np.hstack([-np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1]).astype(np.float32)
    xr = np.hstack([np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1]).astype(np.float32)
    xcrack = np.hstack([-np.random.rand(int(Nb/4),1), np.zeros([int(Nb/4),1])]).astype(np.float32)
    # 随机撒点时候，边界没有角点，这里要加上角点
    xc1 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]], dtype=np.float32) # 边界点效果不好，增加训练点
    xc2 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]], dtype=np.float32)
    xc3 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]], dtype=np.float32)
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


    
def pred(xy, model1, model2):
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
    pred[(xy[:, 1]>0) ] = model1(xy[xy[:, 1]>0])
    # 下区域的预测，由于要用到不同的特解网络
    pred[xy[:, 1]<0] = model2(xy[xy[:, 1]<0])
    
    return pred    


def evaluate(model1, model2):
    N_test = 100
    x = np.linspace(-1, 1, N_test).astype(np.float32)
    y = np.linspace(-1, 1, N_test).astype(np.float32)
    x, y = np.meshgrid(x, y)
    xy_test = np.stack((x.flatten(), y.flatten()),1)
    xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction
    
    u_pred = pred(xy_test, model1, model2)
    u_pred = u_pred.data
    u_pred = u_pred.reshape(N_test, N_test).cpu()
    

    u_exact = np.zeros(x.shape)
    u_exact[y>0] = np.sqrt(np.sqrt(x[y>0]**2+y[y>0]**2))*np.sqrt((1-x[y>0]/np.sqrt(x[y>0]**2+y[y>0]**2))/2)
    u_exact[y<0] = -np.sqrt(np.sqrt(x[y<0]**2+y[y<0]**2))*np.sqrt((1-x[y<0]/np.sqrt(x[y<0]**2+y[y<0]**2))/2)

    u_exact = u_exact.reshape(x.shape)
    u_exact = torch.from_numpy(u_exact)
    error = torch.abs(u_pred - u_exact) # get the error in every points
    error_t = torch.norm(error)/torch.norm(u_exact) # get the total relative L2 error
    return error_t

# learning the homogenous network
def run_scale_law(layers, order_num, epoch_num=2500):
    model_p1 = KAN(layers, base_activation=torch.nn.SiLU, grid_size=10, grid_range=[-1.0, 1.0], spline_order=order_num).cuda()
    model_p2 = KAN(layers, base_activation=torch.nn.SiLU, grid_size=10, grid_range=[-1.0, 1.0], spline_order=order_num).cuda()
    criterion = torch.nn.MSELoss()
    optim_h = torch.optim.Adam(params=chain(model_p1.parameters(), model_p2.parameters()), lr= 0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_h, milestones=[3000, 5000, 5000], gamma = 0.1)
    loss_array = []
    error_array = []
    loss1_array = []
    loss2_array = []    
    lossi_array = []
    nepoch_u0 = epoch_num
    start = time.time()

    
    for epoch in range(epoch_num):
        if epoch %1000==0:
            end = time.time()
            consume_time = end-start
            print('time is %f' % consume_time)
        if epoch%100 == 0:        
            Xb, Xf1, Xf2 = train_data(256, 4096)
            Xi = interface(1000)
            
            Xb1 = Xb[Xb[:, 1]>=0] # 上边界点
            Xb2 = Xb[Xb[:, 1]<=0] # 下边界点
            target_b1 = torch.sqrt(torch.sqrt(Xb1[:,0]**2+Xb1[:, 1]**2))*torch.sqrt((1-Xb1[:,0]/torch.sqrt(Xb1[:,0]**2+Xb1[:,1]**2))/2)
            target_b1 = target_b1.unsqueeze(1)
            target_b2 = -torch.sqrt(torch.sqrt(Xb2[:,0]**2+Xb2[:, 1]**2))*torch.sqrt((1-Xb2[:,0]/torch.sqrt(Xb2[:,0]**2+Xb2[:,1]**2))/2)
            target_b2 = target_b2.unsqueeze(1)
        def closure():  
            global b1,b2,b3,b4,b5
            u_pred1 = model_p1(Xf1)  
            u_pred2 = model_p2(Xf2) # 获得特解
    
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
            
            J1 = torch.sum((du1dxx + du1dyy)**2).mean()
            J2 = torch.sum((du2dxx + du2dyy)**2).mean()
            
    
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
            
            loss = b1 * J1 + b2 * J2 + b3 * loss_b1 + b4 * loss_b2  + b5 * (loss_bi+loss_bdi)
            error_t = evaluate(model_p1, model_p2)
            optim_h.zero_grad()
            loss.backward(retain_graph=True)
            loss_array.append(loss.data.cpu())
            error_array.append(error_t.data.cpu())
            if epoch%10==0:
                print(' epoch : %i, the loss : %f ,  J1 : %f , J2 : %f ,  Jb: %f, Ji: %f, error: %f' % \
                      (epoch, loss.data, J1.data, J2.data,  loss_b.data, (loss_bi+loss_bdi).data, error_t.data))# if epoch%100==0:
        optim_h.step(closure)
        scheduler.step()
        # 统计一下参数量级
        num_para = sum(p.numel() for p in model_p1.parameters())
    return num_para, error_array

if __name__ == "__main__":
    para_list = []
    error_list = []
    layers_list = [[2,5,1],[2,5,5,1],[2,5,5,5,1]]
    spline_order_list = [1,2,3,4,5]
    dict_scale_law = {}
    for layer in layers_list:
        for spline_num in spline_order_list:
            para_e, error_e = run_scale_law(layer, spline_num, epoch_num=3500)
            para_list.append(para_e)
            error_list.append(error_e)
    dict_scale_law = {
        'layers_list': layers_list,
        'spline_order_list': spline_order_list,
        'parameters': para_list,
        'errors': error_list
    }
    np.save( './results/Crack_spline_order.npy', dict_scale_law)
    loaded_data = np.load('./results/Crack_spline_order.npy', allow_pickle=True).item()
    print(loaded_data)
    
    
    
    