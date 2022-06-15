# incoperate L_adaptive and Kronechker neural network 
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
from sklearn.neighbors import KDTree
import time
from itertools import chain

def setup_seed(seed):
# random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

torch.set_default_tensor_type(torch.DoubleTensor) #  make default tensor float to double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # make default tensor float to double in CUDA
train_p = 0 # we already train the particular network in advance, 1: train particular network. 0: not train
nepoch = 2500 # the number of the literative number in CENN
error_total = {} # 
setup_seed(1)


def interface(Ni):
# get the coordinate of the interface point of the crack [0, 0] to [1, 0]
    re = np.random.rand(Ni)
    theta = 0
    x = np.cos(theta) * re
    y = np.sin(theta) * re
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi, requires_grad=True, device='cuda')
    return xi
def train_data(Nb, Nf):
    '''
    

    Parameters
    ----------
    Nb : init
        the number of the essential boundary.
    Nf : int
        the number of the domain .

    Returns
    -------
    Xb : tensor, size [Nb, 2]
        the coordinate of the essential boundary.
    Xf1 : tensor, size [Nf/2, 2]
        the coordinate of the top region in the crack.
    Xf2 : tensor, size [Nf/2, 2]
        the coordinate of the down region in the crack

    '''
    
    # generate the essential points
    xu = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  np.ones([int(Nb/4),1])])
    xd = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  -np.ones([int(Nb/4),1])])
    xl = np.hstack([-np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1])
    xr = np.hstack([np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1])
    xcrack = np.hstack([-np.random.rand(int(Nb/4),1), np.zeros([int(Nb/4),1])])
    
    xc1 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]]) # add the corner point of rectangle
    xc2 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]])
    xc3 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]])
    Xb = np.concatenate((xu, xd, xl, xr, xcrack, xc1, xc2, xc3)) # get the boundary points

    Xb = torch.tensor(Xb, device='cuda') # change to tensor
    # generate the points in domain
    Xf = torch.rand(Nf, 2)*2-1
    
    Xf1 = Xf[Xf[:, 1]>0] # top region
    Xf2 = Xf[Xf[:, 1]<0] # down region
    
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

        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H))) # Xavier initialization for weight of neural network
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+D_out)))
             
        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1) # initialization for bias of the neural network
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)

    def forward(self, x):
        """
        the forword NN for PINN
        """

        y1 = torch.tanh(self.linear1(x))
        y2 = torch.tanh(self.linear2(y1))
        y3 = torch.tanh(self.linear3(y2))
        y = self.linear4(y3)
        return y

# general neural network
class general(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(general, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)
        
        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))

        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1) # initialization for bias of the neural network
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)
    def forward(self, x):
        yt = x
        y1 = torch.tanh(self.linear1(yt))
        y2 = torch.tanh(self.linear2(y1))
        y3 = torch.tanh(self.linear3(y2)) 
        y4 = torch.tanh(self.linear4(y3)) 
        y =  self.linear5(y4)
        return y
    
    
class general_L(torch.nn.Module): # local adaptive activation function
    def __init__(self, D_in, H, D_out):
        super(general_L, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)
        
        self.a1 = torch.nn.Parameter(torch.Tensor([0.1]).cuda())
        self.a2 = torch.nn.Parameter(torch.Tensor([0.1]).cuda())
        self.a3 = torch.nn.Parameter(torch.Tensor([0.1]).cuda())
        self.a4 = torch.nn.Parameter(torch.Tensor([0.1]).cuda())
        self.n1 = 1/self.a1.data.cuda()
        self.n2 = 1/self.a2.data.cuda()
        self.n3 = 1/self.a3.data.cuda()
        self.n4 = 1/self.a4.data.cuda()
        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))

        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1) # initialization for bias of the neural network
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)
    def forward(self, x):
        yt = x
        y1 = torch.tanh(self.n1*self.a1*self.linear1(yt))
        y2 = torch.tanh(self.n2*self.a2*self.linear2(y1))
        y3 = torch.tanh(self.n3*self.a3*self.linear3(y2)) 
        y4 = torch.tanh(self.n4*self.a4*self.linear4(y3)) 
        y =  self.linear5(y4)
        return y
    
class general_Rowdy3(torch.nn.Module): # Rowdy k=3
    def __init__(self, D_in, H, D_out):
        super(general_Rowdy3, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)
        # first layer number, second k number
        self.a11 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        self.a12 = torch.nn.Parameter(torch.Tensor([0.0]).cuda())
        self.a13 = torch.nn.Parameter(torch.Tensor([0.0]).cuda())
        
        self.a21 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        self.a22 = torch.nn.Parameter(torch.Tensor([0.0]).cuda())
        self.a23 = torch.nn.Parameter(torch.Tensor([0.0]).cuda())
        
        self.a31 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        self.a32 = torch.nn.Parameter(torch.Tensor([0.0]).cuda())
        self.a33 = torch.nn.Parameter(torch.Tensor([0.0]).cuda())
        
        
        self.a41 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        self.a42 = torch.nn.Parameter(torch.Tensor([0.0]).cuda())
        self.a43 = torch.nn.Parameter(torch.Tensor([0.0]).cuda())
        
        self.w11 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        self.w12 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        self.w13 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        
        self.w21 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        self.w22 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        self.w23 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        
        self.w31 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        self.w32 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        self.w33 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        
        
        self.w41 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        self.w42 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        self.w43 = torch.nn.Parameter(torch.Tensor([1.0]).cuda())
        
        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))

        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1) # initialization for bias of the neural network
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)
    def forward(self, x):
        yt = x
        y1 = self.a11*torch.tanh(self.w11*self.linear1(yt))+self.a12*self.Rowdy(self.w12*self.linear1(yt), 2, 1)+self.a13*self.Rowdy(self.w13*self.linear1(yt), 3, 1)
        y2 = self.a21*torch.tanh(self.w21*self.linear2(y1))+self.a22*self.Rowdy(self.w22*self.linear2(y1), 2, 1)+self.a23*self.Rowdy(self.w23*self.linear2(y1), 3, 1)
        y3 = self.a31*torch.tanh(self.w31*self.linear3(y2))+self.a32*self.Rowdy(self.w32*self.linear3(y2), 2, 1)+self.a33*self.Rowdy(self.w33*self.linear3(y2), 3, 1)
        y4 = self.a41*torch.tanh(self.w41*self.linear4(y3))+self.a42*self.Rowdy(self.w42*self.linear4(y3), 2, 1)+self.a43*self.Rowdy(self.w43*self.linear4(y3), 3, 1)
        y =  self.linear5(y4)
        return y
    def Rowdy(self, x, k, n):
        '''
        

        Parameters
        ----------
        x : tensor
            the output need to be Rowdy.
        k : int
            Rowdy k.
        n : int
            factor used to accelerate.

        Returns
        -------
        tensor
            Rowdy activation function.

        '''
        return n*torch.sin((k-1)*n*x)
    
def pred(xy):
    # pred the displacement of the crack
    pred = torch.zeros((len(xy), 1), device = 'cuda')
    # the prediction of the top region of admissible function
    pred[(xy[:, 1]>0) ] = model_p1(xy[xy[:, 1]>0]) + \
        RBF(xy[xy[:, 1]>0]) * model_g(xy[xy[:, 1]>0])
    # the prediction of the down region of admissible function
    pred[xy[:, 1]<0] = model_p2(xy[xy[:, 1]<0]) + \
        RBF(xy[xy[:, 1]<0]) * model_g(xy[xy[:, 1]<0])
    return pred    

def evaluate():
    # evaluate the relative L2 error of the CENN
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

    u_exact = u_exact.reshape(x.shape)
    u_exact = torch.from_numpy(u_exact)
    error = torch.abs(u_pred - u_exact) # get the error in every points
    error_t = torch.norm(error)/torch.norm(u_exact) # get the total relative L2 error
    return error_t


# learning the boundary solution
# step 1: get the boundary train data and target for boundary condition

if train_p == 1:
    model_p1 = particular(2, 10, 1).cuda() # define two particular NN, since it is discontinuous in the crack
    model_p2 = particular(2, 10, 1).cuda()
    tol_p = 0.0001
    loss_bn = 100
    epoch_b = 0
    criterion = torch.nn.MSELoss()
    optimp = torch.optim.Adam(params=chain(model_p1.parameters(), model_p2.parameters()), lr= 0.0005)
    loss_b_array = []
    loss_b1_array = []
    loss_b2_array = []
    loss_bi_array = []
    loss_bn_array = []
    # training the particular network for particular solution on the bonudaray
    while loss_bn>tol_p:
        
        if epoch_b%10 == 0:
            Xb, Xf1, Xf2 = train_data(256, 4096)
            Xi = interface(1000)
            Xb1 = Xb[Xb[:, 1]>=0] # up region boundary
            Xb2 = Xb[Xb[:, 1]<=0] # down region boundary
            target_b1 = torch.sqrt(torch.sqrt(Xb1[:,0]**2+Xb1[:, 1]**2))*torch.sqrt((1-Xb1[:,0]/torch.sqrt(Xb1[:,0]**2+Xb1[:,1]**2))/2)
            target_b1 = target_b1.unsqueeze(1)
            target_b2 = -torch.sqrt(torch.sqrt(Xb2[:,0]**2+Xb2[:, 1]**2))*torch.sqrt((1-Xb2[:,0]/torch.sqrt(Xb2[:,0]**2+Xb2[:,1]**2))/2)
            target_b2 = target_b2.unsqueeze(1)
        epoch_b = epoch_b + 1
        def closure():  
            pred_b1 = model_p1(Xb1) # predict the boundary condition
            loss_b1 = criterion(pred_b1, target_b1)  
            pred_b2 = model_p2(Xb2) # predict the boundary condition
            loss_b2 = criterion(pred_b2, target_b2) 
            loss_b = loss_b1 + loss_b2# the loss function of boundary
            # obtain the loss on the interface, since there are two particular network
            pred_bi1 = model_p1(Xi) # predict the boundary condition
            pred_bi2 = model_p2(Xi) # predict the boundary condition
            loss_bi = criterion(pred_bi1, pred_bi2)             
            
            optimp.zero_grad()
            loss_bn = loss_b + loss_bi # the loss function including the interface and boundary loss
            loss_bn.backward()
            loss_b_array.append(loss_b.data)
            loss_b1_array.append(loss_b1.data)
            loss_b2_array.append(loss_b2.data)
            loss_bi_array.append(loss_bi.data)
            loss_bn_array.append(loss_bn.data)
            if epoch_b%10==0:
                print('trianing particular network : the number of epoch is %i, the loss1 is %f, the loss2 is %f, the lossi is %f' % (epoch_b, loss_b1.data, loss_b2.data, loss_bi.data))
            return loss_bn
        optimp.step(closure)
        loss_bn = loss_bn_array[-1]
    torch.save(model_p1, './particular1_nn')
    torch.save(model_p2, './particular2_nn')
model_p1 = torch.load('particular1_nn')
model_p2 = torch.load('particular2_nn')

# learning the distance neural network
def RBF(x):
    d_total_t = torch.from_numpy(d_total).unsqueeze(1).cuda() # d_total is the reference points (center points called in our papar)
    w_t = torch.from_numpy(w).cuda() # the weight of RBF
    x_l = x.unsqueeze(0).repeat(len(d_total_t), 1, 1)
    R2 = torch.norm(d_total_t - x_l, dim=2) # get the distance between the reference points and the input points
    y = torch.mm(torch.exp(-gama*R2.T), w_t) # get the nearest distance of the input points
    return y

# obtain 4 boundary points
n_d = 10 # the number of the points on each essential boundary of the crack
n_dom = 5
gama = 0.5
ep = np.linspace(-1, 1, n_d) # uniform from -1 to 1
# obtain the essential points of the rectangle boundary
ep1 = np.zeros((n_d, 2))
ep1[:, 0], ep1[:, 1] = ep, 1
ep2 = np.zeros((n_d, 2))
ep2[:, 0], ep2[:, 1] = ep, -1
ep3 = np.zeros((n_d, 2))
ep3[:, 0], ep3[:, 1] = -1, ep
ep4 = np.zeros((n_d, 2))
ep4[:, 0], ep4[:, 1] = 1, ep
ep5 = np.zeros((n_d, 2))
ep5[:, 0], ep5[:, 1] = ep/2-0.5, 0
points_d = np.concatenate((ep1, ep2, ep3, ep4, ep5)) # get all the boundary points
points_d = np.unique(points_d, axis=0) # delete the same points
kdt = KDTree(points_d, metric='euclidean') # make the essential boundary to a object


domx = np.linspace(-1, 1, n_dom)[1:-1]
domy = np.linspace(-1, 1, n_dom)[1:-1]
domx, domy = np.meshgrid(domx, domy)
domxy = np.stack((domx.flatten(), domy.flatten()), 1)
domxy = domxy[(domxy[:, 1]!=0)|(domxy[:, 0]>0)]
#domxy = np.unique(domxy, axis=0)
d_dir, _ = kdt.query(points_d, k=1, return_distance = True)
d_dom, _ = kdt.query(domxy, k=1, return_distance = True)
# concatenate the essential and domain points
d_total = np.concatenate((points_d, domxy))
#d_total = np.unique(d_total, axis=0)
# obtain K matrix after get the distance function of the points
dx = d_total[:, 0][:, np.newaxis] - d_total[:, 0][np.newaxis, :]
dy = d_total[:, 1][:, np.newaxis] - d_total[:, 1][np.newaxis, :]
R2 = np.sqrt(dx**2+dy**2)
K = np.exp(-0.5*R2)
b = np.concatenate((d_dir, d_dom))
w = np.dot(np.linalg.inv(K),b)



n_test = 21 # get the test points: n_test**2
domx_t = np.linspace(-1, 1, n_test)
domy_t = np.linspace(-1, 1, n_test)
domx_t, domy_t = np.meshgrid(domx_t, domy_t)
domxy_t = np.stack((domx_t.flatten(), domy_t.flatten()), 1)
domxy_t= torch.from_numpy(domxy_t).requires_grad_(True).cuda() 

dis_RBF = RBF(domxy_t)
# plot the contourf of distance of the RBF
dis_plot_RBF = dis_RBF.data.cpu().numpy().reshape(n_test, n_test)
h1 = plt.contourf(domx_t, domy_t, dis_plot_RBF,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('RBF distance function', size = 7)
plt.show()
#%% CENN origin
# learn the general NN of CENN
model_g = general(2, 20, 1).cuda()
criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(params=model_g.parameters(), lr= 0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[1000, 3000, 5000], gamma = 0.1)
loss_array = []
error_array = []
loss1_array = []
loss2_array = []  
nepoch = int(nepoch)
start = time.time()




for epoch in range(nepoch):
    if epoch ==1000:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0:
        Xb, Xf1, Xf2 = train_data(100, 4096) # obtain the essential boundary and domain points
        Xi = interface(1000) # get the interface not the crack [0, 0] to [1, 0]
    def closure():  
        u_h1 = model_g(Xf1) 
        u_h2 = model_g(Xf2)
        
        u_p1 = model_p1(Xf1)  
        u_p2 = model_p2(Xf2) # get the particular solution
        # construct the admissible function
        u_pred1 = u_p1 + RBF(Xf1)*u_h1
        u_pred2 = u_p2 + RBF(Xf2)*u_h2
        
        du1dxy = grad(u_pred1, Xf1, torch.ones(Xf1.size()[0], 1).cuda(), create_graph=True)[0]
        du1dx = du1dxy[:, 0].unsqueeze(1)
        du1dy = du1dxy[:, 1].unsqueeze(1)

        du2dxy = grad(u_pred2, Xf2, torch.ones(Xf2.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du2dx = du2dxy[:, 0].unsqueeze(1)
        du2dy = du2dxy[:, 1].unsqueeze(1)
        J1 = (0.5 * torch.sum(du1dx**2 + du1dy**2)) * (2/len(Xf1))        
        J2 = (0.5 * torch.sum(du2dx**2 + du2dy**2)) * (2/len(Xf2)) 

        J = J1 + J2
            
        
        loss = J
        error_t = evaluate()
        optim.zero_grad()
        loss.backward()
        loss_array.append(loss.data.cpu())
        error_array.append(error_t.data.cpu())
        loss1_array.append(J1.data.cpu())
        loss2_array.append(J2.data.cpu())
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, J : %f,  J1 : %f, J2 : %f ' % (epoch, loss.data, J.data, J1.data, J2.data))
        return loss
    optim.step(closure)
    scheduler.step()
    
N_test = 100
x = np.linspace(-1, 1, N_test)
y = np.linspace(-1, 1, N_test)
x, y = np.meshgrid(x, y)
xy_test = np.stack((x.flatten(), y.flatten()),1)
xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction

u_pred = pred(xy_test)
u_pred = u_pred.data
u_pred_energy = u_pred.reshape(N_test, N_test).cpu()

u_exact = np.zeros(x.shape)
u_exact[y>0] = np.sqrt(np.sqrt(x[y>0]**2+y[y>0]**2))*np.sqrt((1-x[y>0]/np.sqrt(x[y>0]**2+y[y>0]**2))/2)
u_exact[y<0] = -np.sqrt(np.sqrt(x[y<0]**2+y[y<0]**2))*np.sqrt((1-x[y<0]/np.sqrt(x[y<0]**2+y[y<0]**2))/2)
u_exact = torch.from_numpy(u_exact)
error_energy  = torch.abs(u_pred_energy  - u_exact) # get the error in every points
error_energy_t = torch.norm(error_energy )/torch.norm(u_exact) # get the total relative L2 error

loss_array_energy  = np.array(loss_array)
#loss_array_energy  = loss_array_energy [loss_array_energy <50]

error_array_energy  = np.array(error_array)
#error_array_energy  = error_array_energy[error_array_energy ]


print('the relative error of cenn is %f' % error_energy_t.data)   


#%% CENN with local adaptive
# learn the general NN of CENN
model_g_l = general_L(2, 20, 1).cuda()
criterion = torch.nn.MSELoss()
optim_l = torch.optim.Adam(params=model_g_l.parameters(), lr= 0.001)
scheduler_l = torch.optim.lr_scheduler.MultiStepLR(optim_l, milestones=[1000, 3000, 5000], gamma = 0.1)
loss_array_l = []
error_array_l = []
loss1_array_l = []
loss2_array_l = []  
nepoch = int(nepoch)
start = time.time()

def pred_l(xy):
    # pred the displacement of the crack
    pred = torch.zeros((len(xy), 1), device = 'cuda')
    # the prediction of the top region of admissible function
    pred[(xy[:, 1]>0) ] = model_p1(xy[xy[:, 1]>0]) + \
        RBF(xy[xy[:, 1]>0]) * model_g_l(xy[xy[:, 1]>0])
    # the prediction of the down region of admissible function
    pred[xy[:, 1]<0] = model_p2(xy[xy[:, 1]<0]) + \
        RBF(xy[xy[:, 1]<0]) * model_g_l(xy[xy[:, 1]<0])
    return pred    

def evaluate_l():
    # evaluate the relative L2 error of the CENN
    N_test = 100
    x = np.linspace(-1, 1, N_test)
    y = np.linspace(-1, 1, N_test)
    x, y = np.meshgrid(x, y)
    xy_test = np.stack((x.flatten(), y.flatten()),1)
    xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction
    
    u_pred = pred_l(xy_test)
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


for epoch in range(nepoch):
    if epoch ==1000:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0:
        Xb, Xf1, Xf2 = train_data(100, 4096) # obtain the essential boundary and domain points
        Xi = interface(1000) # get the interface not the crack [0, 0] to [1, 0]
    def closure():  
        u_h1 = model_g_l(Xf1) 
        u_h2 = model_g_l(Xf2)
        
        u_p1 = model_p1(Xf1)  
        u_p2 = model_p2(Xf2) # get the particular solution
        # construct the admissible function
        u_pred1 = u_p1 + RBF(Xf1)*u_h1
        u_pred2 = u_p2 + RBF(Xf2)*u_h2
        
        du1dxy = grad(u_pred1, Xf1, torch.ones(Xf1.size()[0], 1).cuda(), create_graph=True)[0]
        du1dx = du1dxy[:, 0].unsqueeze(1)
        du1dy = du1dxy[:, 1].unsqueeze(1)

        du2dxy = grad(u_pred2, Xf2, torch.ones(Xf2.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du2dx = du2dxy[:, 0].unsqueeze(1)
        du2dy = du2dxy[:, 1].unsqueeze(1)
        J1 = (0.5 * torch.sum(du1dx**2 + du1dy**2)) * (2/len(Xf1))        
        J2 = (0.5 * torch.sum(du2dx**2 + du2dy**2)) * (2/len(Xf2)) 

        J = J1 + J2
            
        Ja = 1/((torch.exp(model_g_l.a1)+torch.exp(model_g_l.a2)+torch.exp(model_g_l.a3)+torch.exp(model_g_l.a4))/4)
        loss = J + Ja 
        error_t = evaluate_l()
        optim_l.zero_grad()
        loss.backward()
        loss_array_l.append(loss.data.cpu())
        error_array_l.append(error_t.data.cpu())
        loss1_array_l.append(J1.data.cpu())
        loss2_array_l.append(J2.data.cpu())
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, J : %f,  J1 : %f, J2 : %f ' % (epoch, loss.data, J.data, J1.data, J2.data))
        return loss
    optim_l.step(closure)
    scheduler_l.step()
    
N_test = 100
x = np.linspace(-1, 1, N_test)
y = np.linspace(-1, 1, N_test)
x, y = np.meshgrid(x, y)
xy_test = np.stack((x.flatten(), y.flatten()),1)
xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction

u_pred_l = pred_l(xy_test)
u_pred_l = u_pred_l.data
u_pred_energy_l = u_pred_l.reshape(N_test, N_test).cpu()

u_exact = np.zeros(x.shape)
u_exact[y>0] = np.sqrt(np.sqrt(x[y>0]**2+y[y>0]**2))*np.sqrt((1-x[y>0]/np.sqrt(x[y>0]**2+y[y>0]**2))/2)
u_exact[y<0] = -np.sqrt(np.sqrt(x[y<0]**2+y[y<0]**2))*np.sqrt((1-x[y<0]/np.sqrt(x[y<0]**2+y[y<0]**2))/2)
u_exact = torch.from_numpy(u_exact)
error_energy_l  = torch.abs(u_pred_energy_l  - u_exact) # get the error in every points
error_energy_t_l = torch.norm(error_energy_l )/torch.norm(u_exact) # get the total relative L2 error

loss_array_energy_l  = np.array(loss_array_l)
#loss_array_energy  = loss_array_energy [loss_array_energy <50]

error_array_energy_l  = np.array(error_array_l)
#error_array_energy  = error_array_energy[error_array_energy ]


print('the relative error of cenn with local adaptive is %f' % error_energy_t_l.data)   

#%% CENN with Rowdy
# learn the general NN of CENN
model_g_Rowdy3 = general_Rowdy3(2, 20, 1).cuda()
criterion = torch.nn.MSELoss()
optim_Rowdy3 = torch.optim.Adam(params=model_g_Rowdy3.parameters(), lr= 0.001)
scheduler_Rowdy3 = torch.optim.lr_scheduler.MultiStepLR(optim_Rowdy3, milestones=[1000, 3000, 5000], gamma = 0.1)
loss_array_Rowdy3 = []
error_array_Rowdy3 = []
loss1_array_Rowdy3 = []
loss2_array_Rowdy3 = []  
nepoch = int(nepoch)
start = time.time()

def pred_Rowdy3(xy):
    # pred the displacement of the crack
    pred = torch.zeros((len(xy), 1), device = 'cuda')
    # the prediction of the top region of admissible function
    pred[(xy[:, 1]>0) ] = model_p1(xy[xy[:, 1]>0]) + \
        RBF(xy[xy[:, 1]>0]) * model_g_Rowdy3(xy[xy[:, 1]>0])
    # the prediction of the down region of admissible function
    pred[xy[:, 1]<0] = model_p2(xy[xy[:, 1]<0]) + \
        RBF(xy[xy[:, 1]<0]) * model_g_Rowdy3(xy[xy[:, 1]<0])
    return pred    

def evaluate_Rowdy3():
    # evaluate the relative L2 error of the CENN
    N_test = 100
    x = np.linspace(-1, 1, N_test)
    y = np.linspace(-1, 1, N_test)
    x, y = np.meshgrid(x, y)
    xy_test = np.stack((x.flatten(), y.flatten()),1)
    xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction
    
    u_pred = pred_Rowdy3(xy_test)
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


for epoch in range(nepoch):
    if epoch ==1000:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0:
        Xb, Xf1, Xf2 = train_data(100, 4096) # obtain the essential boundary and domain points
        Xi = interface(1000) # get the interface not the crack [0, 0] to [1, 0]
    def closure():  
        u_h1 = model_g_Rowdy3(Xf1) 
        u_h2 = model_g_Rowdy3(Xf2)
        
        u_p1 = model_p1(Xf1)  
        u_p2 = model_p2(Xf2) # get the particular solution
        # construct the admissible function
        u_pred1 = u_p1 + RBF(Xf1)*u_h1
        u_pred2 = u_p2 + RBF(Xf2)*u_h2
        
        du1dxy = grad(u_pred1, Xf1, torch.ones(Xf1.size()[0], 1).cuda(), create_graph=True)[0]
        du1dx = du1dxy[:, 0].unsqueeze(1)
        du1dy = du1dxy[:, 1].unsqueeze(1)

        du2dxy = grad(u_pred2, Xf2, torch.ones(Xf2.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du2dx = du2dxy[:, 0].unsqueeze(1)
        du2dy = du2dxy[:, 1].unsqueeze(1)
        J1 = (0.5 * torch.sum(du1dx**2 + du1dy**2)) * (2/len(Xf1))        
        J2 = (0.5 * torch.sum(du2dx**2 + du2dy**2)) * (2/len(Xf2)) 

        J = J1 + J2

        loss = J 
        error_t = evaluate_Rowdy3()
        optim_Rowdy3.zero_grad()
        loss.backward()
        loss_array_Rowdy3.append(loss.data.cpu())
        error_array_Rowdy3.append(error_t.data.cpu())
        loss1_array_Rowdy3.append(J1.data.cpu())
        loss2_array_Rowdy3.append(J2.data.cpu())
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, J : %f,  J1 : %f, J2 : %f ' % (epoch, loss.data, J.data, J1.data, J2.data))
        return loss
    optim_Rowdy3.step(closure)
    scheduler_Rowdy3.step()
    
N_test = 100
x = np.linspace(-1, 1, N_test)
y = np.linspace(-1, 1, N_test)
x, y = np.meshgrid(x, y)
xy_test = np.stack((x.flatten(), y.flatten()),1)
xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction

u_pred_Rowdy3 = pred_Rowdy3(xy_test)
u_pred_Rowdy3 = u_pred_Rowdy3.data
u_pred_energy_Rowdy3 = u_pred_Rowdy3.reshape(N_test, N_test).cpu()

u_exact = np.zeros(x.shape)
u_exact[y>0] = np.sqrt(np.sqrt(x[y>0]**2+y[y>0]**2))*np.sqrt((1-x[y>0]/np.sqrt(x[y>0]**2+y[y>0]**2))/2)
u_exact[y<0] = -np.sqrt(np.sqrt(x[y<0]**2+y[y<0]**2))*np.sqrt((1-x[y<0]/np.sqrt(x[y<0]**2+y[y<0]**2))/2)
u_exact = torch.from_numpy(u_exact)
error_energy_Rowdy3  = torch.abs(u_pred_energy_Rowdy3  - u_exact) # get the error in every points
error_energy_t_Rowdy3 = torch.norm(error_energy_Rowdy3 )/torch.norm(u_exact) # get the total relative L2 error

loss_array_energy_Rowdy3  = np.array(loss_array_Rowdy3)
#loss_array_energy  = loss_array_energy [loss_array_energy <50]

error_array_energy_Rowdy3  = np.array(error_array_Rowdy3)
#error_array_energy  = error_array_energy[error_array_energy ]


print('the relative error of cenn with Rowdy3 is %f' % error_energy_t_Rowdy3.data)  


#  plotting for comparison among CENN, CENN adapive, CENN Rowdy
#%%
fig = plt.figure(dpi=1000, figsize=(15, 6.5)) #plot the loss and L2 relative error comparing 3 method decribed above
plt.subplot(1, 2, 1)
plt.yscale('log')
plt.axhline(y=0.8814, color='r', ls = '--') # plot the exact solution.
plt.plot(loss_array_energy)
plt.plot(loss_array_energy_l, ':')
plt.plot(loss_array_energy_Rowdy3, '-.')
plt.legend(['Exact', 'Cenn', 'CENN_L_LAAF', 'CENN_Rowdy3'], loc = 'upper right', fontsize = 15)
plt.xlabel('Iteration', fontsize = 15)
plt.ylabel('Loss', fontsize = 15)
plt.title('Loss: Adaptive activation functions', fontsize = 20) 

plt.subplot(1, 2, 2)
plt.yscale('log')
plt.plot(error_array_energy)
plt.plot(error_array_energy_l, ls = '--')
plt.plot(error_array_energy_Rowdy3, ':')
plt.legend(['Cenn', 'CENN_L_LAAF', 'CENN_Rowdy3'], loc = 'upper right', fontsize = 15)
plt.xlabel('Iteration', fontsize = 15)
plt.ylabel('${\mathcal{L}_2}$ Error', fontsize = 15)
plt.title('${\mathcal{L}_2}$ Error: Adaptive activation functions', fontsize = 20) 
plt.savefig('picture/crack_cenn_adaptive.pdf', bbox_inches = 'tight')
plt.show()