# different location comparison among data-driven, CPINN and CENN
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
from sklearn.neighbors import KDTree
import time
from itertools import chain

import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei'] # matplotlib for displaying Chinese

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True

torch.set_default_tensor_type(torch.DoubleTensor) # change the float(cpu default tensor) to double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # change the float(gpu default tensor) to double
penalty = 100
delta = 0.0
train_p = 0
nepoch_u0 = 2500 # data-driven epoch
nepoch_u1 = 2500 # CPINN epoch
nepoch_u2 = 2500 # CENN epoch
a = 1

setup_seed(55)
def interface1(Ni1):# up region: the gradient on the interface is large, so we attribute more point around the crack tip
    '''
     generate more point around the up region of crack tip in the shape of half circle 
    '''
    
    theta = np.pi*np.random.rand(Ni1) # angle from 0 to pi
    rp = delta*np.random.rand(Ni1) # radius 0 to delta
    x = np.cos(theta) * rp
    y = np.sin(theta) * rp
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi, requires_grad=True, device='cuda')
    return xi
def interface2(Ni1): # down region 
    '''
     generate more point around the down region of crack tip in the shape of half circle 
    '''
    
    theta = -np.pi*np.random.rand(Ni1) # angle from 0 to -pi
    rp = delta*np.random.rand(Ni1) # radius 0 to delta
    x = np.cos(theta) * rp
    y = np.sin(theta) * rp
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi, requires_grad=True, device='cuda')
    return xi
def interface(Ni):
    '''
     generate random points on the interface
    '''
    re = np.random.rand(Ni) 
    theta = 0
    x = np.cos(theta) * re
    y = np.sin(theta) * re
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi,   requires_grad=True, device='cuda')
    return xi
def train_data(Nb, Nf):
    '''
    generate essential boundary points and domain points 
    '''
    

    xu = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  np.ones([int(Nb/4),1])])
    xd = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  -np.ones([int(Nb/4),1])])
    xl = np.hstack([-np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1])
    xr = np.hstack([np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1])
    xcrack = np.hstack([-np.random.rand(int(Nb/4),1), np.zeros([int(Nb/4),1])])
    # add corner points when attribute random points 
    xc1 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]]) 
    xc2 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]])
    xc3 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]])
    Xb = np.concatenate((xu, xd, xl, xr, xcrack, xc1, xc2, xc3)) 

    Xb = torch.tensor(Xb, device='cuda') 
    Xf = torch.rand(Nf, 2)*2-1
    
    Xf1 = Xf[(Xf[:, 1]>0) & (torch.norm(Xf, dim=1)>=delta)] # points in the up region
    Xf2 = Xf[(Xf[:, 1]<0) & (torch.norm(Xf, dim=1)>=delta)] # points in the down region 
    
    Xf1 = torch.tensor(Xf1, requires_grad=True, device='cuda')
    Xf2 = torch.tensor(Xf2,  requires_grad=True, device='cuda')
    
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

# for the u0 solution satisfying the homogenous boundary condition
class homo(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(homo, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)
        #self.trans = torch.nn.Linear(1, H)
        # self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        

        self.a1 = torch.Tensor([0.1]).cuda()
        self.a2 = torch.Tensor([0.1])
        self.a3 = torch.Tensor([0.1])
        self.n = 1/self.a1.data
        
        self.fre1 = torch.ones(25).cuda()
        self.fre2 = torch.ones(25).cuda()
        
        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)

        torch.nn.init.normal_(self.fre1, mean=0, std=a)
        torch.nn.init.normal_(self.fre2, mean=0, std=a)

        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # yt1 = torch.cat((torch.cos(self.fre1*x[:, 0].unsqueeze(1)), torch.sin(self.fre1*x[:, 0].unsqueeze(1))), 1)
        # yt2 = torch.cat((torch.cos(self.fre2*x[:, 0].unsqueeze(1)), torch.sin(self.fre2*x[:, 0].unsqueeze(1))), 1)
        # yt3 = torch.cat((torch.cos(self.fre1*x[:, 1].unsqueeze(1)), torch.sin(self.fre1*x[:, 1].unsqueeze(1))), 1)
        # yt4 = torch.cat((torch.cos(self.fre2*x[:, 1].unsqueeze(1)), torch.sin(self.fre2*x[:, 1].unsqueeze(1))), 1)        
        # yt = torch.cat((yt1, yt2, yt3, yt4), 1)
        yt = x
        y1 = torch.tanh(self.n*self.a1*self.linear1(yt))
        y2 = torch.tanh(self.n*self.a1*self.linear2(y1))
        y3 = torch.tanh(self.n*self.a1*self.linear3(y2)) + y1
        y4 = torch.tanh(self.n*self.a1*self.linear4(y3)) + y2
        y = self.n*self.a1*self.linear5(y4)
        return y
    
def pred(xy):
    '''
    predict the displacement field (interesting field)

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
    # predict the output in the up region due to different particular NN used
    pred[(xy[:, 1]>0) ] = model_p1(xy[xy[:, 1]>0]) + \
        RBF(xy[xy[:, 1]>0]) * model_h(xy[xy[:, 1]>0])
    # predict the output in the down region due to different particular NN used
    pred[xy[:, 1]<0] = model_p2(xy[xy[:, 1]<0]) + \
        RBF(xy[xy[:, 1]<0]) * model_h(xy[xy[:, 1]<0])
    # the prediction on the interface
    pred[(xy[:, 1]==0) & (xy[:, 0]>0)] = (model_p1(xy[(xy[:, 1]==0) & (xy[:, 0]>0)]) + model_p2(xy[(xy[:, 1]==0) & (xy[:, 0]>0)]))/2  + \
        RBF(xy[(xy[:, 1]==0) & (xy[:, 0]>0)]) * model_h(xy[(xy[:, 1]==0) & (xy[:, 0]>0)])
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


# learning the boundary solution
# step 1: get the boundary train data and target for boundary condition

if train_p == 1:
    model_p1 = particular(2, 10, 1).cuda() # define two particular nn 
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
            Xb1 = Xb[Xb[:, 1]>=0] # boundary points in the up region
            Xb2 = Xb[Xb[:, 1]<=0] # boundary points in the down region
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
            loss_b = loss_b1 + loss_b2 # loss function of the boundary
            # the loss function on the interface, since there are two particular NN
            pred_bi1 = model_p1(Xi) # predict the boundary condition
            pred_bi2 = model_p2(Xi) # predict the boundary condition
            loss_bi = criterion(pred_bi1, pred_bi2)             
            
            optimp.zero_grad()
            loss_bn = loss_b + loss_bi # loss function of boundary and interface
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
    
    d_total_t = torch.tensor(d_total, device='cuda').unsqueeze(1)
    
    w_t = torch.tensor(w,  device='cuda')

    x_l = x.unsqueeze(0).repeat(len(d_total_t), 1, 1)
    # distance matrix for K matrix
    R2 = torch.norm(d_total_t - x_l, dim=2)
    y = torch.mm(torch.exp(-gama*R2.T), w_t)
    return y

n_d = 10 # the number of the points on each essential boundary
n_dom = 5
gama = 0.5
ep = np.linspace(-1, 1, n_d) # uniform distribution from -1 to 1
# obtain the points of essential condtion
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
points_d = np.concatenate((ep1, ep2, ep3, ep4, ep5)) 
points_d = np.unique(points_d, axis=0) # delete the same points
kdt = KDTree(points_d, metric='euclidean') 

# domain points for distance function
domx = np.linspace(-1, 1, n_dom)[1:-1]
domy = np.linspace(-1, 1, n_dom)[1:-1]
domx, domy = np.meshgrid(domx, domy)
domxy = np.stack((domx.flatten(), domy.flatten()), 1)
domxy = domxy[(domxy[:, 1]!=0)|(domxy[:, 0]>0)]
#domxy = np.unique(domxy, axis=0)
d_dir, _ = kdt.query(points_d, k=1, return_distance = True)
d_dom, _ = kdt.query(domxy, k=1, return_distance = True)
# concatenate essential and domain points
d_total = np.concatenate((points_d, domxy))
#d_total = np.unique(d_total, axis=0)
# distance matrix for K matrix used to determine the weight of RBF
dx = d_total[:, 0][:, np.newaxis] - d_total[:, 0][np.newaxis, :]
dy = d_total[:, 1][:, np.newaxis] - d_total[:, 1][np.newaxis, :]
R2 = np.sqrt(dx**2+dy**2)
K = np.exp(-0.5*R2)
b = np.concatenate((d_dir, d_dom))
w = np.dot(np.linalg.inv(K),b)

n_test = 101 # obtain test points: n_test**2个
domx_t = np.linspace(-1, 1, n_test)
domy_t = np.linspace(-1, 1, n_test)
domx_t, domy_t = np.meshgrid(domx_t, domy_t)
domxy_t = np.stack((domx_t.flatten(), domy_t.flatten()), 1)
domxy_t= torch.tensor(domxy_t,  requires_grad=True, device = 'cuda') 

# =============================================================================
# data-driven method
# =============================================================================
model_h = homo(2, 20, 1).cuda()
criterion = torch.nn.MSELoss()
optim_h = torch.optim.Adam(params=chain(model_h.parameters(), model_p1.parameters(), model_p2.parameters()), lr= 0.001)
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
        u_exact1 = torch.zeros((len(Xf1), 1))
        u_exact1 = torch.sqrt(torch.sqrt(Xf1[:, 0]**2+Xf1[:, 1]**2))*torch.sqrt((1-Xf1[:, 0]/torch.sqrt(Xf1[:, 0]**2+Xf1[:, 1]**2))/2)
        u_exact1 = u_exact1.unsqueeze(1)
        u_exacti1 = torch.zeros((len(Xi1), 1))
        u_exacti1 = torch.sqrt(torch.sqrt(Xi1[:, 0]**2+Xi1[:, 1]**2))*torch.sqrt((1-Xi1[:, 0]/torch.sqrt(Xi1[:, 0]**2+Xi1[:, 1]**2))/2)
        u_exacti1 = u_exacti1.unsqueeze(1)
        
        u_exact2 = torch.zeros((len(Xf2), 1))
        u_exact2 = -torch.sqrt(torch.sqrt(Xf2[:, 0]**2+Xf2[:, 1]**2))*torch.sqrt((1-Xf2[:, 0]/torch.sqrt(Xf2[:, 0]**2+Xf2[:, 1]**2))/2)
        u_exact2 = u_exact2.unsqueeze(1)
        u_exacti2 = torch.zeros((len(Xi2), 1))
        u_exacti2 = -torch.sqrt(torch.sqrt(Xi2[:, 0]**2+Xi2[:, 1]**2))*torch.sqrt((1-Xi2[:, 0]/torch.sqrt(Xi2[:, 0]**2+Xi2[:, 1]**2))/2)
        u_exacti2 = u_exacti2.unsqueeze(1)
        
        Xb1 = Xb[Xb[:, 1]>=0] # boundary points on the up region
        Xb2 = Xb[Xb[:, 1]<=0] # boundary points on the down region
        target_b1 = torch.sqrt(torch.sqrt(Xb1[:,0]**2+Xb1[:, 1]**2))*torch.sqrt((1-Xb1[:,0]/torch.sqrt(Xb1[:,0]**2+Xb1[:,1]**2))/2)
        target_b1 = target_b1.unsqueeze(1)
        target_b2 = -torch.sqrt(torch.sqrt(Xb2[:,0]**2+Xb2[:, 1]**2))*torch.sqrt((1-Xb2[:,0]/torch.sqrt(Xb2[:,0]**2+Xb2[:,1]**2))/2)
        target_b2 = target_b2.unsqueeze(1)
    def closure():  

        u_h1 = model_h(Xf1) 
        u_h2 = model_h(Xf2)
        
        u_p1 = model_p1(Xf1)  
        u_p2 = model_p2(Xf2) # obtain the particular solution
        # construct the admissible function
        u_pred1 = u_p1 + RBF(Xf1)*u_h1
        u_pred2 = u_p2 + RBF(Xf2)*u_h2

        J1 = criterion(u_pred1, u_exact1)
        
        J2 = criterion(u_pred2, u_exact2)
        
        u_i1 = model_p1(Xi1)  + RBF(Xi1)*model_h(Xi1) # get the prediction of the interface
        u_i2 = model_p2(Xi2)  + RBF(Xi2)*model_h(Xi2)         


        # calculate the loss 
        Jie1 = criterion(u_i1, u_exacti1) 
        Jie2 = criterion(u_i2, u_exacti2) 
        if delta != 0: # refine points
            J = J1 + J2 + Jie1 + Jie2
        if delta ==0: # not refine
            J = J1 + J2
        
        loss = J
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
            print(' epoch : %i, the loss : %f ,  J1 : %f , J2 : %f , Jie1 : %f, Jie2 : %f' % (epoch, loss.data, J1.data, J2.data, Jie1.data, Jie2.data))
    optim_h.step(closure)
    scheduler.step()
    
# the prediction of data-driven
N_test = 100
x = np.linspace(-1, 1, N_test)
y0 = np.zeros(N_test)
y = np.linspace(-1, 1, N_test)
x0 = np.zeros(N_test)
xy_test = np.stack((x, y0),1)
xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction

u_pred_up_data = model_p1(xy_test) + RBF(xy_test)*model_h(xy_test) # the prediction of the up region
u_pred_up_data = u_pred_up_data.flatten().cpu()

u_exact_up = np.sqrt(np.sqrt(x**2+y0**2))*np.sqrt((1-x/np.sqrt(x**2+y0**2))/2)
u_exact_up = torch.from_numpy(u_exact_up)
error_up_data = torch.abs(u_pred_up_data - u_exact_up) # get the error in every points
error_up_data = torch.norm(error_up_data)/torch.norm(u_exact_up) # get the total relative L2 error

# obtain the prediction of data driven about y=0 down
u_pred_down_data = model_p2(xy_test) + RBF(xy_test)*model_h(xy_test) # the prediction of the down region
u_pred_down_data = u_pred_down_data.flatten().cpu()

u_exact_down = -np.sqrt(np.sqrt(x**2+y0**2))*np.sqrt((1-x/np.sqrt(x**2+y0**2))/2)
u_exact_down = torch.from_numpy(u_exact_down)
error_down_data = torch.abs(u_pred_down_data - u_exact_down) # get the error in every points
error_down_data = torch.norm(error_down_data)/torch.norm(u_exact_down) # get the total relative L2 error

# obtain the prediction of data driven about x=0 down
xy_test = np.stack((x0, y),1)
xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction

u_pred_mid_data = torch.zeros((N_test,1)).cuda()
u_pred_mid_data[y>0] = model_p1(xy_test[y>0]) + RBF(xy_test[y>0])*model_h(xy_test[y>0]) # x=0 up region prediction
u_pred_mid_data[y<0] = model_p2(xy_test[y<0]) + RBF(xy_test[y<0])*model_h(xy_test[y<0]) # x=0 down region prediction
u_pred_mid_data = u_pred_mid_data.flatten().cpu()
u_exact_mid = torch.zeros(N_test) # the exact solution in x=0
u_exact_mid[y>0] = torch.norm(xy_test[y>0], dim=1)**0.5*np.sqrt(2)/2
u_exact_mid[y<0] = -torch.norm(xy_test[y<0], dim=1)**0.5*np.sqrt(2)/2
u_exact_mid= u_exact_mid.cpu()
error_mid_data = torch.abs(u_pred_mid_data - u_exact_mid) # get the error in every points
error_mid_data = torch.norm(error_mid_data)/torch.norm(u_exact_mid) # get the total relative L2 error

# Singular strain evaluation, epsilon y when x > 0 and y=0
n_inter = 11
interx = torch.linspace(0, 1, n_inter)[1:-1] # the test points on the interface
intery = torch.zeros(n_inter)[1:-1]
inter = torch.stack((interx, intery),1)
inter = inter.requires_grad_(True).cuda() 
pred_inter_data = pred(inter)
dudxyi_data = grad(pred_inter_data, inter, torch.ones(inter.size()[0], 1).cuda())[0]
dudyi_data = dudxyi_data[:, 1].unsqueeze(1) # singular strain on the interface
dudyi_e = 0.5/torch.sqrt(torch.norm(inter, dim=1))# the exact strain on the interface
error_dy_data = torch.abs(dudyi_data - dudyi_e) 
error_dy_data = torch.norm(error_dy_data)/torch.norm(dudyi_e)

# =============================================================================
# cpinn method
# =============================================================================
nepoch_u0 = nepoch_u1 
def NTK(predf1, predf2, predb1, predb2):
    paranum = sum(p.numel() for p in model_p1.parameters())-1 
    A1 = torch.zeros((len(predf1), paranum)).cuda() 
    A2 = torch.zeros((len(predf2), paranum)).cuda()
    A3 = torch.zeros((len(predb1), paranum)).cuda()
    A4 = torch.zeros((len(predb2), paranum)).cuda()
    for index,pred_e in enumerate(predf1):
        grad_e = grad(pred_e, model_p1.parameters(),retain_graph=True,  create_graph=True, allow_unused=True) # 
        grad_one = torch.tensor([]).cuda() 
        for i in grad_e[:-1]: 
            grad_one = torch.cat((grad_one, i.flatten())) 
        A1[index] = grad_one 
    K1 = torch.mm(A1, A1.T) # get K by multiplication, note: multiplication of matrix can reduce the calculation instead of "for loop"

    for index,pred_e in enumerate(predf2):
        grad_e = grad(pred_e, model_p2.parameters(),retain_graph=True,  create_graph=True, allow_unused=True) 
        grad_one = torch.tensor([]).cuda()
        for i in grad_e[:-1]:
            grad_one = torch.cat((grad_one, i.flatten())) 
        A2[index] = grad_one 
    K2 = torch.mm(A2, A2.T) 

    for index,pred_e in enumerate(predb1):
        grad_e = grad(pred_e, model_p1.parameters(),retain_graph=True,  create_graph=True, allow_unused=True) 
        grad_one = torch.tensor([]).cuda() 
        for i in grad_e[:-1]:
            grad_one = torch.cat((grad_one, i.flatten())) 
        A3[index] = grad_one 
    K3 = torch.mm(A3, A3.T) 
    
    for index,pred_e in enumerate(predb2):
        grad_e = grad(pred_e, model_p2.parameters(),retain_graph=True,  create_graph=True, allow_unused=True) 
        grad_one = torch.tensor([]).cuda() 
        for i in grad_e[:-1]:
            grad_one = torch.cat((grad_one, i.flatten())) 
        A4[index] = grad_one 
    K4 = torch.mm(A4, A4.T) 
    
    tr1 = torch.trace(K1.data)
    tr2 = torch.trace(K2.data)
    tr3 = torch.trace(K3.data)    
    tr4 = torch.trace(K4.data)
    
    return tr1, tr2, tr3, tr4
def interface1(Ni1): 
    theta = np.pi*np.random.rand(Ni1)
    rp = delta*np.random.rand(Ni1)
    x = np.cos(theta) * rp
    y = np.sin(theta) * rp
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi,  requires_grad=True, device='cuda')
    return xi
def interface2(Ni1): 
    theta = -np.pi*np.random.rand(Ni1)
    rp = delta*np.random.rand(Ni1)
    x = np.cos(theta) * rp
    y = np.sin(theta) * rp
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi,  requires_grad=True, device='cuda')
    return xi
def interface(Ni):
    re = np.random.rand(Ni)
    theta = 0
    x = np.cos(theta) * re
    y = np.sin(theta) * re
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi,  requires_grad=True, device='cuda')
    return xi
def train_data(Nb, Nf):
    xu = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  np.ones([int(Nb/4),1])])
    xd = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  -np.ones([int(Nb/4),1])])
    xl = np.hstack([-np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1])
    xr = np.hstack([np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1])
    xcrack = np.hstack([-np.random.rand(int(Nb/4),1), np.zeros([int(Nb/4),1])])
    xc1 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]]) 
    xc2 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]])
    xc3 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]])
    Xb = np.concatenate((xu, xd, xl, xr, xcrack, xc1, xc2, xc3))
    Xb = torch.tensor(Xb, device='cuda')   
    Xf = torch.rand(Nf, 2)*2-1

    Xf1 = Xf[(Xf[:, 1]>0) & (torch.norm(Xf, dim=1)>=delta)]
    Xf2 = Xf[(Xf[:, 1]<0) & (torch.norm(Xf, dim=1)>=delta)] 
    
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
    pred[(xy[:, 1]>0) ] = model_p1(xy[xy[:, 1]>0])
    pred[xy[:, 1]<0] = model_p2(xy[xy[:, 1]<0])
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

model_p1 = particular(2, 20, 1).cuda()
model_p2 = particular(2, 20, 1).cuda()
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
b3 = 50
b4 = 50
b5 = 10

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
        
        Xb1 = Xb[Xb[:, 1]>=0] # boundary on the up region
        Xb2 = Xb[Xb[:, 1]<=0] # boundary on the down region
        target_b1 = torch.sqrt(torch.sqrt(Xb1[:,0]**2+Xb1[:, 1]**2))*torch.sqrt((1-Xb1[:,0]/torch.sqrt(Xb1[:,0]**2+Xb1[:,1]**2))/2)
        target_b1 = target_b1.unsqueeze(1)
        target_b2 = -torch.sqrt(torch.sqrt(Xb2[:,0]**2+Xb2[:, 1]**2))*torch.sqrt((1-Xb2[:,0]/torch.sqrt(Xb2[:,0]**2+Xb2[:,1]**2))/2)
        target_b2 = target_b2.unsqueeze(1)
    def closure():  
        global b1,b2,b3,b4,b5
        u_pred1 = model_p1(Xf1)  
        u_pred2 = model_p2(Xf2)

        # the exact integral is 0.8814
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
        
        u_i1 = model_p1(Xi1)  
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
        
        Jie1 = torch.sum((du1dxxi + du1dyyi)**2) 
        Jie2 = torch.sum((du2dxxi + du2dyyi)**2) 

        if delta != 0:
            J = J1 + J2 + Jie1 + Jie2
        if delta ==0:
            J = J1 + J2
        
        # the loss function of the essential condition
        pred_b1 = model_p1(Xb1) # predict the boundary condition
        loss_b1 = criterion(pred_b1, target_b1)  
        pred_b2 = model_p2(Xb2) # predict the boundary condition
        loss_b2 = criterion(pred_b2, target_b2) 
        loss_b = loss_b1 + loss_b2
        
        # interface loss of continuity
        pred_bi1 = model_p1(Xi) # predict the boundary condition
        pred_bi2 = model_p2(Xi) # predict the boundary condition
        loss_bi = criterion(pred_bi1, pred_bi2)  
        
        # interface loss of derivative
        du1dxyii = grad(pred_bi1, Xi, torch.ones(Xi.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du2dxyii = grad(pred_bi2, Xi, torch.ones(Xi.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]     
        loss_bdi = criterion(du1dxyii[:, 1].unsqueeze(1), du2dxyii[:, 1].unsqueeze(1))   
        if delta != 0:
            J1 = J1+Jie1
            J2 = J2+Jie2
        loss =b1 * J1 + b2 * J2 + b3 * loss_b1 + b4 * loss_b2 + b5 * (loss_bi+loss_bdi)
        error_t = evaluate()
        optim_h.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        error_array.append(error_t.data.cpu())
        loss1_array.append((b1*J1+b3*loss_b1+b5*loss_bi+b5*loss_bdi).data.cpu())
        loss2_array.append((b2*J2+b4*loss_b2+b5*loss_bi+b5*loss_bdi).data.cpu())
        lossii1_array.append(Jie1.data.cpu())
        lossii2_array.append(Jie2.data.cpu())
        if epoch%10==0:
            print(' epoch : %i, the loss : %f ,  J1 : %f , J2 : %f , Jie1 : %f, Jie2 : %f, Jb: %f, Ji: %f' % (epoch, loss.data, J1.data, J2.data, Jie1.data, Jie2.data, loss_b.data, (loss_bi+loss_bdi).data))
        # if epoch%100==0:
        #     k1, k2, k3, k4 = NTK(du1dxx + du1dyy, du2dxx + du2dyy, pred_b1, pred_b2)
        #     k = k1 + k2 + k3 + k4
        #     b1 = k/k1
        #     b2 = k/k2
        #     b3 = k/k3
        #     b4 = k/k4
        #     b5 = (b1+b2+b3+b4)/4
        #     print('the NTK is done')
        return loss
    optim_h.step(closure)
    scheduler.step()
    
    
    
N_test = 100
x = np.linspace(-1, 1, N_test)
y0 = np.zeros(N_test)
y = np.linspace(-1, 1, N_test)
x0 = np.zeros(N_test)
xy_test = np.stack((x, y0),1)
xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction

u_pred_up_cpinn = model_p1(xy_test) # 得到上部分的预测解
u_pred_up_cpinn = u_pred_up_cpinn.flatten().cpu()

error_up_cpinn = torch.abs(u_pred_up_cpinn - u_exact_up) # get the error in every points
error_up_cpinn = torch.norm(error_up_cpinn)/torch.norm(u_exact_up) # get the total relative L2 error

#Get cpinn's lower prediction for y=0
u_pred_down_cpinn = model_p2 (xy_test) #Get the prediction solution of the lower part
u_pred_down_cpinn = u_pred_down_cpinn.flatten().cpu()

error_down_cpinn = torch.abs(u_pred_down_cpinn - u_exact_down) # get the error in every points
error_down_cpinn = torch.norm(error_down_cpinn)/torch.norm(u_exact_down) # get the total relative L2 error

#Get cpinn's solution for x=0 middle
xy_test = np.stack((x0, y),1)
xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction

u_pred_mid_cpinn = torch.zeros((N_test,1)).cuda()
u_pred_mid_cpinn[y>0] = model_p1 (xy_test[y>0]) #Use the upper half of the network to obtain the predicted solution, try not to include the midpoint in the test
u_pred_mid_cpinn[y<0] = model_p2 (xy_test[y<0]) #obtain that predict solution with the lower half of the network
u_pred_mid_cpinn = u_pred_mid_cpinn.flatten().cpu()
error_mid_cpinn = torch.abs(u_pred_mid_cpinn - u_exact_mid) # get the error in every points
error_mid_cpinn = torch.norm(error_mid_cpinn)/torch.norm(u_exact_mid) # get the total relative L2 error

#singular strain evaluation, y=0, x is greater than 0 y direction strain
n_inter = 11
interx = torch.linspace(0, 1, n_inter)[1: -1] #Get test points for interface
intery = torch.zeros(n_inter)[1:-1]
inter = torch.stack((interx, intery),1)
inter = inter.requires_grad_(True). cuda () #can be differentiated and put into cuda
pred_inter_cpinn = pred(inter)
dudxyi_cpinn = grad(pred_inter_cpinn, inter, torch.ones(inter.size()[0], 1).cuda())[0]
dudyi_cpinn = dudxyi_cpinn[:, 1]. unsqueeze (1)

error_dy_cpinn = torch.abs(dudyi_cpinn - dudyi_e) 
error_dy_cpinn = torch.norm(error_dy_cpinn)/torch.norm(dudyi_e)

# %%
nepoch_u0 = nepoch_u2 

def NTK(pred):
    paranum = sum(p.numel() for p in model_h.parameters())
    A = torch.zeros((len(pred), paranum)).cuda()
    for index,pred_e in enumerate(pred):
        grad_e = grad(pred_e, model_h.parameters(),retain_graph=True,  create_graph=True) 
        grad_one = torch.tensor([]).cuda() 
        for i in grad_e:
            grad_one = torch.cat((grad_one, i.flatten())) 
        A[index] = grad_one 
    K = torch.mm(A, A.T) 
    L,_ = torch.eig(K.data.cpu()) 
    eig = torch.norm(L, dim=1)
    return eig
def interface1(Ni1): 
    theta = np.pi*np.random.rand(Ni1) 
    rp = delta*np.random.rand(Ni1) 
    x = np.cos(theta) * rp
    y = np.sin(theta) * rp
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi, requires_grad=True, device='cuda')
    return xi
def interface2(Ni1): 
    theta = -np.pi*np.random.rand(Ni1) 
    rp = delta*np.random.rand(Ni1) 
    x = np.cos(theta) * rp
    y = np.sin(theta) * rp
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi,  requires_grad=True, device='cuda')
    return xi
def interface(Ni):
    re = np.random.rand(Ni)
    theta = 0
    x = np.cos(theta) * re
    y = np.sin(theta) * re
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi, requires_grad=True, device='cuda')
    return xi
def train_data(Nb, Nf):
    xu = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  np.ones([int(Nb/4),1])])
    xd = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  -np.ones([int(Nb/4),1])])
    xl = np.hstack([-np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1])
    xr = np.hstack([np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1])
    xcrack = np.hstack([-np.random.rand(int(Nb/4),1), np.zeros([int(Nb/4),1])])

    xc1 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]]) 
    xc2 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]])
    xc3 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]])
    Xb = np.concatenate((xu, xd, xl, xr, xcrack, xc1, xc2, xc3)) 

    Xb = torch.tensor(Xb, device='cuda') 
    
    Xf = torch.rand(Nf, 2)*2-1
    

    Xf1 = Xf[(Xf[:, 1]>0) & (torch.norm(Xf, dim=1)>=delta)] 
    Xf2 = Xf[(Xf[:, 1]<0) & (torch.norm(Xf, dim=1)>=delta)]  
    
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

# for the u0 solution satisfying the homogenous boundary condition
class homo(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(homo, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)
#        self.trans = torch.nn.Linear(1, H)
        # self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        

        self.a1 = torch.nn.Parameter(torch.Tensor([0.1])).cuda()
        self.a2 = torch.Tensor([0.1])
        self.a3 = torch.Tensor([0.1])
        self.n = 1/self.a1.data
        
        self.fre1 = torch.ones(25).cuda()
        self.fre2 = torch.ones(25).cuda()
        self.fre3 = torch.ones(25).cuda()
        self.fre4 = torch.ones(25).cuda()
        self.fre5 = torch.ones(25).cuda()
        self.fre6 = torch.ones(25).cuda()
        self.fre7 = torch.ones(25).cuda()
        self.fre8 = torch.ones(25).cuda()
        
        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)

        torch.nn.init.normal_(self.fre1, mean=0, std=a)
        torch.nn.init.normal_(self.fre2, mean=0, std=a*2)
        torch.nn.init.normal_(self.fre1, mean=0, std=a*3)
        torch.nn.init.normal_(self.fre2, mean=0, std=a*4)
        torch.nn.init.normal_(self.fre1, mean=0, std=a*5)
        torch.nn.init.normal_(self.fre2, mean=0, std=a*6)
        torch.nn.init.normal_(self.fre1, mean=0, std=a*7)
        torch.nn.init.normal_(self.fre2, mean=0, std=a*8)
        
        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        #特征变换
        # yt1 = torch.cat((torch.cos(self.fre1*x[:, 0].unsqueeze(1)), torch.sin(self.fre1*x[:, 0].unsqueeze(1))), 1)
        # yt2 = torch.cat((torch.cos(self.fre2*x[:, 0].unsqueeze(1)), torch.sin(self.fre2*x[:, 0].unsqueeze(1))), 1)
        # yt3 = torch.cat((torch.cos(self.fre3*x[:, 1].unsqueeze(1)), torch.sin(self.fre3*x[:, 1].unsqueeze(1))), 1)
        # yt4 = torch.cat((torch.cos(self.fre4*x[:, 1].unsqueeze(1)), torch.sin(self.fre4*x[:, 1].unsqueeze(1))), 1)      
        # yt5 = torch.cat((torch.cos(self.fre5*x[:, 1].unsqueeze(1)), torch.sin(self.fre5*x[:, 1].unsqueeze(1))), 1)
        # yt6 = torch.cat((torch.cos(self.fre6*x[:, 1].unsqueeze(1)), torch.sin(self.fre6*x[:, 1].unsqueeze(1))), 1)        
        # yt7 = torch.cat((torch.cos(self.fre7*x[:, 1].unsqueeze(1)), torch.sin(self.fre7*x[:, 1].unsqueeze(1))), 1)
        # yt8 = torch.cat((torch.cos(self.fre8*x[:, 1].unsqueeze(1)), torch.sin(self.fre8*x[:, 1].unsqueeze(1))), 1)        
        #yt = torch.cat((yt1, yt2, yt3, yt4, yt5, yt6, yt7, yt8), 1)
        yt = x
        y1 = torch.tanh(self.n*self.a1*self.linear1(yt))
        y2 = torch.tanh(self.n*self.a1*self.linear2(y1))
        y3 = torch.tanh(self.n*self.a1*self.linear3(y2)) + y1
        y4 = torch.tanh(self.n*self.a1*self.linear4(y3)) + y2
        y =  self.linear5(y4)
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
    #on the region's prediction, due to the use of different network solution
    pred[(xy[:, 1]>0) ] = model_p1(xy[xy[:, 1]>0]) + \
        RBF(xy[xy[:, 1]>0]) * model_h(xy[xy[:, 1]>0])
    #area under the forecast, due to the use of different solution network
    pred[xy[:, 1]<0] = model_p2(xy[xy[:, 1]<0]) + \
        RBF(xy[xy[:, 1]<0]) * model_h(xy[xy[:, 1]<0])
    #The prediction of the right end of crack uses different special solution networks, so the draw value is removed, that is, the interface of slices. It is better not to use the following code, because if the following code is used, the judgment at the crack will be used, but the judgment at the crack will appear paradox.
    pred[(xy[:, 1]==0) & (xy[:, 0]>0)] = (model_p1(xy[(xy[:, 1]==0) & (xy[:, 0]>0)]) + model_p2(xy[(xy[:, 1]==0) & (xy[:, 0]>0)]))/2  + \
        RBF(xy[(xy[:, 1]==0) & (xy[:, 0]>0)]) * model_h(xy[(xy[:, 1]==0) & (xy[:, 0]>0)])

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
    model_p1 = particular(2, 10, 1).cuda() 
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
            Xb1 = Xb[Xb[:, 1]>=0] 
            Xb2 = Xb[Xb[:, 1]<=0] 
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
            loss_b = loss_b1 + loss_b2

            pred_bi1 = model_p1(Xi) # predict the boundary condition
            pred_bi2 = model_p2(Xi) # predict the boundary condition
            loss_bi = criterion(pred_bi1, pred_bi2)             
            
            optimp.zero_grad()
            loss_bn = loss_b + loss_bi # Sum of intrinsic boundary and interface losses
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
    d_total_t = torch.from_numpy(d_total).unsqueeze(1).cuda()
    w_t = torch.from_numpy(w).cuda()
    x_l = x.unsqueeze(0).repeat(len(d_total_t), 1, 1) 
    # 得到R2大矩阵
    R2 = torch.norm(d_total_t - x_l, dim=2)
    y = torch.mm(torch.exp(-gama*R2.T), w_t)
    return y

n_d = 10 
n_dom = 5
gama = 0.5
ep = np.linspace(-1, 1, n_d) 

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
points_d = np.concatenate((ep1, ep2, ep3, ep4, ep5)) 
points_d = np.unique(points_d, axis=0)
kdt = KDTree(points_d, metric='euclidean') 


domx = np.linspace(-1, 1, n_dom)[1:-1]
domy = np.linspace(-1, 1, n_dom)[1:-1]
domx, domy = np.meshgrid(domx, domy)
domxy = np.stack((domx.flatten(), domy.flatten()), 1)
domxy = domxy[(domxy[:, 1]!=0)|(domxy[:, 0]>0)]

d_dir, _ = kdt.query(points_d, k=1, return_distance = True)
d_dom, _ = kdt.query(domxy, k=1, return_distance = True)

d_total = np.concatenate((points_d, domxy))

dx = d_total[:, 0][:, np.newaxis] - d_total[:, 0][np.newaxis, :]
dy = d_total[:, 1][:, np.newaxis] - d_total[:, 1][np.newaxis, :]
R2 = np.sqrt(dx**2+dy**2)
K = np.exp(-gama*R2)
b = np.concatenate((d_dir, d_dom))
w = np.dot(np.linalg.inv(K),b)



n_test = 21 # Obtain test points n_test** 2
domx_t = np.linspace(-1, 1, n_test)
domy_t = np.linspace(-1, 1, n_test)
domx_t, domy_t = np.meshgrid(domx_t, domy_t)
domxy_t = np.stack((domx_t.flatten(), domy_t.flatten()), 1)
domxy_t= torch.from_numpy(domxy_t).requires_grad_(True).cuda() # 获得了两列的测试点，放入RBF中，这是为了和其他神经网络结构保持一致

dis = RBF(domxy_t)
# %%
model_h = homo(2, 20, 1).cuda()
criterion = torch.nn.MSELoss()
optim_h = torch.optim.Adam(params=model_h.parameters(), lr= 0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_h, milestones=[1000, 3000, 5000], gamma = 0.1)
loss_array = []
error_array = []
loss1_array = []
loss2_array = []  
lossii1_array = []  
lossii2_array = []   
lossi_array = []
eigvalues = []
nepoch_u0 = int(nepoch_u0)
start = time.time()
for epoch in range(nepoch_u0):
    if epoch ==1000:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0:
        Xb, Xf1, Xf2 = train_data(100, 4096)
        Xi1 = interface1(5000)
        Xi2 = interface2(5000)
        Xi = interface(1000)
    def closure():  
        u_h1 = model_h(Xf1) 
        u_h2 = model_h(Xf2)
        
        u_p1 = model_p1(Xf1)  
        u_p2 = model_p2(Xf2)
        # admissible function
        u_pred1 = u_p1 + RBF(Xf1)*u_h1
        u_pred2 = u_p2 + RBF(Xf2)*u_h2
        
        du1dxy = grad(u_pred1, Xf1, torch.ones(Xf1.size()[0], 1).cuda(), create_graph=True)[0]
        du1dx = du1dxy[:, 0].unsqueeze(1)
        du1dy = du1dxy[:, 1].unsqueeze(1)

        du2dxy = grad(u_pred2, Xf2, torch.ones(Xf2.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du2dx = du2dxy[:, 0].unsqueeze(1)
        du2dy = du2dxy[:, 1].unsqueeze(1)

        J1 = (0.5 * torch.sum(du1dx**2 + du1dy**2)) * (2- 0.5*(delta)**2*np.pi)/len(Xf1) 
        
        J2 = (0.5 * torch.sum(du2dx**2 + du2dy**2)) * (2- 0.5*(delta)**2*np.pi)/len(Xf2) 

        
        u_i1 = model_p1(Xi1)  + RBF(Xi1)*model_h(Xi1)  
        u_i2 = model_p2(Xi2)  + RBF(Xi2)*model_h(Xi2)         

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
            
        u_ii1 = model_p1(Xi)  + RBF(Xi)*model_h(Xi)  
        u_ii2 = model_p2(Xi)  + RBF(Xi)*model_h(Xi)  
        Ji = criterion(u_ii1, u_ii2)  
        
        loss = J # penalty * Ji # the functional
        error_t = evaluate()
        optim_h.zero_grad()
        loss.backward()
        loss_array.append(loss.data.cpu())
        error_array.append(error_t.data.cpu())
        loss1_array.append(J1.data.cpu())
        loss2_array.append(J2.data.cpu())
        lossii1_array.append(Jie1.data.cpu())
        lossii2_array.append(Jie2.data.cpu())
        lossi_array.append(Ji.data.cpu())
        if epoch%10==0:
            print(' epoch : %i, the loss : %f ,  J1 : %f , J2 : %f , Jie1 : %f, Jie2 : %f, Ji : %f ' % (epoch, loss.data, J1.data, J2.data, Jie1.data, Jie2.data, Ji.data))
        if epoch%500==0:
            x_ntk = torch.cat((Xf1[0:50], Xf2[0:50])).data
            pred_ntk = pred(x_ntk)
            eigvalue = NTK(pred_ntk)
            eigvalues.append(eigvalue)
            print('the NTK is done')
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
u_pred_energy = u_pred.reshape(N_test, N_test).cpu()

u_exact = np.zeros(x.shape)
u_exact[y>0] = np.sqrt(np.sqrt(x[y>0]**2+y[y>0]**2))*np.sqrt((1-x[y>0]/np.sqrt(x[y>0]**2+y[y>0]**2))/2)
u_exact[y<0] = -np.sqrt(np.sqrt(x[y<0]**2+y[y<0]**2))*np.sqrt((1-x[y<0]/np.sqrt(x[y<0]**2+y[y<0]**2))/2)
u_exact = torch.from_numpy(u_exact)
error_energy  = torch.abs(u_pred_energy  - u_exact) # get the error in every points
error_t = torch.norm(error_energy )/torch.norm(u_exact) # get the total relative L2 error


# plot the prediction solution
Xb, Xf1, Xf2 = train_data(256, 4096)
Xi1 = interface1(1000)
Xi2 = interface2(1000)


N_test = 100
x = np.linspace(-1, 1, N_test)
y0 = np.zeros(N_test)
y = np.linspace(-1, 1, N_test)
x0 = np.zeros(N_test)
xy_test = np.stack((x, y0),1)
xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction

u_pred_up_cenn = model_p1(xy_test) + RBF(xy_test)*model_h(xy_test) 
u_pred_up_cenn = u_pred_up_cenn.flatten().cpu()

error_up_cenn = torch.abs(u_pred_up_cenn - u_exact_up) # get the error in every points
error_up_cenn = torch.norm(error_up_cenn)/torch.norm(u_exact_up) # get the total relative L2 error

#Get the data driven down prediction for y=0
u_pred_down_cenn = model_p2(xy_test) + RBF(xy_test)*model_h(xy_test)
u_pred_down_cenn = u_pred_down_cenn.flatten().cpu()

error_down_cenn = torch.abs(u_pred_down_cenn - u_exact_down) # get the error in every points
error_down_cenn = torch.norm(error_down_cenn)/torch.norm(u_exact_down) # get the total relative L2 error

#Get the solution of datadriven for x=0 middle
xy_test = np.stack((x0, y),1)
xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction

u_pred_mid_cenn = torch.zeros((N_test,1)).cuda()
u_pred_mid_cenn[y>0] = model_p1 (xy_test[y>0]) + RBF(xy_test[y>0])*model_h(xy_test[y>0]) #Use the upper half of the network to obtain the prediction solution, and try not to include the midpoint in test
u_pred_mid_cenn[y<0] = model_p2 (xy_test[y<0]) + RBF(xy_test[y<0])*model_h(xy_test[y<0]) #that prediction solution is obtain with the lower half of the network
u_pred_mid_cenn = u_pred_mid_cenn.flatten().cpu()
error_mid_cenn = torch.abs(u_pred_mid_cenn - u_exact_mid) # get the error in every points
error_mid_cenn = torch.norm(error_mid_cenn)/torch.norm(u_exact_mid) # get the total relative L2 error

# Singular strain evaluation, y=0, y-direction strain where x is greater than 0
n_inter = 11
interx = torch.linspace(0, 1, n_inter)[1:-1] # Get the test points of the interface
intery = torch.zeros(n_inter)[1:-1]
inter = torch.stack((interx, intery),1)
inter = inter.requires_grad_(True).cuda() # can be derived and put into cuda
pred_inter_cenn = pred(inter)
dudxyi_cenn = grad(pred_inter_cenn, inter, torch.ones(inter.size()[0], 1).cuda())[0]
dudyi_cenn = dudxyi_cenn[:, 1].unsqueeze(1) # Obtain the singular strain of the interface

error_dy_cenn = torch.abs(dudyi_cenn - dudyi_e)
error_dy_cenn = torch.norm(error_dy_cenn)/torch.norm(dudyi_e)


# Convert all tensors to numpy
u_pred_up_data = u_pred_up_data.data.cpu().numpy()
u_pred_down_data = u_pred_down_data.data.cpu().numpy()
u_pred_mid_data = u_pred_mid_data.data.cpu().numpy()
dudyi_data = dudyi_data.data.cpu().numpy()

u_pred_up_cpinn = u_pred_up_cpinn.data.cpu().numpy()
u_pred_down_cpinn = u_pred_down_cpinn.data.cpu().numpy()
u_pred_mid_cpinn = u_pred_mid_cpinn.data.cpu().numpy()
dudyi_cpinn = dudyi_cpinn.data.cpu().numpy()

u_pred_up_cenn = u_pred_up_cenn.data.cpu().numpy()
u_pred_down_cenn = u_pred_down_cenn.data.cpu().numpy()
u_pred_mid_cenn = u_pred_mid_cenn.data.cpu().numpy()
dudyi_cenn = dudyi_cenn.data.cpu().numpy()

dudyi_e = dudyi_e.data.cpu().numpy()
interx = interx.data.cpu().numpy()

# %%
#axs = plt.figure(dpi=1000, figsize=(12, 11), constrained_layout = True).subplots(2,2)
#plt.subplot(2, 2, 1)
plt.figure(dpi=1000, figsize=(6, 5.5)) 
plt.plot(x, u_exact_up)
plt.plot(x, u_pred_up_data, linestyle=':')
plt.plot(x, u_pred_up_cpinn, linestyle='--')
plt.plot(x, u_pred_up_cenn, linestyle='-.')
plt.legend(['Exact', 'Data driven', 'cPINN', 'CENN'], loc='upper right')
plt.xlabel('x')
plt.ylabel('u')
plt.savefig('crack_position/y=0_up.png', bbox_inches = 'tight')
plt.show()
#axs[0, 0].set_title('y=0 up', size = 20)

#axs[0, 1].subplot(2, 2, 2)
plt.figure(dpi=1000, figsize=(6, 5.5)) 
plt.plot(x, u_exact_down)
plt.plot(x, u_pred_down_data, linestyle=':')
plt.plot(x, u_pred_down_cpinn, linestyle='--')
plt.plot(x, u_pred_down_cenn, linestyle='-.')
plt.legend(['Exact', 'Data driven', 'cPINN', 'CENN'], loc='upper left')
plt.xlabel('x')
plt.ylabel('u')
plt.savefig('crack_position/y=0_down.png' , bbox_inches = 'tight')
plt.show()
#axs[0, 1].set_title('y=0 down', size = 20)

#axs[1, 0].subplot(2, 2, 3)
plt.figure(dpi=1000, figsize=(6, 5.5)) 
plt.plot(y, u_exact_mid)
plt.plot(y, u_pred_mid_data, linestyle=':')
plt.plot(y, u_pred_mid_cpinn, linestyle='--')
plt.plot(y, u_pred_mid_cenn, linestyle='-.')
plt.legend(['Exact', 'Data driven', 'cPINN', 'CENN'], loc='upper left')
plt.xlabel('y')
plt.ylabel('u')
plt.savefig('crack_position/x=0.png', bbox_inches = 'tight')
plt.show()
#axs[1, 0].set_title('x=0', size = 20)

#axs[1, 1].subplot(2, 2, 4)
plt.figure(dpi=1000, figsize=(6, 5.5)) 
plt.plot(interx, dudyi_e, marker='o')
plt.plot(interx, dudyi_data, linestyle=':', marker = '*')
plt.plot(interx, dudyi_cpinn, linestyle='--', marker = 'v')
plt.plot(interx, dudyi_cenn, linestyle='-.', marker = '^')
plt.legend(['Exact', 'Data driven', 'cPINN', 'CENN'],  loc='upper right')
plt.xlabel('x')
plt.ylabel('$\epsilon_{32}$')
plt.savefig('crack_position/singular_strain.png', bbox_inches = 'tight')
plt.show()
#axs[1, 1].set_title('singular strain e32 on interface', size = 20)


#plt.savefig('../../图片/裂纹/crack_compare_cross/crack_compare_cross%i.pdf' % dd, bbox_inches = 'tight')
print('process is done' )
#plt.show()
print('the relative up  data error is %f' % error_up_data.data)
print('the relative up  cpinn error is %f' % error_up_cpinn.data)
print('the relative up  cenn error is %f' % error_up_cenn.data)
print('the relative down  data error is %f' % error_down_data.data)
print('the relative down  cpinn error is %f' % error_down_cpinn.data)
print('the relative down  cenn error is %f' % error_down_cenn.data)
print('the relative mid  data error is %f' % error_mid_data.data)
print('the relative mid  cpinn error is %f' % error_mid_cpinn.data)
print('the relative mid  cenn error is %f' % error_mid_cenn.data)
print('the relative e32  data error is %f' % error_dy_data.data)
print('the relative e32  cpinn error is %f' % error_dy_cpinn.data)
print('the relative e32  cenn error is %f' % error_dy_cenn.data)
    
