import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from  torch.autograd import grad
from pyevtk.hl import gridToVTK
from pyevtk.hl import pointsToVTK
import xlrd
import sys
sys.path.append("../..") 
from kan_efficiency import *
import time
def setup_seed(seed):
# random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2024)

class MultiLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MultiLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)

        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)

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
        yt = x
        y1 = torch.tanh(self.linear1(yt))
        y2 = torch.tanh(self.linear2(y1))
        y3 = torch.tanh(self.linear3(y2)) + y1
        y4 = torch.tanh(self.linear4(y3)) + y2
        y =  self.linear5(y4)
        return y

def getUST(x):
    '''
    

    Parameters
    ----------
    x : tensor
        coordinate of 2 dimensionality.

    Returns
    -------
    ust_pred : tensor
        get the displacement, strain and stress in 2 dimensionality.

    '''
    x_scale = x/Length

    ust_pred = model(x_scale)
    return ust_pred




# Generate training data
def generate_training_data(dom_num = 200, boundary_num = 1000):
    plate_size = Length
    hole_radius = Radius
    points = []
    for i in range(dom_num):
        for j in range(dom_num):
            x = i * plate_size / (dom_num - 1)
            y = j * plate_size / (dom_num - 1)
            if x**2 + y**2 >= hole_radius**2:
                points.append((x, y))
    dom_point = np.array(points) # 这里必须64，才划分网格不会和32混淆
    
    # 删除大于radius的点
    distances = np.sqrt(np.sum(dom_point**2, axis=1))
    filtered_points = dom_point[distances >= hole_radius]

    # Generate points along the vertical and horizontal boundaries
    left_wall = np.array([[0, y] for y in np.linspace(hole_radius, plate_size, boundary_num)]).astype(np.float32)
    bottom_wall = np.array([[x, 0] for x in np.linspace(hole_radius, plate_size, boundary_num)]).astype(np.float32)
    right_wall = np.array([[Length, y] for y in np.linspace(0, plate_size, boundary_num)]).astype(np.float32)
    up_wall = np.array([[x, Length] for x in np.linspace(0, plate_size, boundary_num)]).astype(np.float32)
    # Generate points along the circular arc
    theta = np.linspace(0, np.pi/2, boundary_num)
    arc = np.array([[hole_radius * np.cos(t), hole_radius* np.sin(t)] for t in theta]).astype(np.float32)

    # Combine all points
    boundary_points = {'l': left_wall, 'd': bottom_wall, 'r': right_wall, 'u': up_wall, 'c': arc}
    
    filtered_points = filtered_points.astype(np.float32) # 换成32方便后面计算
    return filtered_points, boundary_points

def evaluate_model(datatest):

    Nx = len(datatest)
    x = datatest[:, 0].reshape(Nx, 1)
    y = datatest[:, 1].reshape(Nx, 1)
    
    xy = np.concatenate((x, y), axis=1)
    xy_tensor = torch.from_numpy(xy).float()
    xy_tensor = xy_tensor.cuda()
    xy_tensor.requires_grad_(True)
    ust_pred_torch = getUST(xy_tensor)
    duxdxy = grad(ust_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device='cuda'),
                   create_graph=True, retain_graph=True)[0]
    duydxy = grad(ust_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device='cuda'),
                   create_graph=True, retain_graph=True)[0]
    dudx = duxdxy[:, 0]
    dudy = duxdxy[:, 1]
    dvdx = duydxy[:, 0]
    dvdy = duydxy[:, 1]
    exx_pred = dudx
    eyy_pred = dvdy
    e2xy_pred = dudy + dvdx     
    sxx_pred = D11_mat * exx_pred + D12_mat * eyy_pred
    syy_pred = D12_mat * exx_pred + D22_mat * eyy_pred
    sxy_pred = D33_mat * e2xy_pred
    
    ust_pred = ust_pred_torch.detach().cpu().numpy()
    exx_pred = exx_pred.detach().cpu().numpy()
    eyy_pred = eyy_pred.detach().cpu().numpy()
    e2xy_pred = e2xy_pred.detach().cpu().numpy()
    sxx_pred = sxx_pred.detach().cpu().numpy()
    syy_pred = syy_pred.detach().cpu().numpy()
    sxy_pred = sxy_pred.detach().cpu().numpy()
    ust_pred = ust_pred_torch.detach().cpu().numpy()
    F11_pred = np.zeros(Nx) # 因为是小变形，所以我不关心这个量，先全部设为0
    F12_pred = np.zeros(Nx)
    F21_pred = np.zeros(Nx)
    F22_pred = np.zeros(Nx)
    surUx = ust_pred[:, 0]
    surUy = ust_pred[:, 1]
    surUz = np.zeros(Nx)
    surE11 = exx_pred
    surE12 = 0.5*e2xy_pred
    surE13 = np.zeros(Nx)
    surE21 = 0.5*e2xy_pred
    surE22 = eyy_pred
    surE23 = np.zeros(Nx)
    surE33 = np.zeros(Nx)
   
    surS11 = sxx_pred
    surS12 = sxy_pred
    surS13 = np.zeros(Nx)
    surS21 = sxy_pred
    surS22 = syy_pred
    surS23 = np.zeros(Nx)
    surS33 = np.zeros(Nx)

    
    SVonMises = np.float64(np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22) ** 2 + (-surS11) ** 2 + 6 * (surS12 ** 2))))
    U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
    U_mag = (np.float64(surUx)**2 + np.float64(surUy)**2 + np.float64(surUz)**2)**(0.5)
    return U, np.float64(U_mag), np.float64(surS11), np.float64(surS12), np.float64(surS13), np.float64(surS22), np.float64(
        surS23), \
           np.float64(surS33), np.float64(surE11), np.float64(surE12), \
           np.float64(surE13), np.float64(surE22), np.float64(surE23), np.float64(surE33), np.float64(
        SVonMises), \
           np.float64(F11_pred), np.float64(F12_pred), np.float64(F21_pred), np.float64(F22_pred)
# Training
def train_data(model, optimizer, epochs, dom_p):
    criterion = torch.nn.MSELoss()
    U_x_exact = np.load('../abaqus_reference/onehole/U_x.npy')[:,1]
    U_y_exact = np.load('../abaqus_reference/onehole/U_y.npy')[:,1]

    U_exact = np.hstack((U_x_exact.reshape(-1, 1), U_y_exact.reshape(-1, 1)))

    U_x_exact_tensor = torch.from_numpy(U_x_exact).float().cuda()
    U_y_exact_tensor = torch.from_numpy(U_y_exact).float().cuda()
    U_exact_tensor = torch.from_numpy(U_exact).float().cuda()

    Nx = len(dom_p)
    x = dom_p[:, 0].reshape(Nx, 1)
    y = dom_p[:, 1].reshape(Nx, 1)
    
    xy = np.concatenate((x, y), axis=1)
    xy_tensor = torch.from_numpy(xy).float()
    xy_tensor = xy_tensor.cuda()
    
    
    start_time = time.time()
    # train dis_x first
    for epoch in range(epochs):
        optimizer.zero_grad()
        ust_pred_torch = getUST(xy_tensor)
        if epoch % 1000 == 0:
            end_time = time.time()
            print('Time is ' + str(end_time-start_time))
        loss = criterion(ust_pred_torch, U_exact_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            #print(f'Epoch {epoch}, L_balance: {loss_balance.item()}, Ll_dis: {loss_l_x.item()}, Ll_f: {loss_l_y.item()}, Ld_dis: {loss_d_y.item()}, Ld_f: {loss_d_x.item()}, Lr: {loss_r.item()}, Lu: {loss_u.item()}, Lc: {loss_c.item()}')
            print(f'Epoch {epoch}, loss: {loss.item()}')



def train_inverse(optim_E_nu, epochs, dom_p, boundary_p):

    xy_dom_t = torch.tensor(dom_p, requires_grad=True).cuda() # 域内点
    X_l = torch.tensor(boundary_p['l'], requires_grad=True).cuda() # 边界点
    X_d = torch.tensor(boundary_p['d'], requires_grad=True).cuda()
    X_r = torch.tensor(boundary_p['r'], requires_grad=True).cuda()
    X_u = torch.tensor(boundary_p['u'], requires_grad=True).cuda()
    X_c = torch.tensor(boundary_p['c'], requires_grad=True).cuda()
    
    
    for epoch in range(epochs):

        
        optim_E_nu.zero_grad()
        D11_mat = E/(1-nu**2)
        D22_mat = E/(1-nu**2)
        D12_mat = E*nu/(1-nu**2)
        D21_mat = E*nu/(1-nu**2)
        D33_mat = E/(2*(1+nu))
        # u_pred = getUST(xy_dom_t)
        
        # duxdxy = grad(u_pred[:, 0].unsqueeze(1), xy_dom_t, torch.ones(xy_dom_t.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
        # duydxy = grad(u_pred[:, 1].unsqueeze(1), xy_dom_t, torch.ones(xy_dom_t.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
        # duxdx = duxdxy[:, 0].unsqueeze(1)
        # duxdy = duxdxy[:, 1].unsqueeze(1)
        
        # duydx = duydxy[:, 0].unsqueeze(1)
        # duydy = duydxy[:, 1].unsqueeze(1)
        
        # exx_pred = duxdx
        # eyy_pred = duydy
        # e2xy_pred = duxdy + duydx 
    
        # sxx_pred = D11_mat * exx_pred + D12_mat * eyy_pred
        # syy_pred = D12_mat * exx_pred + D22_mat * eyy_pred
        # sxy_pred = D33_mat * e2xy_pred
    
        # # 建立平衡方程
        # dsxxdx = grad(sxx_pred, xy_dom_t, torch.ones(sxx_pred.size()[0], 1, device=xy_dom_t.device), create_graph=True, retain_graph=True)[0][:, 0].unsqueeze(1)
        # dsyydy = grad(syy_pred, xy_dom_t, torch.ones(syy_pred.size()[0], 1, device=xy_dom_t.device), create_graph=True, retain_graph=True)[0][:, 1].unsqueeze(1)
        # dsxydx = grad(sxy_pred, xy_dom_t, torch.ones(sxy_pred.size()[0], 1, device=xy_dom_t.device), create_graph=True, retain_graph=True)[0][:, 0].unsqueeze(1)
        # dsxydy = grad(sxy_pred, xy_dom_t, torch.ones(sxy_pred.size()[0], 1, device=xy_dom_t.device), create_graph=True, retain_graph=True)[0][:, 1].unsqueeze(1)
        # # Balance equations
        # fx = torch.zeros_like(dsxxdx)  # External forces in x-direction (if any)
        # fy = torch.zeros_like(dsyydy)  # External forces in y-direction (if any)
        
        # balance_x = dsxxdx + dsxydy + fx
        # balance_y = dsyydy + dsxydx + fy  
        
        # loss_balance = torch.mean(balance_x**2) + torch.mean(balance_y**2)
    
    #  因为位移场是通过可能位移场构造的，所以不需要再考虑位移边界条件了，我们直接考虑力边界条件
        # 上，下，左还有环是相同的处理方式，我们先处理这些一样的
        # left
        # u_l = getUST(X_l)
        
        # duxdxy_l = grad(u_l[:, 0].unsqueeze(1), X_l, torch.ones(X_l.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
        # duydxy_l = grad(u_l[:, 1].unsqueeze(1), X_l, torch.ones(X_l.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
        # duxdx_l = duxdxy_l[:, 0].unsqueeze(1)
        # duxdy_l = duxdxy_l[:, 1].unsqueeze(1)
        
        # duydx_l = duydxy_l[:, 0].unsqueeze(1)
        # duydy_l = duydxy_l[:, 1].unsqueeze(1)
        
        # exx_pred_l = duxdx_l
        # eyy_pred_l = duydy_l
        # e2xy_pred_l = duxdy_l + duydx_l 
    
        # sxx_pred_l = D11_mat * exx_pred_l + D12_mat * eyy_pred_l
        # syy_pred_l = D12_mat * exx_pred_l + D22_mat * eyy_pred_l
        # sxy_pred_l = D33_mat * e2xy_pred_l
    
        # loss_l_x = 0
        # loss_l_y = torch.mean( (-sxy_pred_l)**2) 
        # # down
        # u_d = getUST(X_d)
        
        # duxdxy_d = grad(u_d[:, 0].unsqueeze(1), X_d, torch.ones(X_d.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
        # duydxy_d = grad(u_d[:, 1].unsqueeze(1), X_d, torch.ones(X_d.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
        # duxdx_d = duxdxy_d[:, 0].unsqueeze(1)
        # duxdy_d = duxdxy_d[:, 1].unsqueeze(1)
        
        # duydx_d = duydxy_d[:, 0].unsqueeze(1)
        # duydy_d = duydxy_d[:, 1].unsqueeze(1)
        
        # exx_pred_d = duxdx_d
        # eyy_pred_d = duydy_d
        # e2xy_pred_d = duxdy_d + duydx_d 
    
        # sxx_pred_d = D11_mat * exx_pred_d + D12_mat * eyy_pred_d
        # syy_pred_d = D12_mat * exx_pred_d + D22_mat * eyy_pred_d
        # sxy_pred_d = D33_mat * e2xy_pred_d
    
        # loss_d_x = torch.mean((-sxy_pred_d)**2)
        # loss_d_y = 0
    
        # # up
        # u_u = getUST(X_u)
        
        # duxdxy_u = grad(u_u[:, 0].unsqueeze(1), X_u, torch.ones(X_u.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
        # duydxy_u = grad(u_u[:, 1].unsqueeze(1), X_u, torch.ones(X_u.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
        # duxdx_u = duxdxy_u[:, 0].unsqueeze(1)
        # duxdy_u = duxdxy_u[:, 1].unsqueeze(1)
        
        # duydx_u = duydxy_u[:, 0].unsqueeze(1)
        # duydy_u = duydxy_u[:, 1].unsqueeze(1)
        
        # exx_pred_u = duxdx_u
        # eyy_pred_u = duydy_u
        # e2xy_pred_u = duxdy_u + duydx_u 
    
        # sxx_pred_u = D11_mat * exx_pred_u + D12_mat * eyy_pred_u
        # syy_pred_u = D12_mat * exx_pred_u + D22_mat * eyy_pred_u
        # sxy_pred_u = D33_mat * e2xy_pred_u
    
        # loss_u = torch.mean((sxy_pred_u)**2 + (syy_pred_u)**2) 
    
        # # circle
        # u_c = getUST(X_c)
        
        # duxdxy_c = grad(u_c[:, 0].unsqueeze(1), X_c, torch.ones(X_c.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
        # duydxy_c = grad(u_c[:, 1].unsqueeze(1), X_c, torch.ones(X_c.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
        # duxdx_c = duxdxy_c[:, 0].unsqueeze(1)
        # duxdy_c = duxdxy_c[:, 1].unsqueeze(1)
        
        # duydx_c = duydxy_c[:, 0].unsqueeze(1)
        # duydy_c = duydxy_c[:, 1].unsqueeze(1)
        
        # exx_pred_c = duxdx_c
        # eyy_pred_c = duydy_c
        # e2xy_pred_c = duxdy_c + duydx_c 
    
        # sxx_pred_c = D11_mat * exx_pred_c + D12_mat * eyy_pred_c
        # syy_pred_c = D12_mat * exx_pred_c + D22_mat * eyy_pred_c
        # sxy_pred_c = D33_mat * e2xy_pred_c
    
        # tx_c = -(sxx_pred_c * X_c[:, 0].unsqueeze(1)  +    sxy_pred_c * X_c[:, 1].unsqueeze(1))/Radius
        # ty_c = -(sxy_pred_c * X_c[:, 0].unsqueeze(1)  +    syy_pred_c * X_c[:, 1].unsqueeze(1))/Radius
        # loss_c = torch.mean(tx_c**2 + ty_c**2) 
    
    
        # right
        u_r = getUST(X_r)
        
        duxdxy_r = grad(u_r[:, 0].unsqueeze(1), X_r, torch.ones(X_r.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
        duydxy_r = grad(u_r[:, 1].unsqueeze(1), X_r, torch.ones(X_r.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
        duxdx_r = duxdxy_r[:, 0].unsqueeze(1)
        duxdy_r = duxdxy_r[:, 1].unsqueeze(1)
        
        duydx_r = duydxy_r[:, 0].unsqueeze(1)
        duydy_r = duydxy_r[:, 1].unsqueeze(1)
        
        exx_pred_r = duxdx_r
        eyy_pred_r = duydy_r
        e2xy_pred_r = duxdy_r + duydx_r 
    
        sxx_pred_r = D11_mat * exx_pred_r + D12_mat * eyy_pred_r
        syy_pred_r = D12_mat * exx_pred_r + D22_mat * eyy_pred_r
        sxy_pred_r = D33_mat * e2xy_pred_r
        
        # 试一下积分条件
        S_XX_PRED = sxx_pred_r*Length
        loss_r = torch.mean((S_XX_PRED-100*Length)**2) # 右边外力100
        
        #loss_r = torch.mean((sxx_pred_r-100)**2 + (sxy_pred_r)**2) # 右边外力100
        if epoch % 10 == 0:
            #print(f'Epoch {epoch}, L_balance: {loss_balance.item()}, Ll_dis: {loss_l_x.item()}, Ll_f: {loss_l_y.item()}, Ld_dis: {loss_d_y.item()}, Ld_f: {loss_d_x.item()}, Lr: {loss_r.item()}, Lu: {loss_u.item()}, Lc: {loss_c.item()}')
            print(f'Epoch {epoch}, Lr: {loss_r.item()}')

    
        loss = loss_r 
        loss.backward(retain_graph=True)
        optim_E_nu.step()        
 
        if epoch % 10 == 0:
            #print(f'Epoch {epoch}, L_balance: {loss_balance.item()}, Ll_dis: {loss_l_x.item()}, Ll_f: {loss_l_y.item()}, Ld_dis: {loss_d_y.item()}, Ld_f: {loss_d_x.item()}, Lr: {loss_r.item()}, Lu: {loss_u.item()}, Lc: {loss_c.item()}')
            print(f'Epoch {epoch}, E: {E.item()}')    
def write_vtk_v2p(filename, dom, U, U_mag, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, U_mag_error, MISES_error): #已经将输出的感兴趣场进行了分类VTK导出
    xx = np.ascontiguousarray(dom[:, 0])
    yy = np.ascontiguousarray(dom[:, 1])
    zz = np.ascontiguousarray(dom[:, 2]) # 点的VTK
    pointsToVTK(filename, xx, yy, zz, data={"displacementX": U[0], "displacementY": U[1], "displacementZ": U[2],\
                                            "S-VonMises": SVonMises, "U-mag": U_mag, "U_mag_error": U_mag_error, "MISES_error": MISES_error,\
                                               "S11": S11, "S12": S12, "S13": S13, \
                                               "S22": S22, "S23": S23, "S33": S33, \
                                               "E11": E11, "E12": E12, "E13": E13, \
                                               "E22": E22, "E23": E23, "E33": E33\
                                               })
# Main function
if __name__ == '__main__':
    Radius = 5.
    Length = 20.
    
    model = KAN([2, 5,5,5,2], base_activation=torch.nn.SiLU, grid_size=10, grid_range=[0, 1.0], spline_order=3).cuda()

    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs_data = 5000

    dom_p, boundary_p = generate_training_data()
    
    N = dom_p.shape[0] # abaqus最后一个维度必须是0，就是必须要输入三位坐标才行
    dom_aba = np.hstack((dom_p, np.zeros((N, 1))))
    np.save('../abaqus_reference/onehole/coordinate_inverse.npy', dom_aba)
    
    #train_data(model, optimizer, epochs_data, dom_p)

    # 保存模型
    #torch.save(model.state_dict(), './model/model_kan.pth')
    
    # 加载模型
    model.load_state_dict(torch.load('./model/model_kan.pth'))


    E = torch.tensor(1200.,  requires_grad=True, device='cuda') # true 1000
    nu = 0.3
    G = E/2/(1+nu)
    D11_mat = E/(1-nu**2)
    D22_mat = E/(1-nu**2)
    D12_mat = E*nu/(1-nu**2)
    D21_mat = E*nu/(1-nu**2)
    D33_mat = E/(2*(1+nu))


    z = np.zeros((dom_p.shape[0], 1))
    datatest = np.concatenate((dom_p, z), 1)
    U, U_mag, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22 = evaluate_model(datatest)
    # 储存一下error contourf
    U_mag_exact = np.load('../abaqus_reference/onehole/U_mag_inverse.npy')[:,1]
    MISES_exact = np.load('../abaqus_reference/onehole/MISES_inverse.npy')[:,1]
    U_mag_error = np.abs(U_mag_exact - U_mag)
    MISES_error = np.abs(MISES_exact - SVonMises)

    filename_out =  '../results/KINN_PINN_inverse/'
    
    write_vtk_v2p(filename_out+'Plate_hole_KINN_PINNs', datatest, U, U_mag, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, U_mag_error, MISES_error)
    surUx, surUy, surUz = U
    

    
    hyper_d = 10
    hyper_l_x = 2000 # x方向的左侧位移，这个容易很低，所以放大一些
    hyper_l_y = 1
    hyper_d_x = 1
    hyper_d_y = 2000 # y方向的位移，这个容易很低，所以放大一些
    hyper_r = 1
    hyper_u = 1
    hyper_c = 1

    epochs_pde = 30000

    optim_E_nu = torch.optim.Adam([{'params': E, 'lr': 0.1}])
    
    train_inverse(optim_E_nu, epochs_pde, dom_p, boundary_p)
    
