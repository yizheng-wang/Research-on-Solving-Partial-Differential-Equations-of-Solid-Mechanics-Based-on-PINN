import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from  torch.autograd import grad
from pyevtk.hl import gridToVTK
from pyevtk.hl import pointsToVTK
import xlrd
import time


U_mag_exact = np.load('./U_mag.npy')[:,1]
MISES_exact = np.load('./MISES.npy')[:,1]
coord = np.load('./coordinate.npy')

def write_vtk_v2p_fem(filename, dom, U_mag, SVonMises): #已经将输出的感兴趣场进行了分类VTK导出
    xx = np.ascontiguousarray(dom[:, 0])
    yy = np.ascontiguousarray(dom[:, 1])
    zz = np.ascontiguousarray(dom[:, 2]) # 点的VTK
    pointsToVTK(filename, xx, yy, zz,  data = {"U-mag": np.ascontiguousarray(U_mag), 'Mise': np.ascontiguousarray(SVonMises)})
    
write_vtk_v2p_fem('FEM_reference', coord, U_mag_exact, MISES_exact)



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
    ust = model(x_scale)
    Ux =  ust[:, 0] # 如果x坐标是0的话，x方向位移也是0
    Uy =  ust[:, 1] # 如果y坐标是0的话，y方向位移也是0

    Ux = Ux.reshape(Ux.shape[0], 1)
    Uy = Uy.reshape(Uy.shape[0], 1)

    ust_pred = torch.cat((Ux, Uy), -1)
    return ust_pred


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
    points = np.array(points) # 这里必须64，才划分网格不会和32混淆
    
    # Create a Delaunay triangulation
    triangulation = tri.Triangulation(points[:, 0], points[:, 1])
    triangles = points[triangulation.triangles]
    # Sum of all triangle areas
    dom_point = triangles.mean(1)

    
    # 删除大于radius的点
    distances = np.sqrt(np.sum(dom_point**2, axis=1))
    filtered_points = dom_point[distances >= hole_radius]



    # Generate points along the vertical and horizontal boundaries
    left_wall = np.array([[0, y] for y in np.linspace(0, plate_size, boundary_num)]).astype(np.float32)
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
E = 1000
nu = 0.3
G = E/2/(1+nu)
D11_mat = E/(1-nu**2)
D22_mat = E/(1-nu**2)
D12_mat = E*nu/(1-nu**2)
D21_mat = E*nu/(1-nu**2)
D33_mat = E/(2*(1+nu))

Radius = 5.
Length = 20.
model = torch.load('../model/PINNs_MLP/PINNs_MLP.pth')

dom_p, boundary_p = generate_training_data()

z = np.zeros((dom_p.shape[0], 1))
datatest = np.concatenate((dom_p, z), 1)
U, U_mag, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22 = evaluate_model(datatest)
# 储存一下error contourf
U_mag_exact = np.ascontiguousarray(np.load('../abaqus_reference/U_mag.npy')[:,1])
MISES_exact = np.ascontiguousarray(np.load('../abaqus_reference/MISES.npy')[:,1])
U_mag_error = np.abs(U_mag_exact - U_mag)
MISES_error = np.abs(MISES_exact - SVonMises)
filename_out =  '../results/PINNs_MLP_penalty/'
write_vtk_v2p(filename_out+'Plate_hole_PINNs', datatest, U, U_mag, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, U_mag_error, MISES_error)