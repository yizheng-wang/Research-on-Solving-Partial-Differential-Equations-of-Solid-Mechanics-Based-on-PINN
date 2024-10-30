# PINN可能位移场的能量形式，不进行分片
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
from sklearn.neighbors import KDTree
import time
from itertools import chain
import matplotlib.tri as tri
from pyevtk.hl import gridToVTK
from pyevtk.hl import pointsToVTK
import sys
sys.path.append("../") 
from kan_efficiency import *



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(2024)

train_p = 0
tol_p = 0.00001

a1 = 1/15
a2 = 1
r0 = 0.5
nepoch = 100000


hyper_d = 1
hyper_b = 5000


def u(xy):
    r = torch.norm(xy, dim=1)
    u_values = torch.where(r < r0,
                           r**4 / a1,
                           r**4 / a2 + r0**4 * (1/a1 - 1/a2))
    return u_values

def domain():
    num_points_per_side = 300
    x = torch.linspace(-1, 1, num_points_per_side)
    y = torch.linspace(-1, 1, num_points_per_side)
    
    # 使用meshgrid生成正方形内部的均匀点
    xx, yy = torch.meshgrid(x, y)
    internal_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return internal_points

def domain_tri():
    def generate_points(plate_size, hole_center, hole_radius, num_points):
        points = []
        for i in range(num_points):
            for j in range(num_points):
                x = i * plate_size / (num_points - 1) - plate_size / 2
                y = j * plate_size / (num_points - 1) - plate_size / 2
                if (x - hole_center[0])**2 + (y - hole_center[1])**2 >= hole_radius**2:
                    points.append((x, y))
        return np.array(points)

    def generate_circle_points(center, radius, num_points):
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        points = np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))
        return points

    # Constants
    x_min, Length = -1, 2
    y_min, Height = -1, 2
    num_points = 200
    hole_center = (0, 0)
    hole_radius = 0.5

    # Generate the points
    points = generate_points(Length, hole_center, hole_radius, num_points)
    
    # Create a Delaunay triangulation
    triangulation = tri.Triangulation(points[:, 0], points[:, 1])
    

    
    # Calculate the area of each triangle
    triangles = points[triangulation.triangles]
    a = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    b = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
    s = 0.5 * (a + b + c)
    areas = np.sqrt(s * (s - a) * (s - b) * (s - c))
    
    # Calculate the center of each triangle
    dom_point = triangles.mean(axis=1)
    
    # Combine the centers and areas into a single array
    dom = np.hstack((dom_point, areas[:, np.newaxis]))
    
    # Determine which triangles are inside the hole
    distances = np.sqrt(np.sum((dom_point - hole_center)**2, axis=1))
    outside_hole_index = distances >= hole_radius
    
    # Filter out the triangles inside and outside the hole
    outside_points = dom[outside_hole_index].astype(np.float32)
    
    # Total area of the mesh inside and outside the hole
    total_area_outside = np.sum(outside_points[:, 2])
# =============================================================================
#     # 上面是划分外部区域三角形，下面是内部圆形
# =============================================================================
    
    num_boundary_points = 1000
    num_internal_points = 10000
    
    boundary_points_c = generate_circle_points(hole_center, hole_radius, num_boundary_points)
    
    # Generate random internal points within the circle
    internal_points = []
    while len(internal_points) < num_internal_points:
        x, y = np.random.uniform(-hole_radius, hole_radius, 2)
        if x**2 + y**2 <= hole_radius**2:
            internal_points.append((x, y))
    internal_points = np.array(internal_points)
    
    # Combine boundary and internal points
    points_c = np.vstack((boundary_points_c, internal_points))

    # Create a Delaunay triangulation
    triangulation_c = tri.Triangulation(points_c[:, 0], points_c[:, 1])

    
    # Calculate the area of each triangle
    triangles_c = points_c[triangulation_c.triangles]
    a = np.linalg.norm(triangles_c[:, 0] - triangles_c[:, 1], axis=1)
    b = np.linalg.norm(triangles_c[:, 1] - triangles_c[:, 2], axis=1)
    c = np.linalg.norm(triangles_c[:, 2] - triangles_c[:, 0], axis=1)
    s = 0.5 * (a + b + c)
    areas_c = np.sqrt(s * (s - a) * (s - b) * (s - c))
    
    # Calculate the center of each triangle
    dom_point_c = triangles_c.mean(axis=1)
    
    # Combine the centers and areas into a single array
    dom_c = np.hstack((dom_point_c, areas_c[:, np.newaxis]))
    
    distances_c = np.sqrt(np.sum((dom_point_c - hole_center)**2, axis=1))
    
    inside_hole_index = distances_c <= hole_radius
    # Filter out the triangles inside and outside the hole
    hole_points = dom_c[inside_hole_index].astype(np.float32)
    
    total_area_inside = np.sum(hole_points[:, 2])
    
    print(f"Total area of the mesh outside the hole: {total_area_outside}")
    print(f"Total area of the mesh inside the hole: {total_area_inside}")
    
    return hole_points, outside_points



def write_arr2DVTK(filename, coordinates, values, name):
    '''
    这是一个将tensor转化为VTK数据可视化的函数，输入coordinates坐标，是一个二维的二列数据，
    value是相应的值，是一个列向量
    name是转化的名称，比如RBF距离函数，以及特解网路等等
    '''
    # displacement = np.concatenate((values[:, 0:1], values[:, 1:2], values[:, 0:1]), axis=1)
    x = np.array(coordinates[:, 0].flatten(), dtype='float32')
    y = np.array(coordinates[:, 1].flatten(), dtype='float32')
    z = np.zeros(len(coordinates), dtype='float32')
    disX = np.array(values[:, 0].flatten(), dtype='float32')
    pointsToVTK(filename, x, y, z, data={name: disX}) # 二维数据的VTK文件导出
    
class MultiLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
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



def pred(xy):
    '''
    

    Parameters
    ----------
    输入xytensor，然而输出位移，都是GPU的张量
    '''
    pred =  model(xy)
    
    # pred = torch.zeros([len(xy), 1]).cuda()
    # pred[(xy[:,0]**2+xy[:,1]**2)<r0**2] = 1/a1*torch.norm(xy[(xy[:,0]**2+xy[:,1]**2)<r0**2], dim=1, keepdim=True)**4 # 现在网格里面赋值
    # pred[(xy[:,0]**2+xy[:,1]**2)>=r0**2] = 1/a2*torch.norm(xy[(xy[:,0]**2+xy[:,1]**2)>=r0**2], dim=1, keepdim=True)**4 + r0**4*(1/a1-1/a2) # 现在网格外面赋值

    
    return pred    
    
def evaluate():
    N_test = 100
    dom_koch_n = domain() # 均匀在科赫雪花内部步点
    dom_koch_t = torch.tensor(dom_koch_n, device = 'cuda')
    
    u_pred = pred(dom_koch_t)
    u_pred = u_pred.data.cpu()
    

    u_exact = np.zeros((len(dom_koch_n) ,1))
    u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2] = 1/a1*torch.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2], dim=1, keepdim=True)**4 # 现在网格里面赋值
    u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2] = 1/a2*torch.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2], dim=1, keepdim=True)**4 + r0**4*(1/a1-1/a2) # 现在网格外面赋值

    u_exact = torch.from_numpy(u_exact)
    error = torch.abs(u_pred - u_exact) # get the error in every points
    error_t = torch.norm(error)/torch.norm(u_exact) # get the total relative L2 error
    return error_t

def boundary():
    # 生成从-1到1的1000个点
    points = torch.linspace(-1, 1, 1000)
    
    # 上边界 (-1, 1) 到 (1, 1)
    top_edge = torch.stack((points, torch.ones(1000)), dim=1)
    
    # 下边界 (-1, -1) 到 (1, -1)
    bottom_edge = torch.stack((points, -torch.ones(1000)), dim=1)
    
    # 左边界 (-1, -1) 到 (-1, 1)
    left_edge = torch.stack((-torch.ones(1000), points), dim=1)
    
    # 右边界 (1, -1) 到 (1, 1)
    right_edge = torch.stack((torch.ones(1000), points), dim=1)
    
    # 合并所有边界点
    all_edges = torch.cat((top_edge, bottom_edge, left_edge, right_edge), dim=0)
    return all_edges

if train_p == 1:
    start = time.time()
    model_p = MultiLayerNet(2, 20, 1).cuda() # 定义两个特解，这是因为在裂纹处是间断值

    
    loss_bn = 100
    epoch_b = 0
    criterion = torch.nn.MSELoss()
    optimp = torch.optim.Adam(params=model_p.parameters(), lr= 0.0005)

    loss_bn_array = []
    # training the particular network for particular solution on the bonudaray
    while loss_bn>tol_p:
        
        if epoch_b%10 == 0:
            Xb = boundary()
            Xb = torch.tensor(Xb).cuda()
            target_b = 1/a2*torch.norm(Xb, dim=1, keepdim=True)**4+(1/a1-1/a2)*r0**4

        epoch_b = epoch_b + 1
        def closure():  
            pred_b = model_p(Xb) # predict the boundary condition
            loss_bn = criterion(pred_b, target_b)  
           
            optimp.zero_grad()
            loss_bn.backward()
            loss_bn_array.append(loss_bn.data)
            if epoch_b%10==0:
                print('trianing particular network : the number of epoch is %i, the loss is %f' % (epoch_b, loss_bn.data))
            return loss_bn
        optimp.step(closure)
        loss_bn = loss_bn_array[-1]
    torch.save(model_p, './particular_nn')

model_p = torch.load('particular_nn')


#model = MultiLayerNet(2,30,1).cuda()
model = KAN([2, 5, 5, 1], base_activation=torch.nn.SiLU, grid_size=20, grid_range=[-1, 1.0], spline_order=3).cuda()
criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(params=model.parameters(), lr= 0.001) # 0.001比较好，两个神经网络

scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[10000, 30000], gamma = 0.1)

loss_array = []
loss_d1_array = [] # 内部r<ro
loss_d2_array = [] # 内部r>ro
loss_b_array = [] # 内部加上边界
error_array = []


start = time.time()

for epoch in range(nepoch): # cpinn先优化1
    if epoch ==2000:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0:
        hole, outside = domain_tri() # 获得n_test个koch的随机分布点

        dom_koch_t1= torch.tensor(hole[:,:-1],  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor
        dom_koch_t2= torch.tensor(outside[:,:-1],  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor
        
        hole_J = torch.from_numpy(hole[:,-1, np.newaxis]).cuda()
        outside_J = torch.from_numpy(outside[:,-1, np.newaxis]).cuda()
        
        f1 = -16 * torch.norm(dom_koch_t1, dim = 1, keepdim=True).data**2 # 定义体力 
        f2 = -16 * torch.norm(dom_koch_t2, dim = 1, keepdim=True).data**2 # 定义体力

        Xb = boundary().cuda() # 获得边界点，用来优化网络2的外部
        target_b = 1/a2*torch.norm(Xb, dim=1, keepdim=True)**4+(1/a1-1/a2)*r0**4 # 得到相应的标签   
    def closure():  

        # 构造可能位移场
        u_pred1 = pred(dom_koch_t1)
        u_pred2 = pred(dom_koch_t2)
        
        du1dxy = grad(u_pred1, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
        du1dx = du1dxy[:, 0].unsqueeze(1)
        du1dy = du1dxy[:, 1].unsqueeze(1)

        du1dxxy = grad(du1dx, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
        du1dxx = du1dxxy[:, 0].unsqueeze(1)
        du1dxy = du1dxxy[:, 1].unsqueeze(1)




        du2dxy = grad(u_pred2, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
        du2dx = du2dxy[:, 0].unsqueeze(1)
        du2dy = du2dxy[:, 1].unsqueeze(1)

        du2dxy = grad(u_pred2, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
        du2dx = du2dxy[:, 0].unsqueeze(1)
        du2dy = du2dxy[:, 1].unsqueeze(1)



        J1 =  torch.sum((0.5 * a1 * (du1dx**2 + du1dy**2) - f1*u_pred1) * hole_J)
        
        J2 =  torch.sum((0.5 * a2 * (du2dx**2 + du2dy**2) - f2*u_pred2) * outside_J)
        
        u_b = pred(Xb)
        Jb = criterion(u_b, target_b)
        
        loss = J1+J2 + hyper_b*Jb
        
        error_t = evaluate()
        optim.zero_grad()
        loss.backward()
        loss_array.append(loss.data.cpu())
        loss_d1_array.append(J1.data.cpu())
        loss_d2_array.append(J2.data.cpu())
        loss_b_array.append(Jb.data.cpu())
        error_array.append(error_t.data.cpu())

        if epoch%10==0:
            print(' epoch : %i, the loss : %f , loss_d1 : %f, loss_d2 : %f, loss_b : %f, error : %f' % (epoch, loss.data, J1.data, J2.data,  Jb.data, error_t.data))
        return loss
    optim.step(closure)
    scheduler.step()
    # 网络1损失函数下不去，所以我们用交界面来作为本质边界条件训练网络1
        

# %%   
n_test = 300
dom_koch_n = domain() # 获得n_test个koch的随机分布点
xy_test= torch.tensor(dom_koch_n,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor


u_pred = pred(xy_test)
u_pred = u_pred.data.cpu()

u_exact = np.zeros((len(dom_koch_n) ,1))
u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2] = 1/a1*torch.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2], dim=1, keepdim=True)**4 # 现在网格里面赋值
u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2] = 1/a2*torch.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2], dim=1, keepdim=True)**4 + r0**4*(1/a1-1/a2) # 现在网格外面赋值

u_exact = torch.from_numpy(u_exact.astype(np.float32)) # 将精确解从array变成tensor

error = torch.abs(u_pred - u_exact) # get the error in every points
error_t = torch.norm(error)/torch.norm(u_exact) # get the total relative L2 error

# # plot the prediction solution
fig = plt.figure(figsize=(20, 20))

plt.subplot(3, 3, 1)
Xb =  boundary()
Xf = domain()
  # 将边界和内部点画图
plt.scatter(Xb[:, 0], Xb[:, 1], c='r', s = 0.1)
plt.scatter(Xf[:, 0], Xf[:, 1], c='b', s = 0.1)

write_arr2DVTK('./output_ntk/DEM_KINN/pred_dem_KINN', dom_koch_n, u_pred, 'pred_KINN')

write_arr2DVTK('./output_ntk/DEM_KINN/error_dem_KINN', dom_koch_n, error, 'error_KINN')

plt.subplot(3, 3, 2)
loss_d1_array = np.array(loss_d1_array)
loss_d2_array = np.array(loss_d2_array)
loss_b_array = np.array(loss_b_array)
plt.yscale('log')
plt.plot(loss_d1_array)
plt.plot(loss_d2_array)
plt.plot(loss_b_array)
plt.legend(['loss_d1', 'loss_d2', 'loss_b'])
plt.xlabel('the iteration')
plt.ylabel('loss')
plt.title('loss evolution') 

plt.subplot(3, 3, 3)
error_array = np.array(error_array)
plt.yscale('log')
plt.plot(error_array)
plt.xlabel('the iteration')
plt.ylabel('error')
plt.title('relative total error evolution') 

np.save('./results/DEM_KINN/error_L2.npy', error_array)

plt.subplot(3, 3, 4) # 做x=0的位移预测
n_test = 100
x0 = np.zeros((n_test, 2))
x0[:, 1] = np.linspace(-1, 1, 100)
exactx0 = np.zeros((n_test, 1))
exactx0[np.linalg.norm(x0, axis=1)<r0] =  1/a1*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)<r0], axis=1, keepdims=True)**4 # 不同的区域进行不同的解析解赋予
exactx0[np.linalg.norm(x0, axis=1)>=r0] =  1/a2*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)>=r0], axis=1, keepdims=True)**4 + (1/a1-1/a2)*r0**4
x0t = torch.tensor(x0.astype(np.float32)).cuda()
predx0 = pred(x0t).data.cpu().numpy() # 预测x=0的原函数
plt.plot(x0[:, 1], exactx0.flatten())
plt.plot(x0[:, 1], predx0.flatten())
plt.xlabel('y')
plt.ylabel('u')
plt.legend(['exact', 'pred'])
plt.title('x=0 u') 

x0_pred_save = np.vstack([x0[:, 1], predx0.flatten()]).T
np.save('./results/DEM_KINN/x0_u_pred.npy', x0_pred_save)
x0_exact_save = np.vstack([x0[:, 1], exactx0.flatten()]).T
np.save('./results/DEM_KINN/x0_u_exact.npy', x0_exact_save)

plt.subplot(3, 3, 5) # 做y=0的位移预测
n_test = 100
y0 = np.zeros((n_test, 2))
y0[:, 0] = np.linspace(-1, 1, 100)
exacty0 = np.zeros((n_test, 1))
exacty0[np.linalg.norm(y0, axis=1)<r0] =  1/a1*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)<r0], axis=1, keepdims=True)**4 # 不同的区域进行不同的解析解赋予
exacty0[np.linalg.norm(y0, axis=1)>=r0] =  1/a2*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)>=r0], axis=1, keepdims=True)**4 + (1/a1-1/a2)*r0**4
y0t = torch.tensor(y0.astype(np.float32)).cuda()
predy0 = pred(y0t).data.cpu().numpy() # 预测x=0的原函数
plt.plot(y0[:, 0], exacty0.flatten())
plt.plot(y0[:, 0], predy0.flatten())
plt.xlabel('x')
plt.ylabel('u')
plt.legend(['exact', 'pred'])
plt.title('y=0 u') 

plt.subplot(3, 3, 6) # 做x=0的位移导数预测 dudy
n_test = 100
x0 = np.zeros((n_test, 2))
x0[:, 1] = np.linspace(-1, 1, 100)
exactdx0 = np.zeros((n_test, 1))
exactdx0[np.linalg.norm(x0, axis=1)<r0] =  4/a1*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)<r0], axis=1, keepdims=True)**3 # 不同的区域进行不同的解析解赋予
exactdx0[np.linalg.norm(x0, axis=1)>=r0] =  4/a2*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)>=r0], axis=1, keepdims=True)**3
x0t = torch.tensor(x0.astype(np.float32), requires_grad=True).cuda() # 将numpy 变成 tensor
predx0 = pred(x0t) # 预测一下
dudxy = grad(predx0, x0t, torch.ones(x0t.size()[0], 1).cuda(), create_graph=True)[0]
dudx = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
dudy = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
dudy[x0[:, 1]<0] = -dudy[x0[:, 1]<0] # y小于0的导数添加负号
plt.plot(x0[:, 1], exactdx0.flatten())
plt.plot(x0[:, 1], dudy.flatten())
plt.xlabel('y')
plt.ylabel('u')
plt.legend(['exact', 'pred'])
plt.title('x=0 dudy')

x0dudy_pred_save = np.vstack([x0[:, 1], dudy.flatten()]).T
np.save('./results/DEM_KINN/x0_dudy_pred.npy', x0dudy_pred_save)
x0dudy_exact_save = np.vstack([x0[:, 1], exactdx0.flatten()]).T
np.save('./results/DEM_KINN/x0_dudy_exact.npy', x0dudy_exact_save)

plt.subplot(3, 3, 7) # 做y=0的位移导数预测 dudy
n_test = 100
y0 = np.zeros((n_test, 2))
y0[:, 0] = np.linspace(-1, 1, 100)
exactdy0 = np.zeros((n_test, 1))
exactdy0[np.linalg.norm(y0, axis=1)<r0] =  4/a1*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)<r0], axis=1, keepdims=True)**3 # 不同的区域进行不同的解析解赋予
exactdy0[np.linalg.norm(y0, axis=1)>=r0] =  4/a2*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)>=r0], axis=1, keepdims=True)**3
y0t = torch.tensor(y0.astype(np.float32), requires_grad=True).cuda()
predy0 = pred(y0t) # 预测x=0的原函数
dudxy = grad(predy0, y0t, torch.ones(y0t.size()[0], 1).cuda(), create_graph=True)[0]
dudx = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
dudy = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
dudx[y0[:, 0]<0] = -dudx[y0[:, 0]<0] # y小于0的导数添加负号
plt.plot(y0[:, 0], exactdy0.flatten())
plt.plot(y0[:, 0], dudx.flatten())
plt.xlabel('x')
plt.ylabel('u')
plt.legend(['exact', 'pred'])
plt.title('y=0 dudx') 


plt.suptitle("pinn")
plt.savefig('./pic/dem_KINN_plane.png')
plt.show()
print('the relative error is %f' % error_t.data)   
    
# 在0.5附近放大
N_u = 100
Y_u_05_z = np.linspace(0.49, 0.51, N_u)
x0_z = np.zeros_like(Y_u_05_z)
x0y0_z = np.vstack([x0_z, Y_u_05_z]).T
exact_u_x0_z = u(torch.tensor(x0y0_z, dtype=torch.float32)).numpy()
x0y0_tensor_z = torch.tensor(x0y0_z, dtype=torch.float32).cuda()
pred_u_x0_z = pred(x0y0_tensor_z).cpu().data.numpy()

u_x0_z_pred_save = np.vstack([Y_u_05_z, pred_u_x0_z.flatten()]).T
np.save('./results/DEM_KINN/x0_u_z_pred.npy', u_x0_z_pred_save)
x0_exact_save = np.vstack([Y_u_05_z, exact_u_x0_z.flatten()]).T
np.save('./results/DEM_KINN/x0_u_z_exact.npy', x0_exact_save)
    