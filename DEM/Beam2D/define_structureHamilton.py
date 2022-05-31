import sys
sys.path.insert(0, '/home/sg/SeaDrive/My Libraries/开题报告/超材料的PINN/dem_hyperelasticity-master') # add路径
from dem_hyperelasticity.config import *
from dem_hyperelasticity.Beam2D.config import *


def setup_domain(): # 2D悬臂梁撒点
    x_dom = x_min, Length, Nx
    y_dom = y_min, Height, Ny
    t_dom = t_min, Period, Nt
    # create points
    lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
    lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
    lin_t = np.linspace(t_dom[0], t_dom[1], t_dom[2])
    dom = np.zeros((Nt, Nx * Ny, 3))
    
    m = 0
    # 对dom进行时间维度的赋值
    for t in np.nditer(lin_t):
        dom[m, :, 2] = t # 将dom每一个时间步进行赋值
        c_count = 0
        for x in np.nditer(lin_x):
            tb = y_dom[2] * c_count
            te = tb + y_dom[2]
            c_count += 1
            dom[m, tb:te, 0] = float(x)
            dom[m, tb:te, 1] = lin_y
        m += 1
    print(dom.shape)
    dom_plt = dom.reshape(Nx*Ny*Nt, 3) # 为了构建可以画图的梁
    #np.meshgrid(lin_x, lin_y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') # 生成一个子图，1*1的第一个，作为ax对象
    ax.scatter(dom_plt[:, 0], dom_plt[:, 1], dom_plt[:, 2], s=0.005, facecolor='blue')
    ax.set_xlabel('X', fontsize=3)
    ax.set_ylabel('Y', fontsize=3)
    ax.set_zlabel('Z', fontsize=3)
    ax.set(xlim = (0, Length+1), ylim = (0, Height+1), zlim = (0, Period + 1)) # 将坐标轴的尺度改为正常，否则由于yz过小，会显得结构非常奇怪
    ax.tick_params(labelsize=4)
    ax.view_init(elev=120., azim=-90) # 这是个3D的图像，所以这是调整视角的参数
    # ------------------------------------ BOUNDARY ----------------------------------------
    # Left boundary condition (Dirichlet BC)
    bcl_u_pts_idx_left = np.where(dom[:, :, 0] == x_min) # 找到相应边界的点的位置
    bcl_u_pts_left_plt = dom[bcl_u_pts_idx_left] # 找到相应边界点的坐标
    bcl_u_pts_left = bcl_u_pts_left_plt.reshape(Nt, Ny, 3) # 得到和dom一样的数据结构
    bcl_u_left = np.ones(np.shape(bcl_u_pts_left)) * [known_left_ux, known_left_uy, 0] # 找到相应边界点的值
    bcl_u_left[:, :, 2] = bcl_u_pts_left[:, :, 2] # 时间轴更新一下

    # Right boundary condition (Neumann BC)
    bcl_t_pts_idx_right = np.where(dom[:, :, 0] == Length) # 找到相应边界的点的位置
    bcl_t_pts_right_plt = dom[bcl_t_pts_idx_right] # 找到相应边界点的坐标
    bcl_t_pts_right = bcl_t_pts_right_plt.reshape(Nt, Ny, 3) # 得到和dom一样的数据结构
    if bc_right_style == 'sin':
        bcl_t_right = np.ones(np.shape(bcl_t_pts_right))  # 找到相应边界点的值
        one_sin = np.sin(lin_t) # 每一个时刻的力,单位力的变换，之后乘以一个幅值就行了
        for index, et in enumerate(one_sin):
            bcl_t_right[index] = [ et * known_right_tx, et * known_right_ty, 0]
        bcl_t_right[:, :, 2] = bcl_t_pts_right[:, :, 2] # 时间轴更新一下
    
    # Down boundary condition (Neumann BC)
    bcl_t_pts_idx_down = np.where(dom[:, :, 1] == y_min) # 找到相应边界的点的位置
    bcl_t_pts_down_plt = dom[bcl_t_pts_idx_down] # 找到相应边界点的坐标
    bcl_t_pts_down = bcl_t_pts_down_plt.reshape(Nt, Nx, 3) # 得到和dom一样的数据结构
    bcl_t_down = np.ones(np.shape(bcl_t_pts_down)) * [known_down_tx, known_down_ty, 0]    
    bcl_t_down[:, :, 2] = bcl_t_pts_down[:, :, 2] # 时间轴更新一下
    # Up boundary condition (Neumann BC)
    bcl_t_pts_idx_up = np.where(dom[:, :, 1] == Height) # 找到相应边界的点的位置
    bcl_t_pts_up_plt = dom[bcl_t_pts_idx_up] # 找到相应边界点的坐标
    bcl_t_pts_up = bcl_t_pts_up_plt.reshape(Nt, Nx, 3) # 得到和dom一样的数据结构
    bcl_t_up = np.ones(np.shape(bcl_t_pts_up)) * [known_up_tx, known_up_ty, 0]    
    bcl_t_up[:, :, 2] = bcl_t_pts_up[:, :, 2] # 时间轴更新一下   
    # ------------------------------------ INITIAL ----------------------------------------
    initial_displacement = np.ones((Nx*Ny, 2)) * initial_d # 定义初始位移条件
    initial_velocity = np.ones((Nx*Ny, 2)) * initial_v # 定义初始速度条件 
    
    
    
    
    
    

    ax.scatter(bcl_u_pts_left_plt[:, 0], bcl_u_pts_left_plt[:, 1], bcl_u_pts_left_plt[:, 2], s=0.5, facecolor='red')
    ax.scatter(bcl_t_pts_down_plt[:, 0], bcl_t_pts_down_plt[:, 1], bcl_t_pts_down_plt[:, 2], s=0.5, facecolor='green')
    ax.scatter(bcl_t_pts_right_plt[:, 0], bcl_t_pts_right_plt[:, 1], bcl_t_pts_right_plt[:, 2], s=0.5, facecolor='green')
    ax.scatter(bcl_t_pts_up_plt[:, 0], bcl_t_pts_up_plt[:, 1], bcl_t_pts_up_plt[:, 2], s=0.5, facecolor='green')
    plt.show()
    # exit()
    boundary_neumann = {
        # condition on the right * *******************************************************************************************
        "neumann_right": {
            "coord": bcl_t_pts_right,
            "known_value": bcl_t_right,
            "penalty": bc_right_penalty,
            "boundary_normal2d": bc_right_normal2d
        },
        "neumann_down": {
            "coord": bcl_t_pts_down,
            "known_value": bcl_t_down,
            "penalty": bc_down_penalty,
            "boundary_normal2d": bc_down_normal2d
        },
        "neumann_up": {
            "coord": bcl_t_pts_up,
            "known_value": bcl_t_up,
            "penalty": bc_up_penalty,
            "boundary_normal2d": bc_up_normal2d
        }        
        # adding more boundary condition here ...
    }
    boundary_dirichlet = {
        # condition on the left
        "dirichlet_1": {
            "coord": bcl_u_pts_left,
            "known_value": bcl_u_left,
            "penalty": bc_left_penalty,
            "boundary_normal2d": bc_left_normal2d
        }

        # adding more boundary condition here ...
    }
    return dom, boundary_neumann, boundary_dirichlet, initial_displacement, initial_velocity


# -----------------------------------------------------------------------------------------------------
# prepare inputs for testing the model
# -----------------------------------------------------------------------------------------------------
def get_datatest(Nx=num_test_x, Ny=num_test_y):
    x_dom_test = x_min, Length, Nx
    y_dom_test = y_min, Height, Ny
    # create points
    x_space = np.linspace(x_dom_test[0], x_dom_test[1], x_dom_test[2])
    y_space = np.linspace(y_dom_test[0], y_dom_test[1], y_dom_test[2])
    xGrid, yGrid = np.meshgrid(x_space, y_space)
    data_test = np.concatenate(
        (np.array([xGrid.flatten()]).T, np.array([yGrid.flatten()]).T), axis=1)
    return x_space, y_space, data_test

def get_datatestdom(Nx=num_test_x, Ny=num_test_y, Nt = num_test_t):
    x_dom = x_min, Length, Nx
    y_dom = y_min, Height, Ny
    t_dom = t_min, Period, Nt
    # create points
    lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
    lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
    lin_t = np.linspace(t_dom[0], t_dom[1], t_dom[2])
    dom = np.zeros((Nt, Nx * Ny, 3))
    m = 0
    # 对dom进行时间维度的赋值
    for t in np.nditer(lin_t):
        dom[m, :, 2] = t # 将dom每一个时间步进行赋值
        c_count = 0
        for x in np.nditer(lin_x):
            tb = y_dom[2] * c_count
            te = tb + y_dom[2]
            c_count += 1
            dom[m, tb:te, 0] = float(x)
            dom[m, tb:te, 1] = lin_y
        m += 1
    dom_plt = dom.reshape(Nx*Ny*Nt, 3) # 为了构建可以画图的梁
    return dom_plt
# ------------------------------------
# Author: minh.nguyen@ikm.uni-hannover.de
# Initial date: 09.09.2019
# an additional functionality : get interior points without taking boundary points
# ------------------------------------
def setup_domain_v2(interData=False):
    Nx_temp, Ny_temp = 2000, 500
    x_dom = x_min, Length, Nx_temp
    y_dom = y_min, Height, Ny_temp
    # create points
    lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
    lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
    dom = np.zeros((Nx_temp * Ny_temp, 2))
    c = 0
    for x in np.nditer(lin_x):
        tb = y_dom[2] * c
        te = tb + y_dom[2]
        c += 1
        dom[tb:te, 0] = x
        dom[tb:te, 1] = lin_y
    print(dom.shape)
    np.meshgrid(lin_x, lin_y)
    fig = plt.figure(figsize=(30, 1))
    ax = fig.add_subplot(111)
    ax.scatter(dom[:, 0], dom[:, 1], s=0.005, facecolor='blue')
    ax.set_xlabel('X', fontsize=3)
    ax.set_ylabel('Y', fontsize=3)
    ax.tick_params(labelsize=4)
    # ------------------------------------ BOUNDARY ----------------------------------------
    # Left boundary condition (Dirichlet BC)
    bcl_u_pts_idx = np.where(dom[:, 0] == x_min)
    bcl_u_pts = dom[bcl_u_pts_idx, :][0]
    bcl_u = np.ones(np.shape(bcl_u_pts)) * [known_left_ux, known_left_uy]

    # Right boundary condition (Neumann BC)
    bcr_t_pts_idx = np.where(dom[:, 0] == Length)
    bcr_t_pts = dom[bcr_t_pts_idx, :][0]
    bcr_t = np.ones(np.shape(bcr_t_pts)) * [known_right_tx, known_right_ty]

    ax.scatter(dom[:, 0], dom[:, 1], s=0.005, facecolor='blue')
    ax.scatter(bcl_u_pts[:, 0], bcl_u_pts[:, 1], s=0.5, facecolor='red')
    ax.scatter(bcr_t_pts[:, 0], bcr_t_pts[:, 1], s=0.5, facecolor='green')
    plt.show()
    if interData == 1:
        x_dom = x_min, Length, Nx
        y_dom = y_min, Height, Ny
        # create points
        lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
        lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
        dom = np.zeros((Nx * Ny, 2))
        c = 0
        for x in np.nditer(lin_x):
            tb = y_dom[2] * c
            te = tb + y_dom[2]
            c += 1
            dom[tb:te, 0] = x
            dom[tb:te, 1] = lin_y
        id1 = np.where((dom[:, 1] > y_min))
        dom = dom[id1, :][0]
        id2 = np.where((dom[:, 1] < Height))
        dom = dom[id2, :][0]
        id3 = np.where((dom[:, 0] > x_min))
        dom = dom[id3, :][0]
        id4 = np.where((dom[:, 0] < Length))
        dom = dom[id4, :][0]
        fig = plt.figure(figsize=(30, 1))
        ax = fig.add_subplot(111)
        ax.scatter(dom[:, 0], dom[:, 1], s=0.005, facecolor='blue')
        plt.show()
    # exit()
    boundary_neumann = {
        # condition on the right
        "neumann_1": {
            "coord": bcr_t_pts,
            "known_value": bcr_t,
            "penalty": bc_right_penalty
        }
        # adding more boundary condition here ...
    }
    boundary_dirichlet = {
        # condition on the left
        "dirichlet_1": {
            "coord": bcl_u_pts,
            "known_value": bcl_u,
            "penalty": bc_left_penalty
        }
        # adding more boundary condition here ...
    }
    return dom, boundary_neumann, boundary_dirichlet

def get_datatest_v2(Nx=num_test_x, Ny=num_test_y, interData=False):
    x_dom_test = x_min, Length, Nx
    y_dom_test = y_min, Height, Ny
    # create points
    x_space = np.linspace(x_dom_test[0], x_dom_test[1], x_dom_test[2])
    y_space = np.linspace(y_dom_test[0], y_dom_test[1], y_dom_test[2])
    xGrid, yGrid = np.meshgrid(x_space, y_space)
    data_test = np.concatenate(
        (np.array([xGrid.flatten()]).T, np.array([yGrid.flatten()]).T), axis=1)
    if interData == 1:
        id1 = np.where((data_test[:, 1] > y_min))
        data_test = data_test[id1, :][0]
        id2 = np.where((data_test[:, 1] < Height))
        data_test = data_test[id2, :][0]
        id3 = np.where((data_test[:, 0] > x_min))
        data_test = data_test[id3, :][0]
        id4 = np.where((data_test[:, 0] < Length))
        data_test = data_test[id4, :][0]
        fig = plt.figure(figsize=(30, 1))
        ax = fig.add_subplot(111)
        ax.scatter(data_test[:, 0], data_test[:, 1], s=0.005, facecolor='blue')
        plt.show()
        return x_space[1:-1], y_space[1:-1], data_test
    fig = plt.figure(figsize=(30, 1))
    ax = fig.add_subplot(111)
    ax.scatter(data_test[:, 0], data_test[:, 1], s=0.005, facecolor='blue')
    plt.show()
    return x_space, y_space, data_test

if __name__ == '__main__':
    setup_domain()
