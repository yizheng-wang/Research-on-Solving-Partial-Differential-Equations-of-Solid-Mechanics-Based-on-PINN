import sys
sys.path.insert(0, '/home/sg/SeaDrive/My Libraries/开题报告/PINN最小能量原理/dem_hyperelasticity-master') # add路径
from dem_hyperelasticity.config import *
from dem_hyperelasticity.plate_hole.config import *


def setup_domain(): # 四分之一2D方板圆孔
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
    # 删除离中心距离小于5的点，利用二范数    
    dom_idx = np.where(np.linalg.norm(dom, axis = 1)>5) # 找到离中心距离大于5的点的位置
    dom = dom[dom_idx, :][0] # 重新覆盖dom
    # 生成离中心距离等于5的点
    eve_angle = np.pi / 2 / Nx
    angle = 0
    boundary_inner = np.zeros((Nx+1, 2))
    for i in range(Nx+1):
        boundary_inner[i] = r * np.cos(angle), r * np.sin(angle)
        angle += eve_angle
    dom = np.concatenate((dom, boundary_inner))
    bound_id = np.ones(np.shape(dom)) # 生成一个储存本质边界条件的ndarray
    print(dom.shape)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(dom[:, 0], dom[:, 1], s=0.005, facecolor='blue')
    ax.set_xlabel('X', fontsize=3)
    ax.set_ylabel('Y', fontsize=3)
    ax.tick_params(labelsize=4)
    # ------------------------------------ BOUNDARY ----------------------------------------
    # Left boundary condition (Dirichlet BC)
    bcl_u_pts_idx_left = np.where(dom[:, 0] == x_min) # 找到x=0的坐标位置
    bcl_u_pts_left = dom[bcl_u_pts_idx_left, :][0] # 找到左边界条件的坐标点
    bcl_u_left = np.ones(np.shape(bcl_u_pts_left)) * [known_left_ux, known_left_uy]
    bound_id[bcl_u_pts_idx_left, 0]=0 # 将左边界条件赋予bound_id中，从1变成0，0代表位移约束条件
    
    # down boundary condition (Dirichlet BC)
    bcl_u_pts_idx_down = np.where(dom[:, 1] == y_min)
    bcl_u_pts_down = dom[bcl_u_pts_idx_down, :][0]
    bcl_u_down = np.ones(np.shape(bcl_u_pts_down)) * [known_down_ux, known_down_uy]
    bound_id[bcl_u_pts_idx_down, 1]=0 # 将下边界条件赋予bound_id中，从1变成0，0代表位移约束条件
    
    # combine the left and down boundary condition
    bcl_u = np.concatenate((bcl_u_left, bcl_u_down))   
    bcl_u_pts = np.concatenate((bcl_u_pts_left, bcl_u_pts_down))  
    # Right boundary condition (Neumann BC)
    bcr_t_pts_idx = np.where(dom[:, 0] == Length) # 返回一个元组
    bcr_t_pts = dom[bcr_t_pts_idx, :][0] # 由于返回一个元组，所以生成的是一个3维的ndarray，第一个维度是由于bcr_t_pts_idx是元组才形成的，因此后面要加[0]
    bcr_t = np.ones(np.shape(bcr_t_pts)) * [known_right_tx, known_right_ty]

    ax.scatter(dom[:, 0], dom[:, 1], s=0.005, facecolor='blue')
    ax.scatter(bcl_u_pts[:, 0], bcl_u_pts[:, 1], s=0.5, facecolor='red')
    ax.scatter(bcr_t_pts[:, 0], bcr_t_pts[:, 1], s=0.5, facecolor='green')
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
            "coord": bcl_u_pts_left,
            "known_value": bcl_u_left,
            "penalty": bc_left_penalty,
            "dir_normal2d": bc_left_normal2d
        },
        # adding more boundary condition here ...
        "dirichlet_2": {
            "coord": bcl_u_pts_down,
            "known_value": bcl_u_down,
            "penalty": bc_down_penalty,
            "dir_normal2d": bc_down_normal2d
        }
    }
    return dom, boundary_neumann, boundary_dirichlet, bound_id # 多输出一个本质边界条件位置的ndarray：bound_id


# -----------------------------------------------------------------------------------------------------
# prepare inputs for testing the model
# -----------------------------------------------------------------------------------------------------
def get_datatest(Nx=num_test_x, Ny=num_test_y):
    '''
    

    Parameters
    ----------
    Nx : int, optional
        the number of points along the x axis. The default is num_test_x.
    Ny : int, optional
        the number of points along the y axis. The default is num_test_y.

    Returns
    -------
    dom : ndarray
        the test points in plane with hole.
        the size of dom is (Nx*Ny,2) because the dimensionality of the problem is 2D.

    '''
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
    # 删除离中心距离小于5的点，利用二范数    
    dom_idx = np.where(np.linalg.norm(dom, axis = 1)>5) # 找到离中心距离大于5的点的位置
    dom = dom[dom_idx, :][0] # 重新覆盖dom
    # 生成离中心距离等于5的点
    eve_angle = np.pi / 2 / Nx
    angle = 0
    boundary_inner = np.zeros((Nx+1, 2))
    for i in range(Nx+1):
        boundary_inner[i] = r * np.cos(angle), r * np.sin(angle)
        angle += eve_angle
    dom = np.concatenate((dom, boundary_inner))    
    
    return dom


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
