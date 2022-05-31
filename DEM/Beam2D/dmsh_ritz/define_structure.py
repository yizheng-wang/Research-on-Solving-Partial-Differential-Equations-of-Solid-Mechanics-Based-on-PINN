import sys
sys.path.insert(0, '/home/sg/SeaDrive/My Libraries/开题报告/超材料的PINN/dem_hyperelasticity-master') # add路径
from dem_hyperelasticity.config import *
from dem_hyperelasticity.Beam2D.config import *
import dmsh

def setup_domain(s=0.5): # 2D悬臂梁撒点
    geo = dmsh.Rectangle(0.0, 4.0, 0.0, 1.0)
    path = dmsh.Path([[0.0, 0.0], [0.0, 1.0]])
    X, cells = dmsh.generate(geo, s, tol = 1e-10)
    dom_dmsh = cells_to_mid(cells, X) # 得到内部点与相应的面积
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
    print(dom.shape)
    np.meshgrid(lin_x, lin_y)
    fig = plt.figure(figsize=(5, 1))
    ax = fig.add_subplot(111)

    ax.set_xlabel('X', fontsize=3)
    ax.set_ylabel('Y', fontsize=3)
    ax.tick_params(labelsize=4)
    # ------------------------------------ BOUNDARY ----------------------------------------
    # Left boundary condition (Dirichlet BC)
    bcl_u_pts_idx_left = np.where(dom[:, 0] == x_min) # 找到相应边界的点的位置
    bcl_u_pts_left = dom[bcl_u_pts_idx_left, :][0] # 找到相应边界点的坐标
    bcl_u_left = np.ones(np.shape(bcl_u_pts_left)) * [known_left_ux, known_left_uy] # 找到相应边界点的值

    # Right boundary condition (Neumann BC)
    bcr_t_pts_idx_right = np.where(dom[:, 0] == Length)
    bcr_t_pts_right = dom[bcr_t_pts_idx_right, :][0] 
    bcr_t_right = np.ones(np.shape(bcr_t_pts_right)) * [known_right_tx, known_right_ty]
    
    # Down boundary condition (Neumann BC)
    bcr_t_pts_idx_down = np.where(dom[:, 1] == y_min)
    bcr_t_pts_down = dom[bcr_t_pts_idx_down, :][0]
    bcr_t_down = np.ones(np.shape(bcr_t_pts_down)) * [known_down_tx, known_down_ty]    
    
    # Up boundary condition (Neumann BC)
    bcr_t_pts_idx_up = np.where(dom[:, 1] == Height)
    bcr_t_pts_up = dom[bcr_t_pts_idx_up, :][0]
    bcr_t_up = np.ones(np.shape(bcr_t_pts_up)) * [known_up_tx, known_up_ty]    

    ax.scatter(dom_dmsh[:, 0], dom_dmsh[:, 1], s=0.005, facecolor='blue')
    ax.scatter(bcl_u_pts_left[:, 0], bcl_u_pts_left[:, 1], s=0.5, facecolor='red')
    ax.scatter(bcr_t_pts_down[:, 0], bcr_t_pts_down[:, 1], s=0.5, facecolor='green')
    ax.scatter(bcr_t_pts_right[:, 0], bcr_t_pts_right[:, 1], s=0.5, facecolor='green')
    ax.scatter(bcr_t_pts_up[:, 0], bcr_t_pts_up[:, 1], s=0.5, facecolor='green')
    plt.show()
    # exit()
    boundary_neumann = {
        # condition on the right * *******************************************************************************************
        "neumann_down": {
            "coord": bcr_t_pts_down,
            "known_value": bcr_t_down,
            "penalty": bc_down_penalty,
            "boundary_normal2d": bc_down_normal2d
        },
        "neumann_right": {
            "coord": bcr_t_pts_right,
            "known_value": bcr_t_right,
            "penalty": bc_right_penalty,
            "boundary_normal2d": bc_right_normal2d
        },
        "neumann_up": {
            "coord": bcr_t_pts_up,
            "known_value": bcr_t_up,
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
    return dom_dmsh, boundary_neumann, boundary_dirichlet


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
def cells_to_mid(cells, X):
    dom = np.zeros((len(cells), 3))
    # 输入单元的坐标位置与节点坐标，然后输出单元的中点坐标以及第三列是面积
    for idx, i in enumerate(cells):
        points_e = X[i] # 获得每个单元的三点坐标，是一个2*3的array
        mid_p = np.mean(points_e, 0, keepdims = True) # 获得中点坐标
        area_e = calc_area(points_e) # 获得单元的面积
        c_a = np.concatenate((mid_p, np.array([[area_e]])), 1) # 获得中点坐标以及面积
        dom[idx] = c_a
    return dom


def calc_area(P):  
    (x1, y1), (x2, y2), (x3, y3) = P[0], P[1], P[2]
    return 0.5 * abs(x2 * y3 + x1 * y2 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3)
if __name__ == '__main__':
    setup_domain()
