import dmsh
import numpy as np
import matplotlib.pyplot as plt 
r = 0.25
Length =1 
Height = 1
Nx = 100
Ny = 100
def setup_domain(): # 四分之一2D方板圆孔

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('X', fontsize=3)
    ax.set_ylabel('Y', fontsize=3)
    ax.tick_params(labelsize=4)
    
    rect = dmsh.Rectangle(0.0, +1.0, 0.0, 1.0)
    c = dmsh.Circle([0.5, 0.5], 0.25)
    geo = dmsh.Difference(rect, c)
    X, cells = dmsh.generate(geo, lambda pts: np.abs(c.dist(pts))/20+0.020, tol=1.0e-15) # 划分693个点
    dom = cells_to_mid(cells, X) # 得到内部点与相应的面积
    
    x_dom = r, Length, Nx
    y_dom = r, Height, Ny
    # create points
    lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
    lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2]) 

    return dom



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
    dom = setup_domain()
plt.scatter(dom[:, 0], dom[:, 1])
plt.show()
np.save('1hole_dmsh', dom)