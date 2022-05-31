import numpy as np
import matplotlib.pyplot as plt

theta = np.radians(60)
c, s = np.cos(theta), np.sin(theta)
rot60 = np.matrix([[c, -s], [s, c]])

def get_point_list(x1, x5):
    """generate the list of points given two end points"""
    x2 = (1 / 3) * x5 + (2 / 3) * x1
    x4 = (2 / 3) * x5 + (1 / 3) * x1
    x3 = rot60.dot(x4 - x2).A1 + x2
    x6 = rot60.dot(x2 - x1).A1 + x1
    x7 = rot60.dot(x5 - x4).A1 + x4
    return x1, x2, x3, x4, x5, x6, x7

def inside_triangle(p, pt1, pt2, pt3):
    """determine if p is inside of a triangle given three points p0, p1, p2"""
    s, t = np.array(np.dot(np.matrix(np.vstack((pt2 - pt1, pt3 - pt1))).T.I, (p - pt1).T)) # 返回了一个
    return (0 <= s) & (s<= 1) & (0 <= t) & (t<= 1) & (s + t <= 1)


def inside_Koch_curve(x, n, x1=np.array([0, 0]), x5=np.array([1, 0])): # 这里只针对n=2的情形
    x1, x2, x3, x4, x5, x6, x7 = get_point_list(x1, x5)
    if n==1:
        return inside_triangle(x, x2, x3, x4)
    
    in3 = inside_triangle(x, x2, x3, x4)  # 3
    
    in1 = inside_Koch_curve(x, n-1, x1, x2)

    in2 = inside_Koch_curve(x, n-1, x2, x3)

    in4 = inside_Koch_curve(x, n-1, x3, x4)

    in5 = inside_Koch_curve(x, n-1, x4, x5)
    return in1 | in2 | in3 | in4 | in5

def get_koch_points(N_train):
    x = np.random.rand(N_train)*30-15
    y = np.random.rand(N_train)*20*np.sqrt(3)-10*np.sqrt(3)
    x = x/10
    y = y/10
    # 定义大三角形的三个顶点
    pt1 = np.array([-15, 5*np.sqrt(3)])/10
    pt2 = np.array([15, 5*np.sqrt(3)])/10
    pt3 = np.array([0, -10*np.sqrt(3)])/10
    p = np.stack((x.flatten(), y.flatten()), 1)
    
    deter_in_tri = inside_triangle(p, pt1, pt2, pt3) # 首先判断点是不是在内部三角形，然而在判断点是否在每一条科赫雪花的曲线中，用or语句判断
    down_curve1 = inside_Koch_curve(p,  2,  pt1, pt2)
    down_curve2 = inside_Koch_curve(p,  2,  pt2, pt3)
    down_curve3 = inside_Koch_curve(p,  2,  pt3, pt1)
    p = p[down_curve1 | down_curve2 | down_curve3 | deter_in_tri]
    plt.scatter(p[:, 0], p[:, 1], s = 0.2)
    return p

def get_koch_points_lin(N_train):
    x = np.linspace(-15, 15, N_train)
    y = np.linspace(-10*np.sqrt(3), 10*np.sqrt(3), N_train)
    x = x/10
    y = y/10
    x, y = np.meshgrid(x, y)
    # 定义大三角形的三个顶点
    pt1 = np.array([-15, 5*np.sqrt(3)])/10
    pt2 = np.array([15, 5*np.sqrt(3)])/10
    pt3 = np.array([0, -10*np.sqrt(3)])/10
    p = np.stack((x.flatten(), y.flatten()), 1)
    
    deter_in_tri = inside_triangle(p, pt1, pt2, pt3) # 首先判断点是不是在内部三角形，然而在判断点是否在每一条科赫雪花的曲线中，用or语句判断
    down_curve1 = inside_Koch_curve(p,  2,  pt1, pt2)
    down_curve2 = inside_Koch_curve(p,  2,  pt2, pt3)
    down_curve3 = inside_Koch_curve(p,  2,  pt3, pt1)
    p = p[down_curve1 | down_curve2 | down_curve3 | deter_in_tri]
    plt.scatter(p[:, 0], p[:, 1], s = 0.2)
    return p

def whether_koch(points):
    pt1 = np.array([-15, 5*np.sqrt(3)])/10
    pt2 = np.array([15, 5*np.sqrt(3)])/10
    pt3 = np.array([0, -10*np.sqrt(3)])/10
    p = points
    
    deter_in_tri = inside_triangle(p, pt1, pt2, pt3) # 首先判断点是不是在内部三角形，然而在判断点是否在每一条科赫雪花的曲线中，用or语句判断
    down_curve1 = inside_Koch_curve(p,  2,  pt1, pt2)
    down_curve2 = inside_Koch_curve(p,  2,  pt2, pt3)
    down_curve3 = inside_Koch_curve(p,  2,  pt3, pt1)    
    return down_curve1 | down_curve2 | down_curve3 | deter_in_tri