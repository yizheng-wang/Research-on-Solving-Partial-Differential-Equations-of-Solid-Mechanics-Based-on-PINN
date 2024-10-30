"""
@author: 王一铮, 447650327@qq.com
"""

import numpy as np
import matplotlib.pyplot as plt


num_bound = 2
class direct_vec:

    def __init__(self, theta):
        self.vx, self.vy = np.cos(theta), np.sin(theta)
        self.theta = theta

    def rot(self, theta):
        return direct_vec(theta + self.theta)

    def slope(self):
        return self.vy / self.vx

    def get_cos(self):
        return self.vx

    def get_sin(self):
        return self.vy

class point:

    def __init__(self, x0, y0):
        self.x = x0
        self.y = y0

    def get_point(self):
        return (self.x, self.y)

    def set_point(self, x=None, y=None):
        if x!=None:
            self.x = x
        if y!=None:
            self.y = y

    def copy(self):
        return point(self.x, self.y)

def plot_linear(point, vec, length):
    x0, y0= point.get_point()
    x_range = np.linspace(start=x0, stop=x0 + vec.vx * length, num=num_bound)
    plt.plot(x_range, vec.slope() * (x_range - x0) + y0)
    return np.stack((x_range, vec.slope() * (x_range - x0) + y0), 1)

def plot_random(point, vec, length):
    x0, y0= point.get_point()
    x_range = x0 + vec.vx*length*np.random.rand(num_bound)
    plt.plot(x_range, vec.slope() * (x_range - x0) + y0)
    return np.stack((x_range, vec.slope() * (x_range - x0) + y0), 1)

def koch(p0, vec, L, n):
    global points_vec
    p = p0.copy() # 第一个vec是60度
    if n < 0:
        print("error")
        return None
    if n == 0:
        # 直線のみ
        points = plot_linear(p, vec, 3 * L) # 画一个小段
        points_vec = np.concatenate((points_vec, points))
        return
    # 再帰数nが正の数の場合
    # 裾野部分
    koch(p, vec, L / 3, n-1) #第一个用这个函数的时候是n=2，n-1=1，第二次是n=1， n-1=0
    p.set_point(p0.x + vec.get_cos() * 2 * L, p0.y + vec.get_sin() * 2 * L)
    koch(p, vec, L / 3, n-1)
    # 山の左を作る
    p.set_point(p0.x + vec.get_cos() * L, p0.y + vec.get_sin() * L)
    koch(p, vec.rot(np.pi / 3), L / 3, n - 1)
    # 山の右を作る
    p.set_point(p0.x + vec.get_cos() * L + vec.rot(np.pi / 3).vx * L, p0.y + vec.get_sin() * L + vec.rot(np.pi / 3).vy * L)
    koch(p, vec.rot(- np.pi / 3), L / 3, n-1)

def koch_rand(p0, vec, L, n):
    global points_vec
    p = p0.copy() # 第一个vec是60度
    if n < 0:
        print("error")
        return None
    if n == 0:
        # 直線のみ
        points = plot_random(p, vec, 3 * L) # 画一个小段
        points_vec = np.concatenate((points_vec, points))
        return
    # 再帰数nが正の数の場合
    # 裾野部分
    koch_rand(p, vec, L / 3, n-1) #第一个用这个函数的时候是n=2，n-1=1，第二次是n=1， n-1=0
    p.set_point(p0.x + vec.get_cos() * 2 * L, p0.y + vec.get_sin() * 2 * L)
    koch_rand(p, vec, L / 3, n-1)
    # 山の左を作る
    p.set_point(p0.x + vec.get_cos() * L, p0.y + vec.get_sin() * L)
    koch_rand(p, vec.rot(np.pi / 3), L / 3, n - 1)
    # 山の右を作る
    p.set_point(p0.x + vec.get_cos() * L + vec.rot(np.pi / 3).vx * L, p0.y + vec.get_sin() * L + vec.rot(np.pi / 3).vy * L)
    koch_rand(p, vec.rot(- np.pi / 3), L / 3, n-1)

vec = direct_vec(0)
L = 1 # 中间的正三角形变成是30
N = 2
#p0 = point(0, 0) 
p0 = point(-3*L/2, -np.sqrt(3)*L/2)
#fig = plt.figure(figsize=(10,10))

def koch_outer(N, p0, vec, L):
    plt.cla() # 清除当前坐标轴

    p = p0.copy()
    rot_vec = vec.rot(np.pi / 3)
    koch(p, rot_vec, L, N) # 第一个用这个函数的时候，是L=1，N=2，角度是60度，p是原点

    p.set_point(p0.x + rot_vec.get_cos() * 3 * L, p0.y + rot_vec.get_sin() * 3 * L)
    rot_vec = vec.rot(-np.pi / 3)
    koch(p, rot_vec, L, N)

    p.set_point(p0.x + vec.get_cos() * 3 * L, p0.y + vec.get_sin() * 3 * L)
    rot_vec = vec.rot(-np.pi / 3)
    koch(p, vec.rot(-np.pi), L, N)

    plt.title("Koch Curve n=" + str(N))
    
def koch_outer_rand(N, p0, vec, L):
    plt.cla() # 清除当前坐标轴

    p = p0.copy()
    rot_vec = vec.rot(np.pi / 3)
    koch_rand(p, rot_vec, L, N) # 第一个用这个函数的时候，是L=1，N=2，角度是60度，p是原点

    p.set_point(p0.x + rot_vec.get_cos() * 3 * L, p0.y + rot_vec.get_sin() * 3 * L)
    rot_vec = vec.rot(-np.pi / 3)
    koch_rand(p, rot_vec, L, N)

    p.set_point(p0.x + vec.get_cos() * 3 * L, p0.y + vec.get_sin() * 3 * L)
    rot_vec = vec.rot(-np.pi / 3)
    koch_rand(p, vec.rot(-np.pi), L, N)

    plt.title("Koch Curve n=" + str(N))
    
points_vec = np.zeros((1, 2))
def point_bound(num):
    global num_bound, points_vec
    num_bound = num
    points_vec = np.zeros((1, 2))
    koch_outer(N, p0, vec, L)
    points_vec = points_vec[1:]# 去除第一个0，0点
    points_vec = np.unique(points_vec, axis=0)
    return points_vec

def point_bound_rand(num):
    global num_bound, points_vec
    num_bound = num
    points_vec = np.zeros((1, 2))
    koch_outer_rand(N, p0, vec, L)
    points_vec = points_vec[1:]# 去除第一个0，0点
    points_vec = np.unique(points_vec, axis=0)
    return points_vec
# plt.savefig("Koch_curve.png")
#ani.save("koch.gif", writer='imagemagick')
#plt.show()