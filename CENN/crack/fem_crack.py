from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as  plt 

mesh = RectangleMesh(Point(-1.0, 0.0), Point(1.0, 1.0), 10, 10)
V = FunctionSpace(mesh, 'Lagrange', 2)
u_d = Expression('sqrt((sqrt(x[0]*x[0] + x[1]*x[1]) - x[0])/2)', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_d, boundary)

u = TrialFunction(V)
v = TestFunction(V)
f = Constant('0')
a = dot(grad(u), grad(v))*dx
L = f*v*dx

u1 = Function(V)
solve(a==L, u1, bc)
plot(u1)
#plt.colorbar()
plt.show()
plot(mesh)
plt.show()


error_L21 = errornorm(u_d, u1, 'H1')

vertex_ex = u_d.compute_vertex_values(mesh)
vertex_p = u1.compute_vertex_values(mesh)

error_max1 = np.max(np.abs(vertex_ex-vertex_p))
 

# 下半部分
mesh = RectangleMesh(Point(-1.0, -1.0), Point(1.0, 0.0), 10, 10)
V = FunctionSpace(mesh, 'Lagrange', 2)
u_d = Expression('-sqrt((sqrt(x[0]*x[0] + x[1]*x[1]) - x[0])/2)', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_d, boundary)

u = TrialFunction(V)
v = TestFunction(V)
f = Constant('0')
a = dot(grad(u), grad(v))*dx
L = f*v*dx

u2 = Function(V)
solve(a==L, u2, bc)
plot(u2)
#plt.colorbar()
plt.show()
plot(mesh)
plt.show()


vtkfile = File('picture/fem_solution.pvd')
vtkfile << u1,u2

error_L22 = errornorm(u_d, u2, 'H1')

vertex_ex = u_d.compute_vertex_values(mesh)
vertex_p = u2.compute_vertex_values(mesh)

error_max2 = np.max(np.abs(vertex_ex-vertex_p))
# 将上半部分和下半部分的解空间合并起来
Nt = 100 # 取100个测试点
x = np.linspace(-1,1,Nt)
y = np.linspace(-1,1,Nt)
x, y = np.meshgrid(x, y)

u = np.zeros((Nt**2, 1))
xy = np.stack((x.flatten(), y.flatten()), 1)
for index,i in enumerate(xy):
    if i[1]>=0:
        u[index] = u1(i)
    if i[1]<=0:
        u[index] = u2(i)
u = u.reshape((x.shape))
plt.contourf(x, y, u, levels=100)
plt.colorbar()
plt.show()





