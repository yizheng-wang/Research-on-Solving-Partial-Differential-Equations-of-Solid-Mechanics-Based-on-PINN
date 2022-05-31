from dolfin import *
import fenics as fe
import time
import meshio
import numpy as np
from pyevtk.hl import gridToVTK
from pyevtk.hl import pointsToVTK

tol = 1E-14
class Obstacle0(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= 0.5-tol
class Obstacle1(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= 0.5+tol

def write_vtk_vts(filename, x_space, y_space, z_space, U,  SVonMises):
    #已经将输出的感兴趣场进行了分类VTK导出,用VTs格式方便数据可视化
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={ "displacement": U, \
                                               "S-VonMises": SVonMises
                                               })

start = time.time()
# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
# parameters["form_compiler"]["representation"] = "quadrature"  # change quadrature to uflacs if there's problem
parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
# mesh = UnitCubeMesh(40, 20, 20)
p0 = Point(0.0, 0.0)
p1 = Point(4.0, 1.0)
nx = 40
ny = 10 # 只要不过中线project的效果就不错
nz = 2

mesh = RectangleMesh(p0, p1, nx, ny)
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# 构造下面的超弹性材料1，上面的超材料区域为0
obstacle0 = Obstacle0()
obstacle1 = Obstacle1()
domains =  MeshFunction("size_t", mesh, 2)
obstacle0.mark(domains, 0)
obstacle1.mark(domains, 1)

dx = Measure("dx", domain=mesh, subdomain_data=domains) # 上下区域进行分界


# Sub domain for clamp at left end
# def left(x, on_boundary):
#     return x[0] < 10e-14 and on_boundary

# Sub domain for rotation at right end
# def right(x, on_boundary):
#     return x[0] > 1 - 10e-14 and on_boundary
# Mark boundary subdomians
# https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html
left = CompiledSubDomain("near(x[0], side) && on_boundary", side=0.0, tol=10e-15)  # Code C++
# def boundary_L(x, on_boundary):
#     tol = 1E-14
#     return on_boundary and near(x[0], 0, tol)
# right = CompiledSubDomain("near(x[0], side) && on_boundary", side=20.0)
# Sub domain for clamp at left end
# def left(x, on_boundary):
#     return x[0] < DOLFIN_EPS and on_boundary

# Sub domain for rotation at right end
# def right(x, on_boundary):
#     return (x[0] - 20.0) < DOLFIN_EPS and on_boundary
# Define Dirichlet boundary (x = 0 or x = 1)
c = Expression(("0.0", "0.0"), degree=2)
t = Expression(("0.0", "-5.0"), degree=2) # 将力从-5修改为-100

# bcl = DirichletBC(V, c, left)
# bcr = DirichletBC(V, t, right)
# bcs = [bcl, bcr]
# bcs = [bcl]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v = TestFunction(V)             # Test function
u = Function(V)                 # Displacement from previous iteration
# B  = Constant((0.0, -0.5, 0.0))  # Body force per unit volume
# T = Constant((1.0,  0.0, 0.0))  # Traction force on the boundary
# traction = Constant((0.0, -1.25, 0.0))
# n = FacetNormal(mesh)

# Create mesh function over the cell facets
# boundary_subdomains = MeshFunction("size_t", mesh, 3)
# boundary_subdomains.set_all(0)
# AutoSubDomain(left).mark(boundary_subdomains, 1)
# AutoSubDomain(right).mark(boundary_subdomains, 2)
# dss = ds(subdomain_data=boundary_subdomains)

# neumann_domain = MeshFunction("size_t", mesh, 3)
D = mesh.topology().dim()
print(D)
neumann_domain = MeshFunction("size_t", mesh, D-1) # 这里不知道为什么要减1，我猜测可能是从0开始计数
neumann_domain.set_all(0)
CompiledSubDomain("near(x[0], side) && on_boundary", side=4.0, tol=10e-15).mark(neumann_domain, 1)
plot(neumann_domain, interactive=True)
ds = Measure("ds", subdomain_data=neumann_domain)


bcs = DirichletBC(V, c, left)
# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C) # 定义三个主不变量
I2 = 0.5*(Ic**2-tr(C*C))
J = det(F)
c = 100
c1 = 630
c2 = -1.2
d = 2*(c1 + 2 * c2)
# Elasticity parameters # 上面的材料0




E0, nu0 = 1000, 0.3
mu0, lmbda0 = Constant(E0/(2*(1 + nu0))), Constant(E0*nu0/((1 + nu0)*(1 - 2*nu0)))

E1, nu1 = 10000, 0.3
mu1, lmbda1 = Constant(E1/(2*(1 + nu1))), Constant(E1*nu1/((1 + nu1)*(1 - 2*nu1)))
# Stored strain energy density (compressible neo-Hookean model)
psi0 = (mu0/2)*(Ic - 3) - mu0*ln(J) + (lmbda0/2)*(ln(J))**2 # Neo
psi1 = (mu1/2)*(Ic - 3) - mu1*ln(J) + (lmbda1/2)*(ln(J))**2 # Neo
# psi1 = c*(J-1)**2 - d*ln(J) + c1*(Ic-2) + c2*(I2-1) # 下层Mooney

# Total potential energy
Pi = psi0*dx(0) + psi1*dx(1) - dot(t, u)*ds(1)

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)
problem = NonlinearVariationalProblem(F, u, bcs, J)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters
# prm['newton_solver']['absolute_tolerance'] = 1E-8
# prm['newton_solver']['relative_tolerance'] = 1E-7
# prm['newton_solver']['maximum_iterations'] = 25
# prm['newton_solver']['relaxation_parameter'] = 1.0
# prm['newton_solver']['linear_solver'] = 'gmres'
prm['newton_solver']['linear_solver'] = 'petsc'
# prm['newton_solver']['linear_solver'] = 'minres'
# prm['newton_solver']['linear_solver'] = 'umfpack'
# list_linear_solver_methods()
solver.solve()
# Solve variational problem
# solve(F == 0, u, bcs, J=J, form_compiler_parameters=ffc_options)
# solve(F == 0, u, bcs, J=J, solver_parameters={"linear_solver":"lu"})
# Save solution in VTK format
# file = File("./output/fem/beam3d_4x1_fem_quad.pvd");
# file << u;

L2 = inner(u, u) * dx
H10 = inner(grad(u), grad(u)) * dx
energynorm = sqrt(assemble(psi0*dx(0) + psi1*dx(1)))
# H1 = inner(F, P) * dx
L2norm = sqrt(assemble(L2))
H10norm = sqrt(assemble(H10))
# print("L2 norm = %.10f" % L2norm)
# print("H1 norm = %.10f" % sqrt(L2norm**2 + H10norm**2))
# print("H10 norm (H1 seminorm) = %.10f" % H10norm)
print("L2 norm = %.10f" % norm(u, norm_type="L2"))
print("H1 norm = %.10f" % norm(u, norm_type="H1"))
print("H10 norm = %.10f" % norm(u, norm_type="H10"))
print("Running time = %.3f" % float(time.time()-start))

# Plot solution
#plot(u, title='Displacement', mode='displacement')
F = I + grad(u)
P =  Expression('x[1] >= 0.5 - tol ? k_0 : k_1', degree=0, tol=tol, k_0=mu0, k_1=mu1)*F + (Expression('x[1] >= 0.5 - tol ? k_0 : k_1', degree=0, tol=tol, k_0=lmbda0, k_1=lmbda1)*ln(det(F)) - Expression('x[1] >= 0.5 - tol ? k_0 : k_1', degree=0, tol=tol, k_0=mu0, k_1=mu1)) * inv(F).T
# secondPiola = Expression('x[1] >= 0.5 - tol ? k_0 : k_1', degree=0, tol=tol, k_0=1, k_1=0) * inv(F) * P + Expression('x[1] >= 0.5 - tol ? k_0 : k_1', degree=0, tol=tol, k_0=0, k_1=1) * ((2*c1+2*c2*Ic)*I - 2*c2*C.T + (2*c*(det(F)-1)*det(F) -d)*inv(C).T) 
secondPiola =  inv(F) * P
Sdev = secondPiola - (1./3)*tr(secondPiola)*I # deviatoric stress
von_Mises = sqrt(3./2*inner(Sdev, Sdev))
V = FunctionSpace(mesh, "Lagrange", 1)
W = TensorFunctionSpace(mesh, "Lagrange", 1)
von_Mises = project(von_Mises, V)
# von_Mises = fe.interpolate(von_Mises, V)
Stress = project(secondPiola, W, solver_type='petsc')
Stresspk1 = project(P, W, solver_type='petsc')
#plot(von_Mises, title='Stress intensity')

# Compute magnitude of displacement
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)
#plot(u_magnitude, 'Displacement magnitude')
# print('min/max u:',
#       u_magnitude.vector().array().min(),
#       u_magnitude.vector().array().max())

# # Save solution to file in VTK format
File('./output/fem2d/elasticity/displacement.pvd') << u
File('./output/fem2d/elasticity/von_mises.pvd') << von_Mises
File('./output/fem2d/elasticity/magnitude.pvd') << u_magnitude
File('./output/fem2d/elasticity/Stresspk2.pvd') << Stress
File('./output/fem2d/elasticity/Stresspk1.pvd') << Stresspk1
# %%
femdis = meshio.read('./output/fem2d/elasticity/displacement000000.vtu') # 读入有限元的位移解
femvon = meshio.read('./output/fem2d/elasticity/von_mises000000.vtu') # 读入有限元的位移解
femcoor = femdis.points # 得到坐标
print('von_Mise min:', np.min(von_Mises.compute_vertex_values()))
print('uy min:', np.min(u.compute_vertex_values()))

femdis = sorted(femdis.point_data.values())[0] # 得到有限元相应的y方向的位移解，这是一个一维度的ARRAY，用来评估error

femvon = sorted(femvon.point_data.values())[0] # 得到有限元的vonmise应力，是一个一维度的array

femvonx2 = femvon[(femcoor[:, 0]==2)]
print('von_Mise min x=2:', np.min(femvonx2))
print('von_Mise max x=2:', np.max(femvonx2))

x = np.linspace(0, 4, nx+1)
y = np.linspace(0, 1, ny+1)
z = np.array([0])
# 整理一下顺序，需要将应力与应变画在同一张图上
femdisz = np.zeros((1, len(y), len(x)))
write_vtk_vts('./output/fem2d/elasticity/fem', x, y, z, (femdis[:,0].reshape(1, ny+1, nx+1).transpose(1,2,0).copy(), femdis[:,1].reshape(1, ny+1, nx+1).transpose(1,2,0).copy(), femdisz) ,femvon.reshape(1, ny+1, nx+1).transpose(1,2,0).copy())
