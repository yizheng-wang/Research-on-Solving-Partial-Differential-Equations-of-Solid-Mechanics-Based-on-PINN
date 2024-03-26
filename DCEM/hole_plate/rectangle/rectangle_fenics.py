from fenics import *
from dolfin import *
import matplotlib.pyplot as plt
import time
start = time.time()
# Create mesh and define function space
L = 1.0  # Length of the domain
W = 1.0  # Width of the domain
nx, ny = 100, 100  # Number of elements in x and y directions
mesh = RectangleMesh(Point(0, 0), Point(L, W), nx, ny)
V = VectorFunctionSpace(mesh, 'Lagrange', 3)

# Define boundary conditions
def left_boundary(x, on_boundary):
    return near(x[0], 0.) and on_boundary

bc_left = DirichletBC(V, Constant((0, 0)), left_boundary)

# Define material parameters
E = Constant(1000)  # Young's modulus
nu = Constant(0.3)  # Poisson's ratio
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Define strain and stress
def epsilon(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return 2 * mu * epsilon(u)  +  lmbda * Identity(2) * (1-nu/(1-nu)) * tr(epsilon(u))

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant((0., 0.))  # Body force

# Define the force on the right boundary (Neumann boundary condition)
g = Expression(("100*sin(x[1]*pi)", "0."), degree = 2)


right = AutoSubDomain(lambda x: near(x[0], 1.0))
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
right.mark(boundaries, 1)
ds = ds(subdomain_data=boundaries)

a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx + inner(g, v) * ds(1)  # Apply force on the right boundary

# Compute solution
u = Function(V)
solve(a == L, u, bc_left)

stress = sigma(u)
I = Identity(2)  
Sdev = stress - (1./3)*tr(stress)*I
von_Mises = sqrt(3./2*inner(Sdev, Sdev))
u_magnitude = sqrt(dot(u, u))

V_f = FunctionSpace(mesh, "Lagrange", 3)
W = TensorFunctionSpace(mesh, "Lagrange", 3)
u_magnitude = project(u_magnitude, V_f)
von_Mises = project(von_Mises, V_f)
stress = project(stress, W)
end = time.time()
# Save solution to file or plot the displacement
# ...
File("./output/fem/rectangle/rectangle2d_1x1_fem_u.pvd") << u;
File("./output/fem/rectangle/rectangle2d_1x1_fem_mises.pvd") << von_Mises;
File("./output/fem/rectangle/rectangle2d_1x1_fem_stress.pvd") << stress
File("./output/fem/rectangle/rectangle2d_1x1_fem_u_mag.pvd") << u_magnitude
print('The max u is ' +  str(u_magnitude.vector().get_local().max()))
# Plot solution
fig = plot(u, title='Displacement', mode='displacement')
plt.colorbar(fig)
plt.show()
# interactive()
print("The time of FEM with %i is %f" % (nx*ny, end-start))
