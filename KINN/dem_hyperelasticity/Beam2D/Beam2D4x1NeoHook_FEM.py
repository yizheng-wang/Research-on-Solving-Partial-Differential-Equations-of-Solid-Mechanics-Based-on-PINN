from dolfin import *
import fenics as fe
import time

start = time.time()
# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
# parameters["form_compiler"]["representation"] = "quadrature"  # change quadrature to uflacs if there's problem
# parameters["form_compiler"]["quadrature_degree"] = 1
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
# mesh = UnitCubeMesh(40, 20, 20)
# p0 = Point(0.0, 0.0, 0.0)
# p1 = Point(20.0, 5.0, 1.0)
# mesh = BoxMesh(p0, p1, 100, 25, 5)
mesh = RectangleMesh(Point(0.0, 0.0), Point(4.0, 1.0), 200, 50, "crossed")
# mesh = UnitSquareMesh.create(10, 10, CellType.Type.quadrilateral)
V = VectorFunctionSpace(mesh, "Lagrange", 2)
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
t = Expression(("0.0", "-5.0"), degree=2)

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
neumann_domain = MeshFunction("size_t", mesh, D-1)
neumann_domain.set_all(0)
CompiledSubDomain("near(x[0], side) && on_boundary", side=4.0, tol=10e-15).mark(neumann_domain, 1)
ds = Measure("ds", subdomain_data=neumann_domain)


bcs = DirichletBC(V, c, left)
# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J = det(F)

# Elasticity parameters
E, nu = 10**3, 0.3
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 2) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# Total potential energy
Pi = psi*dx - dot(t, u)*ds(1)

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
# prm['newton_solver']['linear_solver'] = 'umfpack'
# list_linear_solver_methods()
solver.solve()
# Solve variational problem
# solve(F == 0, u, bcs, J=J, form_compiler_parameters=ffc_options)
# solve(F == 0, u, bcs, J=J, solver_parameters={"linear_solver":"lu"})
# Save solution in VTK format
file = File("./output/fem/beam2d_4x1_fem_v1.pvd");
file << u;

L2 = inner(u, u) * dx
H10 = inner(grad(u), grad(u)) * dx
energynorm = sqrt(assemble(psi*dx))
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
plot(u, title='Displacement', mode='displacement')
F = I + grad(u)
P = mu * F + (lmbda * ln(det(F)) - mu) * inv(F).T
secondPiola = inv(F) * P
Sdev = secondPiola - (1./3)*tr(secondPiola)*I # deviatoric stress
von_Mises = sqrt(3./2*inner(Sdev, Sdev))
V = FunctionSpace(mesh, "Lagrange", 2)
W = TensorFunctionSpace(mesh, "Lagrange", 2)
von_Mises = project(von_Mises, V)
Stress = project(secondPiola, W)
plot(von_Mises, title='Stress intensity')

# Compute magnitude of displacement
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)
plot(u_magnitude, 'Displacement magnitude')
# print('min/max u:',
#       u_magnitude.vector().array().min(),
#       u_magnitude.vector().array().max())

# Save solution to file in VTK format
File('./output/fem/elasticity/displacement.pvd') << u
File('./output/fem/elasticity/von_mises.pvd') << von_Mises
File('./output/fem/elasticity/magnitude.pvd') << u_magnitude
File('./output/fem/elasticity/Stress.pvd') << Stress
# interactive()