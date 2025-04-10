
# ------------------------------ network settings ---------------------------------------------------
iteration = 20
D_in = 2
H = 100
D_out = 2
lr = 0.1
# ------------------------------ material parameter -------------------------------------------------
model_energy = 'NeoHookean2D'
E = 200
nu = 1/3
param_c1 = 630
param_c2 = -1.2
param_c = 100
# ----------------------------- define structural parameters ---------------------------------------
Length = 4.0
Height = 1.0
Depth = 1.0

known_left_ux = 0
known_left_uy = 0
bc_left_penalty = 1.0

known_right_ux = 0
known_right_uy = 0
bc_right_penalty = 1.0

known_up_tx = 0
known_up_ty = -1.0
bc_up_penalty = 1.0
# ------------------------------ define domain and collocation points -------------------------------
Nx = 401 # 120  # 120
Ny = 101 # 30  # 60
x_min, y_min = (0.0, 0.0)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
shape = [Nx, Ny]
dxdy = [hx, hy]

# ------------------------------ data testing -------------------------------------------------------
num_test_x = 200
num_test_y = 50