from dem_hyperelasticity.importlib import *
import scipy.integrate as sp

# convert numpy BCs to torch
def ConvBCsToTensors(bc_d):
    size_in_1 = len(bc_d)
    size_in_2 = len(bc_d[0][0])
    bc_in = torch.empty(size_in_1, size_in_2, device=dev)
    c = 0
    for bc in bc_d:
        bc_in[c, :] = torch.from_numpy(bc[0])
        c += 1
    return bc_in


# --------------------------------------------------------------------------------
# purpose: doing something in post processing for visualization in 3D
# --------------------------------------------------------------------------------
def write_vtk(filename, x_space, y_space, z_space, Ux, Uy, Uz):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    displacement = (Ux, Uy, Uz)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": displacement})

# --------------------------------------------------------------------------------
# purpose: doing something in post processing for visualization in 3D
# --------------------------------------------------------------------------------
def write_vtk_v2(filename, x_space, y_space, z_space, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    disX = U[0]
    disY = U[1]
    disZ = U[2]
    def check_shapes(*arrays):
        shape = arrays[0].shape
        for index, array in enumerate(arrays):
            assert array.shape == shape, f"Array shapes are inconsistent: {shape} and {array.shape}, index:{index}"
    
    check_shapes(xx, yy, zz, disX, disY, disZ, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises)
 

    gridToVTK(filename, xx, yy, zz, pointData={"dis_x": disX, "dis_y": disY, "dis_z": disZ,\
                                               "S-VonMises": SVonMises, \
                                               "S11": S11, "S12": S12, "S13": S13, \
                                               "S22": S22, "S23": S23, "S33": S33, \
                                               "E11": E11, "E12": E12, "E13": E13, \
                                               "E22": E22, "E23": E23, "E33": E33})
    # gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})

def write_vtk_v2p(filename, dom, disy): #已经将输出的感兴趣场进行了分类VTK导出
    xx = np.ascontiguousarray(dom[:, 0])
    yy = np.ascontiguousarray(dom[:, 1])
    zz = np.ascontiguousarray(dom[:, 2]) # 点的VTK
    pointsToVTK(filename, xx, yy, zz, data={ "dis_Y_error": disy})
# --------------------------------------------------------------------------------
# purpose: doing something in post processing for visualization in 3D
# --------------------------------------------------------------------------------
def write_arr2DVTK(filename, coordinates, values):
    # displacement = np.concatenate((values[:, 0:1], values[:, 1:2], values[:, 0:1]), axis=1)
    x = np.array(coordinates[:, 0].flatten(), dtype='float32')
    y = np.array(coordinates[:, 1].flatten(), dtype='float32')
    z = np.zeros(x.shape, dtype='float32')
    disX = np.array(values[:, 0].flatten(), dtype='float32')
    disY = np.array(values[:, 1].flatten(), dtype='float32')
    disZ = np.zeros(disX.shape, dtype='float32')
    displacement = (disX, disY, disZ)
    gridToVTK(filename, x, y, z, pointData={"displacement": displacement})

# --------------------------------------------------------------------------------
# purpose: doing something in post processing for visualization in 3D
# --------------------------------------------------------------------------------
def write_vtk_2d(filename, x_space, y_space, Ux, Uy):
    xx, yy = np.meshgrid(x_space, y_space)
    displacement = (Ux, Uy, Ux)
    gridToVTK(filename, xx, yy, xx,  pointData={"displacement": displacement})


# --------------------------------------------------------------------------------
# purpose: plotting loss convergence
# --------------------------------------------------------------------------------
def plot_loss_convergence(loss_array):
    print('Loss convergence')
    range = np.arange(1, len(loss_array) + 1)
    loss_plt, = plt.semilogx(range, loss_array, label='total loss')
    plt.legend(handles=[loss_plt])
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.show()


def plot_deformed_displacement(surfaceUx, surfaceUy, defShapeX, defShapeY):
    fig, axes = plt.subplots(nrows=2)
    cs1 = axes[0].contourf(defShapeX, defShapeY, surfaceUx, 255, cmap=cm.jet)
    cs2 = axes[1].contourf(defShapeX, defShapeY, surfaceUy, 255, cmap=cm.jet)
    fig.colorbar(cs1, ax=axes[0])
    fig.colorbar(cs2, ax=axes[1])
    axes[0].set_title("Displacement in x")
    axes[1].set_title("Displacement in y")
    fig.tight_layout()
    for tax in axes:
        tax.set_xlabel('$x$')
        tax.set_ylabel('$y$')
    plt.show()


def getL2norm2D(surUx, surUy, Nx, Ny, hx, hy):
    uX1D = surUx.flatten()
    uY1D = surUy.flatten()
    uXY = np.concatenate((np.array([uX1D]).T, np.array([uY1D]).T), axis=-1)
    N = Nx * Ny
    udotu = np.zeros(N)
    for i in range(N):
        udotu[i] = np.dot(uXY[i, :], uXY[i, :].T)
    udotuTensor = udotu.reshape(Nx, Ny)
    # ||u||_L^2 = \sqrt(\int (u.u))
    L2norm = np.sqrt(np.trapz(np.trapz(udotuTensor, dx=hy), dx=hx))
    # L2norm = np.sqrt(sp.simps(sp.simps(udotuTensor, dx=hy), dx=hx))
    return L2norm


def getL2norm(surUx, surUy, surUz, Nx, Ny, Nz, hx, hy, hz, dim=3):
    if dim == 2:
        uX1D = surUx.flatten()
        uY1D = surUy.flatten()
        uXY= np.concatenate((np.array([uX1D]).T, np.array([uY1D]).T), axis=-1)
        N = Nx * Ny
        udotu = np.zeros(N)
        for i in range(N):
            udotu[i] = np.dot(uXY[i, :], uXY[i, :].T)
        udotuTensor = udotu.reshape(Nx, Ny)
        # ||u||_L^2 = \sqrt(\int (u.u))
        L2norm = np.sqrt(np.trapz(np.trapz(udotuTensor, dx=hy), dx=hx))
        # L2norm = np.sqrt(sp.simps(sp.simps(udotuTensor, dx=hy), dx=hx))
    else:
        uX1D = surUx.flatten()
        uY1D = surUy.flatten()
        uZ1D = surUz.flatten()
        uXYZ = np.concatenate((np.array([uX1D]).T, np.array([uY1D]).T, np.array([uZ1D]).T), axis=-1)
        N = Nx * Ny * Nz
        udotu = np.zeros(N)
        for i in range(N):
            udotu[i] = np.dot(uXYZ[i, :], uXYZ[i, :].T)
        udotuTensor = udotu.reshape(Nx, Ny, Nz)
        # ||u||_L^2 = \sqrt(\int (u.u))
        L2norm = np.sqrt(np.trapz(np.trapz(np.trapz(udotuTensor, dx=hz), dx=hy), dx=hx))
        # L2norm = np.sqrt(sp.simps(sp.simps(sp.simps(udotuTensor, dx=hz), dx=hy), dx=hx))
    return L2norm

def getH10norm(F11, F12, F13, F21, F22, F23, F31, F32, F33, Nx, Ny, Nz, hx, hy, hz, dim=3):
    if dim == 2:
        FinnerF = (F11-1)**2 + F12**2 + F21**2 + (F22-1)**2
        FinnerFTensor = FinnerF.reshape(Nx, Ny)
        H10norm = np.sqrt(np.trapz(np.trapz(FinnerFTensor, dx=hy), dx=hx))
        # H10norm = np.sqrt(sp.simps(sp.simps(FinnerFTensor, dx=hy), dx=hx))
    else:
        # ||u||_H^1_0 = \sqrt(\int (Gradu : Gradu)) = Aij Bij
        # FinnerF = (F11-1)*(F11-1) + F12*F21 + F13*F31 + F21*F12 + (F22-1)*(F22-1) + F23*F32 + F31*F13 + F32*F23 + (F33-1)*(F33-1)  # WRONG
        FinnerF = (F11 - 1) * (F11 - 1) + F12 * F12 + F13 * F13 + F21 * F21 + (F22 - 1) * (
                    F22 - 1) + F23 * F23 + F31 * F31 + F32 * F32 + (F33 - 1) * (F33 - 1)
        FinnerFTensor = FinnerF.reshape(Nx, Ny, Nz)
        H10norm = np.sqrt(np.trapz(np.trapz(np.trapz(FinnerFTensor, dx=hz), dx=hy), dx=hx))
        # H10norm = np.sqrt(sp.simps(sp.simps(sp.simps(FinnerFTensor, dx=hz), dx=hy), dx=hx))
    return H10norm

def getH10norm2D(F11, F12, F21, F22, Nx, Ny, hx, hy):
    FinnerF = (F11 - 1) ** 2 + F12 ** 2 + F21 ** 2 + (F22 - 1) ** 2
    FinnerFTensor = FinnerF.reshape(Nx, Ny)
    H10norm = np.sqrt(np.trapz(np.trapz(FinnerFTensor, dx=hy), dx=hx))
    # H10norm = np.sqrt(sp.simps(sp.simps(FinnerFTensor, dx=hy), dx=hx))
    return H10norm


