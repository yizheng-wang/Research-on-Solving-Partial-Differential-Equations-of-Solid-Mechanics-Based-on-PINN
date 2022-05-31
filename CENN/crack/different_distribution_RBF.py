# comparision different RBF distribution way
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import time
from scipy.stats import qmc

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     #random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(0)
# =============================================================================
# use RBF to approximate distance function
# =============================================================================

# learning the boundary solution
# step 1: get the boundary train data and target for boundary condition
# learning the distance neural network by RBF
def RBF(x):
    d_total_t = torch.from_numpy(d_total).unsqueeze(1) # d_total is the internal and boundary points
    w_t = torch.from_numpy(w) # weight of the RBF
    x_l = x.unsqueeze(0).repeat(len(d_total_t), 1, 1) # input 
    R = torch.norm(d_total_t - x_l, dim=2) # compute the distance between the reference and input points
    y = torch.mm(torch.exp(-gama*R.T), w_t) # get the output
    return y


# =============================================================================
# uniform
# =============================================================================

n_d = 11 # the number of the points in each boundary.
n_dom = 11
gama = 0.5

n_test = 201
ep = np.linspace(-1, 1, n_test) # distribute points from -1 to 1 uniformly
# obtain 4 boundary points(for more accurate evaluation)
ep1 = np.zeros((n_test, 2)) # up
ep1[:, 0], ep1[:, 1] = ep, 1
ep2 = np.zeros((n_test, 2)) # down
ep2[:, 0], ep2[:, 1] = ep, -1
ep3 = np.zeros((n_test, 2)) # left
ep3[:, 0], ep3[:, 1] = -1, ep
ep4 = np.zeros((n_test, 2)) # right
ep4[:, 0], ep4[:, 1] = 1, ep
ep5 = np.zeros((n_test, 2)) # center
ep5[:, 0], ep5[:, 1] = ep/2-0.5, 0
points_d_test = np.concatenate((ep1, ep2, ep3, ep4, ep5)) # get the coordinate of the essential boundary points
points_d_test = np.unique(points_d_test, axis=0) # exclude the same points
kdt_t = KDTree(points_d_test, metric='euclidean') # for more accurate evaluation


# obtain 4 boundary points
ep = np.linspace(-1, 1, n_d) # distribute points from -1 to 1 uniformly
ep1 = np.zeros((n_d, 2)) # up
ep1[:, 0], ep1[:, 1] = ep, 1
ep2 = np.zeros((n_d, 2)) # down
ep2[:, 0], ep2[:, 1] = ep, -1
ep3 = np.zeros((n_d, 2)) # left
ep3[:, 0], ep3[:, 1] = -1, ep
ep4 = np.zeros((n_d, 2)) # right
ep4[:, 0], ep4[:, 1] = 1, ep
ep5 = np.zeros((n_d, 2)) # center
ep5[:, 0], ep5[:, 1] = ep, 0
ep5 = ep5[ep5[:,0]<=0]
points_d = np.concatenate((ep1, ep2, ep3, ep4, ep5)) # get the coordinate of the essential boundary points
points_d = np.unique(points_d, axis=0) # exclude the same points
kdt = KDTree(points_d, metric='euclidean') # make the essential boundary points to be a object.

domx = np.linspace(-1, 1, n_dom)[1:-1] # get the domain points
domy = np.linspace(-1, 1, n_dom)[1:-1] 
domx, domy = np.meshgrid(domx, domy)
domxy_uni = np.stack((domx.flatten(), domy.flatten()), 1)
domxy_uni = domxy_uni[(domxy_uni[:, 1]!=0)|(domxy_uni[:, 0]>0)] 
d_dir, _ = kdt.query(points_d, k=1, return_distance = True) # get the distance label of the boundary points (0)
d_dom, _ = kdt.query(domxy_uni, k=1, return_distance = True) # get the distance label of the dom points
# piece the essential and domain points
d_total = np.concatenate((points_d, domxy_uni))

start = time.time()
# get the K matrix used to determine the weight of RBF
dx = d_total[:, 0][:, np.newaxis] - d_total[:, 0][np.newaxis, :]
dy = d_total[:, 1][:, np.newaxis] - d_total[:, 1][np.newaxis, :]
R = np.sqrt(dx**2+dy**2)
K = np.exp(-gama*R)

b = np.concatenate((d_dir, d_dom)) # the label of the distance corresponding to every points
w = np.dot(np.linalg.inv(K),b) # get the weight of the RBF
end = time.time()
consume_time_RBF = end-start

n_test = 201 # obtain the test points for the contourf plot, the number of the test points is n_test**2
domx_t = np.linspace(-1, 1, n_test)
domy_t = np.linspace(-1, 1, n_test)
domx_t, domy_t = np.meshgrid(domx_t, domy_t)
domxy_t = np.stack((domx_t.flatten(), domy_t.flatten()), 1)
domxy_t= torch.from_numpy(domxy_t).requires_grad_(True) # it is possible to AD with respect to the coordinates

dis_RBF = RBF(domxy_t) # get the distance result by RBF

# plot RBF contourf
fig = plt.figure(dpi=1000,figsize = (3,3))
fig.tight_layout()
plt.subplots_adjust(wspace = 0.4, hspace=0)
# plot the distribution points of RBF
plt.scatter(d_total[:, 0], d_total[:, 1])
ax = plt.gca()
ax.set_aspect(1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('RBF distribution way: uniform', size = 14)
plt.show()
# for plot the distance contourf by RBF
dis_plot_RBF = dis_RBF.data.cpu().numpy().reshape(n_test, n_test)
fig = plt.figure(dpi=1000,figsize = (4,3))
h1 = plt.contourf(domx_t, domy_t, dis_plot_RBF,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('RBF contourf of distance: uniform', size = 14)
#plt.savefig('../../图片/裂纹/RBF.pdf', bbox_inches = 'tight')
plt.show()

# get average error of the RBF network
dis_RBF_test = RBF(torch.from_numpy(points_d_test)).numpy()
L2_rbf_uni = np.mean(dis_RBF_test)
print('L2 error of the RBF uniform is '+str(L2_rbf_uni))

# =============================================================================
# halton
# =============================================================================
# get the halton points
halton_sampler = qmc.Halton(d=2, scramble=False) 
halton_points = halton_sampler.random(n_dom**2)
l_bounds = [-1, -1]
u_bounds = [1, 1]
domxy_hal = qmc.scale(halton_points, l_bounds, u_bounds)
plt.scatter(domxy_hal[:, 0], domxy_hal[:,1])
# new total points

d_dom, _ = kdt.query(domxy_hal, k=1, return_distance = True) # get the distance label of the dom points
# piece the essential and domain points
d_total = domxy_hal

start = time.time()
# get the K matrix used to determine the weight of RBF
dx = d_total[:, 0][:, np.newaxis] - d_total[:, 0][np.newaxis, :]
dy = d_total[:, 1][:, np.newaxis] - d_total[:, 1][np.newaxis, :]
R = np.sqrt(dx**2+dy**2)
K = np.exp(-gama*R)

b = d_dom # the label of the distance corresponding to every points
w = np.dot(np.linalg.inv(K),b) # get the weight of the RBF
end = time.time()
consume_time_RBF = end-start

n_test = 201 # obtain the test points for the contourf plot, the number of the test points is n_test**2
domx_t = np.linspace(-1, 1, n_test)
domy_t = np.linspace(-1, 1, n_test)
domx_t, domy_t = np.meshgrid(domx_t, domy_t)
domxy_t = np.stack((domx_t.flatten(), domy_t.flatten()), 1)
domxy_t= torch.from_numpy(domxy_t).requires_grad_(True) # it is possible to AD with respect to the coordinates

dis_RBF = RBF(domxy_t) # get the distance result by RBF

# plot RBF contourf
fig = plt.figure(dpi=1000,figsize = (3,3))
fig.tight_layout()
plt.subplots_adjust(wspace = 0.4, hspace=0)
# plot the distribution points of RBF
plt.scatter(d_total[:, 0], d_total[:, 1])
ax = plt.gca()
ax.set_aspect(1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('RBF distribution way: halton', size = 14)
plt.show()
# for plot the distance contourf by RBF
dis_plot_RBF = dis_RBF.data.cpu().numpy().reshape(n_test, n_test)
fig = plt.figure(dpi=1000,figsize = (4,3))
h1 = plt.contourf(domx_t, domy_t, dis_plot_RBF,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('RBF contourf of distance: halton', size = 14)
#plt.savefig('../../图片/裂纹/RBF.pdf', bbox_inches = 'tight')
plt.show()

# get average error of the RBF network
dis_RBF_test = RBF(torch.from_numpy(points_d_test)).numpy()
L2_rbf_hal = np.mean(dis_RBF_test)
print('L2 error of the RBF halton is '+str(L2_rbf_hal))