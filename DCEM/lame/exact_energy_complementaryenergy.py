import numpy as np

Po = 10
Pi = 5
a = 0.5
b = 1.0

nepoch = 10000

dom_num = 10000
N_test = 10000
E = 1000
nu = 0.3
G = E/2/(1+nu)

Ura = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*a + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/a)
Urb = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*b + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/b)

r = np.linspace(a, b, N_test)
# theta = np.linspace(0, 2*np.pi, N_test) # y方向N_test个点
# r_mesh, theta_mesh = np.meshgrid(r, theta)
sigma_rr = a**2/(b**2-a**2)*(1-b**2/r**2)*Pi - b**2/(b**2-a**2)*(1-a**2/r**2)*Po
sigma_theta = a**2/(b**2-a**2)*(1+b**2/r**2)*Pi - b**2/(b**2-a**2)*(1+a**2/r**2)*Po  
epsilon_rr = 1/E*(sigma_rr - nu * sigma_theta)
epsilon_theta = 1/E*(sigma_theta - nu * sigma_rr)

complementary_d = 1/2*(sigma_rr*epsilon_rr+sigma_theta*epsilon_theta)*2*np.pi*r

complementary_d_energy = np.mean(complementary_d)*(b-a)

sigma_ra_dis = 2*G/(1+nu)*(Ura/(b-a)*(nu*(b-a)/a-1))