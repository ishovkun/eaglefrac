import numpy as np
import matplotlib.pyplot as plt

inch = 2.54*1e-2
sin = np.sin
cos = np.cos
sqrt = np.sqrt

# data = np.loadtxt("./post.txt")
data = np.loadtxt("../solid-three_point_bending/boundary_load-1.txt")
t = data[:, 0]
load_x = data[:, 1]
load_y = -data[:, 2]

# compute fracture toughness
r = 0.75*inch  # core radius
a = r/5.
# thick = 16*inch
thick = 1.
E = 1.0e3
nu = 0.3
E_star = E/(1.-nu**2)

P_max = max(load_y)
# P_max = max(sqrt(load_y**2 + load_x*2))

# main exp
Y_I = 5.6 - 22.2*(a/r) + 166.9*(a/r)**2 - 576.2*(a/r)**3 + 928.8*(a/r)**4 - 505.9*(a/r)**5
K_IC = P_max*sqrt(np.pi*a)/(2.*r*thick)*Y_I
# print(Y_I)

# third paper
# Y_I = 4.782 + 1.219*(a/r) + 0.063*np.exp(7.045*(a/r))
# K_IC = P_max*sqrt(np.pi*a)/(2.*r*thick)*Y_I

# original paper
# Y_I = 4.782 - 1.219*(a/r) + 0.063*np.exp(7.045*(a/r))
# B = 6.55676 + 16.64035*(a/r)**2.5 + 27.97042*(a/r)**6.5 + 215.0839*(a/r)**16
# K_IC = P_max*sqrt(np.pi*a)/(2.*r*thick)*(Y_I + B*(a/r))

# theta = 0.  # plane || to the fracture tip
# K_I = K_IC
# K_II = 0.  # only open mode I
#
# k_I = 0.5*cos(theta/2)*(K_I*(1+cos(theta)) - 3*K_II*sin(theta))
# # this one is just 0
# k_II = 0.5*cos(theta/2)*(K_I*sin(theta) + K_II*(3*cos(theta) - 1))
# G = (k_I**2 + k_II**2)/E_star

G = K_IC**2/E_star

print("Pmax", P_max)
print("K_IC", K_IC)
print("G_c", G)

# values summary
'''
# Arc load
phi     G_c
5       0.27
10      0.22
15      0.29
20      0.31
30      0.23

# Point load
phi     G_c
5       0.49
10      0.72
20      0.65
30      0.63
'''

plt.plot(t, load_y)
# plt.xlim(0, 1e-2)
# plt.ylim(0, 1e3)
plt.show()
