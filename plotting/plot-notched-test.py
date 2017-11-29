import numpy as np
import matplotlib.pyplot as plt

# data = np.loadtxt("./post.txt")
data = np.loadtxt("../solid-notched_test/boundary_load-3.txt")
time = data[:, 0]
load_x = data[:, 1]
load_y = data[:, 2]

plt.plot(time, load_y)
plt.xlim(0, 1e-2)
plt.ylim(0, 1e3)
plt.show()
