import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# inch = 2.54*1e-2
# sin = np.sin
# cos = np.cos
# sqrt = np.sqrt

data = np.loadtxt("../pressurized-sneddon/cod-1.txt")
x = data[:, 0]
cod = data[:, 1]

data_fe = pd.read_csv("../pressurized-sneddon/line_plots/width_line_plot.csv")
w = data_fe['width'].values
arc_l = data_fe['arc_length'].values

# analytical solution (Sneddon)
E = 1
nu = 0.2
p = 1e-3
l0 = 0.4/2
E_prime = E/(1.0-nu**2)

cod_an = 2*p*l0/E_prime*(np.clip(1.0 - (x - 2.0)**2/l0**2, 0, 1))**0.5
cod_an *= 2

fig = plt.figure(figsize=(10, 8))
plt.plot(x, cod*1e3, "ko", label="explicit integraion")
plt.plot(x, cod_an*1e3, label="analytical")
plt.plot(arc_l, w*1e3, "ro", label="solver")
plt.xlabel("x (m)", fontsize=20)
plt.ylabel("COD (mm)", fontsize=20)
plt.legend(frameon=False, fontsize=20)
# plt.ylim(-1e-4, None)
plt.xlim(1.5, 2.5)

# plt.subplots_adjust(left=0.2, right=0.95, top=0.98, bottom=0.15, hspace=0.25)
plt.tick_params(labelsize=20)
plt.show()
