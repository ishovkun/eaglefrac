import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# inch = 2.54*1e-2
# sin = np.sin
# cos = np.cos
# sqrt = np.sqrt

#  data = np.loadtxt("../pressurized-sneddon-old/cod-1.txt")
#  x = data[:, 0]
#  cod = data[:, 1]
#  E = 1
#  nu = 0.2
#  p = 1e-3
#  xf = 0.4/2
#  c = 2.0 # center


#  data = np.loadtxt("../pressurized-sneddon/cod-1.txt")
data = np.loadtxt("../pressurized-sneddon_4+5/cod-1.txt")
x = data[:, 0]
cod = data[:, 1]
E = 1e9
nu = 0.2
p = 1e5
xf = 10./2
c = 20.0 # center

#  data_fe = pd.read_csv("../pressurized-sneddon/width_line_plot.csv")
#  w = data_fe['width'].values
#  arc_l = data_fe['arc_length'].values

# analytical solution (Sneddon)


E_prime = E/(1.0-nu**2)
x_an = np.linspace(c-xf, c+xf, 150, endpoint=True)
# Valko page 34
#  cod_an = 4.*p*xf/E_prime*(np.clip(1.0 - (x - c)**2/xf**2, 0., 1.))**0.5
#  cod_an = 4.*p/E_prime*(xf**2 - (x_an - c)**2)**0.5
cod_an = 4.*p*xf/E_prime*(np.clip(1.0 - (x_an - c)**2/xf**2, 0., 1.))**0.5

fig = plt.figure(figsize=(10, 8))
plt.plot(x, cod*1e3, "k", label="explicit integraion")
plt.plot(x_an, cod_an*1e3, label="analytical")
#  plt.plot(arc_l, w*1e3, "ro", label="solver")
plt.xlabel("x (m)", fontsize=20)
plt.ylabel("Width (mm)", fontsize=20)
plt.legend(loc=8, frameon=False, fontsize=20)
plt.ylim(-1e-4, None)
plt.xlim(c-xf, c+xf)

# plt.subplots_adjust(left=0.2, right=0.95, top=0.98, bottom=0.15, hspace=0.25)
plt.tick_params(labelsize=20)
plt.show()
