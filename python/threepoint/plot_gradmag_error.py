#plane : V: 5184x3 F: 10082x3 Bounds: [-10] -> [10]

import numpy as np
import matplotlib.pyplot as plt
from math import *
import sys

G = []
E = []
G_mag = []
E_mag = []
diff_mag = []

for line in sys.stdin:
    tokens = line.split()
    geo = float(tokens[1])
    euc = float(tokens[3])
    geo_mag = float(tokens[15])
    euc_mag = float(tokens[17])
    G.append(geo)
    E.append(euc)
    G_mag.append(geo_mag)
    E_mag.append(euc_mag)

    d_mag = abs(geo_mag - euc_mag)
    diff_mag.append(d_mag)

G = np.array(G)
E = np.array(E)
diff = np.abs(G - E)

G_mag = np.array(G_mag)
E_mag = np.array(E_mag)
diff_mag = np.array(diff_mag)


plt.plot(E, diff_mag, 'x')
plt.xlabel('Euclidean distance')
plt.ylabel('Absolute error in mag of gradient')
plt.suptitle('Comparing error in magnitude of gradient of geodesic for a plane')
plt.title('(using ground truth geodesic)')
plt.show()
