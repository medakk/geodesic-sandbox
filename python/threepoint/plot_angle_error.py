#plane : V: 5184x3 F: 10082x3 Bounds: [-10] -> [10]

import numpy as np
import matplotlib.pyplot as plt
from math import *
import sys

G = []
E = []
G_ang = []
E_ang = []
diff_ang = []

for line in sys.stdin:
    tokens = line.split()
    geo = float(tokens[1])
    euc = float(tokens[3])
    geo_angle = float(tokens[8])
    euc_angle = float(tokens[10])
    G.append(geo)
    E.append(euc)
    G_ang.append(geo_angle)
    E_ang.append(euc_angle)

    d_ang = abs(atan2(sin(geo_angle - euc_angle), cos(geo_angle - euc_angle)))
    diff_ang.append(d_ang)

G = np.array(G)
E = np.array(E)
diff = np.abs(G - E)

G_ang = np.array(G_ang)
E_ang = np.array(E_ang)
diff_ang = np.array(diff_ang)


plt.plot(E, diff_ang, 'x')
plt.xlabel('Euclidean distance')
plt.ylabel('Absolute error in angle of gradient (radians)')
plt.ylim([0.0,pi])
plt.suptitle('Comparing error in gradient of geodesic for a plane')
plt.title('(using ground truth geodesic)')
plt.show()
