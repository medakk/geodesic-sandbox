#plane : V: 5184x3 F: 10082x3 Bounds: [-10] -> [10]

import numpy as np
import matplotlib.pyplot as plt
import sys

G = []
E = []

for line in sys.stdin:
    tokens = line.split()
    geo = float(tokens[1])
    euc = float(tokens[3])
    G.append(geo)
    E.append(euc)

G = np.array(G)
E = np.array(E)
diff = np.abs(G - E)

plt.plot(E, diff, 'x')
plt.show()
