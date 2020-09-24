import sys

import numpy as np
import matplotlib.pyplot as plt

benchfile = sys.argv[1]
errors = []
with open(benchfile) as fd:
    for line in fd:
        errors.append([float(x) for x in line.split()])

errors = np.array(errors)

plt.hist(errors, 10, label=['VTK Dijkstra', 'IGL Exact', 'IGL Heat'])
plt.legend()
plt.title('Absolute Errors with Analytic Solution')
plt.show()
