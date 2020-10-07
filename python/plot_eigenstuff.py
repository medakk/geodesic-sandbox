import sys

import numpy as np
import matplotlib.pyplt as plt

fname = 


plt.matshow(D)
plt.colorbar()
plt.title('Vertex-Vertex distances for sphere(9802 vertices)')
plt.xlabel('Vertex i')
plt.ylabel('Vertex j')
plt.show()





v = scipy.linalg.eigvals(D)
first_k = 5000
plt.plot(np.abs(np.real((v[:first_k]))))
plt.xlabel('k')
plt.ylabel('k-th absolute eigenvalue REAL component')
plt.title(f'Eigen spectrum for sphere distance matrix(first {first_k} k)')
plt.yscale('log')
plt.grid(axis='y')
plt.show()
