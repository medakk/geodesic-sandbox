import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)

from matplotlib.pyplot import figure

dirs = [
    'benchmarks/sphere_100',
    'benchmarks/sphere_150',
    'benchmarks/sphere_200',
]

for d in dirs:
    mem_values = []
    mem_labels = []
    with open(os.path.join(d, 'memory.txt')) as fd:
        line = fd.readline()
        n_verts = line.split()[-1]
        for line in fd.readlines():
            k, v = line.split(':')
            mem_labels.append(k)
            mem_values.append(int(v[:-2]))
    
    seq = list(range(len(mem_values)))
    figure(num=None, figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.bar(seq, mem_values)
    plt.xticks(seq, mem_labels)
    plt.legend()
    plt.ylabel('Memory(kb)')
    plt.xlabel('Method')
    plt.title(f'Memory usage with {n_verts} vertices')
    plt.grid(axis='y')
    plt.show()

    ts = np.ndarray([1000, 3])
    errors = np.ndarray([1000, 3])
    for col, method in enumerate(['vtk_dijkstra', 'igl_exact', 'igl_heat']):
        benchfile = os.path.join(d, method) + '.txt'
        print(benchfile)
        with open(benchfile) as fd:
            for row, line in enumerate(fd):
                t, err = [float(x) for x in line.split()]
                ts[row, col] = t
                errors[row, col] = err

    figure(num=None, figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.hist(errors, 10, label=['VTK Dijkstra', 'IGL Exact', 'IGL Heat'])
    plt.legend()
    plt.title(f'Absolute Errors with {n_verts} vertices')
    plt.xlabel('L1 Error')
    plt.show()

    avg_ts = np.average(ts, axis=0)
    print(f'Average times: {avg_ts}')
    figure(num=None, figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.bar(seq, avg_ts)
    plt.xticks(seq, mem_labels)
    plt.yscale('log')
    plt.grid(axis='y')
    plt.legend()
    plt.title(f'Average compute time {n_verts} vertices')
    plt.xlabel('microseconds')
    plt.show()

