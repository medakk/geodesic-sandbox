import sys
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)

def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')

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
    rects = plt.bar(seq, avg_ts)
    autolabel(rects)
    plt.xticks(seq, mem_labels)
    plt.yscale('log')
    plt.grid(axis='y')
    plt.legend()
    plt.title(f'Average compute time {n_verts} vertices')
    plt.ylabel('microseconds')
    plt.show()
