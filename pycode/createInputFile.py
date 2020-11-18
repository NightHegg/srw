from itertools import product, chain
from scipy.spatial import Delaunay
import numpy as np
import os
import sys

'''
dirichlet_conditions:
1) сторона
2) количество ограничений:
0 - по x
1 - по y
2 - по обеим координатам
3) ограничение (одно или два)
'''

nameFile = "test_1.dat"

paramDivider = 25
dimTask = 2
amntSubdomains = [4, 1]
coefOverlap = 0.3
E = 70e+9
nyu = 0.34
coef_u = 1000
coef_sigma = 1e-3

bounds = np.array([[0.01, 0], [0.02, 0], [0.02, 0.01], [0.01, 0.01], [0.01, 0]])
x_list, y_list = np.linspace(bounds[0, 0], bounds[1, 0], paramDivider), np.linspace(bounds[0, 1], bounds[2, 1], paramDivider)
points = np.array(list(product(x_list,y_list)))
tri = Delaunay(points)

dirichlet_conditions = [[0, 1, 0], [1, 0, 0], [3, 0, 0]]
neumann_conditions = [[2, 0, -2e+7]]

with open("tests/" + nameFile,"w") as f:
    f.write(str(dimTask)+'\n')
    f.write("{:g}\n".format(len(bounds)))
    for point in bounds:
        f.write("{:g} {:g}\n".format(point[0], point[1]))
    f.write("{:d}\n".format(paramDivider))
    f.write("{:g}\n".format(coefOverlap))
    f.write("{:d} {:d}\n".format(amntSubdomains[0], amntSubdomains[1]))
    f.write("{:g} {:g}\n".format(E, nyu))
    f.write("{:g} {:g}\n".format(coef_u, coef_sigma))
    f.write("{:g}\n".format(len(dirichlet_conditions)))
    for cond in dirichlet_conditions:
        for val in cond:
            f.write(f"{val:g} ")
        f.write(f"\n")
        #f.write("{:g} {:g} {:g}\n".format(cond[0], cond[1], cond[2]))
    f.write("{:g}\n".format(len(neumann_conditions)))
    for cond in neumann_conditions:
        f.write("{:g} {:g} {:g}\n".format(cond[0], cond[1], cond[2]))
    f.write("{:d}\n".format(len(points)))
    for point in points:
        f.write("{:g} {:g}\n".format(point[0], point[1]))
    f.write("{:d}\n".format(len(tri.simplices)))
    for element in tri.simplices:
        f.write("{:g} {:g} {:g}\n".format(element[0], element[1], element[2]))