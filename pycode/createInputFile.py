from itertools import product, chain
from scipy.spatial import Delaunay
import numpy as np
import os
import sys

def writeAddInfo(listAmntSubds, listParamDivider):
    with open("tests/tests_info.dat", "w") as f:
        f.write(" ".join([str(x) for x in listAmntSubds]) + "\n")
        f.write(" ".join([str(x) for x in listParamDivider]) + "\n")

'''
dirichlet_conditions:
1) сторона
2) количество ограничений:
0 - по x
1 - по y
2 - по обеим координатам
3) ограничение (одно или два)
'''

typeTask = 2
paramDivider = 5
dimTask = 2
amntSubdomains = [8, 1]
coefOverlap = 0.3
E = 70e+9
nyu = 0.34
coef_u = 1000
coef_sigma = 1e-3

nameFile = f"test_{typeTask}_{paramDivider}_{amntSubdomains[0]}.dat"

bounds = np.array([[0.01, 0], [0.02, 0], [0.02, 0.01], [0.01, 0.01], [0.01, 0]])
x_list, y_list = np.linspace(bounds[0, 0], bounds[1, 0], paramDivider), np.linspace(bounds[0, 1], bounds[2, 1], paramDivider)
points = np.array(list(product(x_list,y_list)))
tri = Delaunay(points)

if typeTask == 1:
    dirichlet_conditions = [[0, 1, 0], [1, 0, 0], [3, 0, 0]]
elif typeTask == 2:
    dirichlet_conditions = [[0, 1, 0], [3, 0, 0]]

neumann_conditions = [[2, 0, -2e+7]]

#writeAddInfo([2, 4, 8], [5, 10, 20])

with open("tests/" + nameFile,"w") as f:
    f.write(str(dimTask)+'\n')
    f.write("{:g}\n".format(len(bounds)))
    for point in bounds:
        f.write("{:g} {:g}\n".format(point[0], point[1]))
    f.write(f"{paramDivider:d}\n")
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
