import os
import sys
from itertools import chain, product
import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial import Delaunay

import dmsh, meshio


def writeAddInfo(listAmntSubds, listSplitCoefs, coefsConvergence):
    with open("input_files/tests_info.dat", "w") as f:
        f.write(" ".join([str(x) for x in listAmntSubds]) + "\n")
        f.write(" ".join([str(x) for x in listSplitCoefs]) + "\n")
        f.write(" ".join([str(x) for x in coefsConvergence]) + "\n")

def write_mesh(edge_size):
    nameFile = f"mesh_{edge_size:.0e}.dat"

    bounds = np.array([[0.01, 0], [0.02, 0], [0.02, 0.01], [0.01, 0.01]])

    geo = dmsh.Polygon(bounds)
    X, cells = dmsh.generate(geo, edge_size)

    #meshio.write_points_cells("circle.dat", X, {"triangle": cells})
    bounds = np.array([[0.01, 0], [0.02, 0], [0.02, 0.01], [0.01, 0.01], [0.01, 0]])

    with open("scr/operations_input_files/input_files/mesh/" + nameFile,"w") as f:
        f.write("{:g}\n".format(len(bounds)))
        for point in bounds:
            f.write("{:g} {:g}\n".format(point[0], point[1]))
        f.write("{:d}\n".format(len(X)))
        for point in X:
            f.write("{:g} {:g}\n".format(point[0], point[1]))
        f.write("{:d}\n".format(len(cells)))
        for element in cells:
            f.write("{:g} {:g} {:g}\n".format(element[0], element[1], element[2]))

def write_task(task):
    '''
    dirichlet_conditions:
    1) сторона
    2) количество ограничений:
    0 - по x
    1 - по y
    2 - по обеим координатам
    3) ограничение (одно или два)
    '''

    dimTask = 2
    E = 70e+9
    nyu = 0.34
    coef_u = 1000
    coef_sigma = 1e-3
    coef_overlap = 0.3

    nameFile = f"task_{task}.dat"

    if task == 1:
        dirichlet_conditions = [[0, 1, 0], [1, 0, 0], [3, 0, 0]]
    else:
        dirichlet_conditions = [[0, 1, 0], [3, 0, 0]]

    neumann_conditions = [[2, 0, -2e+7]]

    with open("modules/creation_input_files/input_files/" + nameFile,"w") as f:
        f.write(str(dimTask)+'\n')
        f.write("{:g} {:g}\n".format(E, nyu))
        f.write("{:g}\n".format(len(dirichlet_conditions)))
        for cond in dirichlet_conditions:
            for val in cond:
                f.write(f"{val:g} ")
            f.write(f"\n")
        f.write("{:g}\n".format(len(neumann_conditions)))
        for cond in neumann_conditions:
            f.write("{:g} {:g} {:g}\n".format(cond[0], cond[1], cond[2]))
        f.write("{:g} {:g}\n".format(coef_u, coef_sigma))
        f.write("{:g}\n".format(coef_overlap))

if __name__ == "__main__":
    list_edge_size = [1e-3]
    list_tasks = [1, 2]

    #for task in list_tasks:
    #    write_task(task)

    for cur_edge_size in list_edge_size:
        write_mesh(cur_edge_size)

    
    #writeAddInfo(list_subds, list_mesh, list_coefs_convergence)
