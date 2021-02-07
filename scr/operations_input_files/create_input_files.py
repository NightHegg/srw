import os
import sys
from itertools import chain, product

import numpy as np
from scipy.spatial import Delaunay


def writeAddInfo(listAmntSubds, listSplitCoefs, coefsConvergence):
    with open("input_files/tests_info.dat", "w") as f:
        f.write(" ".join([str(x) for x in listAmntSubds]) + "\n")
        f.write(" ".join([str(x) for x in listSplitCoefs]) + "\n")
        f.write(" ".join([str(x) for x in coefsConvergence]) + "\n")

def write_mesh(splitCoef):
    bounds = np.array([[0.01, 0], [0.02, 0], [0.02, 0.01], [0.01, 0.01], [0.01, 0]])
    x_list, y_list = np.linspace(bounds[0, 0], bounds[1, 0], splitCoef), np.linspace(bounds[0, 1], bounds[2, 1], splitCoef)
    points = np.array(list(product(x_list,y_list)))
    tri = Delaunay(points)

    nameFile = f"mesh_{splitCoef}.dat"

    with open("modules/creation_input_files/input_files/mesh/" + nameFile,"w") as f:
        f.write("{:g}\n".format(len(bounds)))
        for point in bounds:
            f.write("{:g} {:g}\n".format(point[0], point[1]))
        f.write("{:d}\n".format(len(points)))
        for point in points:
            f.write("{:g} {:g}\n".format(point[0], point[1]))
        f.write("{:d}\n".format(len(tri.simplices)))
        for element in tri.simplices:
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
    list_subds = [2, 4, 8]
    list_mesh = [6, 7, 8, 9, 11, 12, 13, 14]
    list_tasks = [1, 2]
    list_coefs_convergence = [1e-3, 1e-4, 1e-5]

    #for task in list_tasks:
    #    write_task(task)

    #for cur_mesh in list_mesh:
    #    write_mesh(cur_mesh)
    
    #writeAddInfo(list_subds, list_mesh, list_coefs_convergence)
