import math
import os
import sys

from ray.worker import init
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import time
import numpy as np
from scipy.sparse import coo_matrix
import math

def create_barycentric_coords(element, points):
    M = np.hstack((np.ones((points[element].shape[0], 1)), points[element])).T
    A = 0.5*np.linalg.det(M)
    M_inv = np.linalg.inv(M)*2*A

    a, b, c = M_inv[:, 0], M_inv[:, 1], M_inv[:, 2]
    return np.vstack((a, b, c)), A


def calculate_local_matrix_stiffness(element, points):
    [a, b, c], A = create_barycentric_coords(element, points)
    B = np.hstack(list(map(lambda x, y, z: [[y, 0], [0, z], [z/2, y/2]], a, b, c))) / 2 / A
    return B, A


def calculate_sparse_matrix_stiffness(area_elements, area_points_coords, amnt_area_points, D, dim_task, modifier = {}):
    row, col, data = [], [], []
    for element in area_elements:
        M = np.hstack((np.ones((area_points_coords[element].shape[0], 1)), area_points_coords[element])).T
        A = 0.5*np.linalg.det(M)
        M_inv = np.linalg.inv(M)
        a, b, c = M_inv[:, 0], M_inv[:, 1], M_inv[:, 2]
        test = np.array([
            [a[0], 0, a[1], 0, a[2], 0],
            [b[0], 0, b[1], 0, b[2], 0],
            [c[0], 0, c[1], 0, c[2], 0],
            [0, a[0], 0, a[1], 0, a[2]],
            [0, b[0], 0, b[1], 0, b[2]],
            [0, c[0], 0, c[1], 0, c[2]]
        ])
        B = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0.5, 0, 0.5, 0]
        ]) @ test
        
        K_element = B.T @ D @ B * A
        for i in range(3):
            for j in range(3):
                point_1 = modifier[element[i]] if modifier else element[i]
                point_2 = modifier[element[j]] if modifier else element[j]
                for k in range(dim_task):
                    for z in range(dim_task):
                        row.append(point_1 * dim_task + k)
                        col.append(point_2 * dim_task + z)
                        data.append(K_element[i * dim_task + k, j * dim_task + z])
    # print(time_1, time_2, time_3, time_4, time_5, time_6)
    return coo_matrix((data, (row, col)), shape = (amnt_area_points * dim_task, amnt_area_points * dim_task)).tolil()        


def func(a, b, c):
    if (c[0] == a[0] and c[1] == a[1]) or (c[0] == b[0] and c[1] == b[1]):
        on_and_between = True
    elif (a[0] == b[0] == c[0]) or (a[0] == b[0]) or (c[0] == a[0]) or (b[0] == c[0]):
        on_and_between = a[1] <= c[1] <= b[1]
    else:
        slope = (b[1] - a[1]) / (b[0] - a[0])
        pt3_on = (c[1] - a[1]) == slope * (c[0] - a[0])
        pt3_between = (min(a[0], b[0]) <= c[0] <= max(a[0], b[0])) and (min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))
        on_and_between = pt3_on and pt3_between
    return on_and_between

if __name__ == "__main__":
    pass
