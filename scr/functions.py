import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import time
from itertools import groupby
import numpy as np
import scipy
from scipy.sparse import coo_matrix, lil_matrix
import math


def calculate_local_matrix_stiffness(element, points, dimTask):
    '''
    element - элемент, в котором идёт расчёт \n
    points - массив координат \n
    dimTask - размерность \n
    '''
    B = np.zeros((3, 6))
    x_points = np.array([points[element[i], 0] for i in range(3)])
    y_points = np.array([points[element[i], 1] for i in range(3)])

    a = np.array([x_points[1] * y_points[2] - x_points[2] * y_points[1], 
                  x_points[2] * y_points[0] - x_points[0] * y_points[2],
                  x_points[0] * y_points[1] - x_points[1] * y_points[0]] )
    b = np.array([y_points[1]-y_points[2], y_points[2]-y_points[0], y_points[0]-y_points[1]])
    c = np.array([x_points[2]-x_points[1], x_points[0]-x_points[2], x_points[1]-x_points[0]])

    A = 0.5*np.linalg.det(np.vstack((np.ones(3), x_points, y_points)).T)
    for i in range(len(element)):
        B[:, i*dimTask:i*dimTask + 2] += np.array([[b[i], 0], [0, c[i]], [c[i]/2, b[i]/2]])/2/A
    return B, A


def calculate_sparse_matrix_stiffness(area_elements, area_points_coords, D, dimTask):
    row, col, data = [], [], []
    for element in area_elements:
                B, A = calculate_local_matrix_stiffness(element, area_points_coords, dimTask)
                K_element = B.T @ D @ B * A
                for i in range(3):
                    for j in range(3):
                        for k in range(dimTask):
                            for z in range(dimTask):
                                row.append(element[i] * dimTask + k)
                                col.append(element[j] * dimTask + z)
                                data.append(K_element[i * dimTask + k, j * dimTask + z])

    return coo_matrix((data, (row, col)), shape = (area_points_coords.size, area_points_coords.size)).tolil()        


def calculate_subd_parameters(area_bounds, area_points_coords, area_elements, coef_overlap, cur_amnt_subds):
    time_list = []

    init_time = time.time()
    temp_cond = lambda val, i: val[i] in [area_bounds[0, i], area_bounds[i + 1, i]]
    area_boundary_points = [idx for idx, val in enumerate(area_points_coords) if temp_cond(val, 0) or temp_cond(val, 1)]
    area_points = [num for num, _ in enumerate(area_points_coords)]
    time_list.append(time.time() - init_time)

    subd_bounds = []
    overlap_bounds = []
    list_subd_elements = []
    list_subd_points = []
    dict_elements_contain_point = {}
    subd_boundary_overlap_points = []
    subd_boundary_points = []
    subd_internal_points = []
    list_subd_points_coords = []
    list_full_subd_elements = []

    init_time = time.time()
    for idx, val in enumerate(cur_amnt_subds):
        temp_bounds = [area_bounds[idx, idx] + i*(area_bounds[idx + 1, idx] - area_bounds[idx, idx])/val for i in range(1, val)]
        subd_bounds.append([area_bounds[idx, idx]] + temp_bounds + [area_bounds[idx + 1, idx]])
    time_list.append(time.time() - init_time)

    init_time = time.time()
    subd_limit = lambda x, func, type: func[x] + type * (func[x] - func[x - 1]) * coef_overlap
    for subd in subd_bounds:
        temp_bounds = sum([[subd_limit(x, subd, -1), subd_limit(x, subd, 1)] for x in range(1, len(subd)-1)], [])
        overlap_bounds.append([subd[0]] + temp_bounds + [subd[-1]])
    time_list.append(time.time() - init_time)


    condition_overlap = lambda type, element, coef_1, coef_2: overlap_bounds[type][coef_1] < sum([area_points_coords[element[i], type] for i in range(3)])/3 < overlap_bounds[type][coef_2]

    init_time = time.time()
    if len(overlap_bounds[0]) == 2:
        list_subd_elements.append([idx for idx, element in enumerate(area_elements) if condition_overlap(1, element, 0, 2)])     
        for i in range(1, len(overlap_bounds[1]) - 3, 2):
            list_subd_elements.append([idx for idx, element in enumerate(area_elements) if condition_overlap(1, element, i, i + 3)])
        list_subd_elements.append([idx for idx, element in enumerate(area_elements) if condition_overlap(1, element, -3, -1)])
    elif len(overlap_bounds[1]) == 2:
        list_subd_elements.append([idx for idx, element in enumerate(area_elements) if condition_overlap(0, element, 0, 2)])
        for i in range(1, len(overlap_bounds[0]) - 3, 2):
            list_subd_elements.append([idx for idx, element in enumerate(area_elements) if condition_overlap(0, element, i, i + 3)])
        list_subd_elements.append([idx for idx, element in enumerate(area_elements) if condition_overlap(0, element, -3, -1)])
    else:
        
        list_subd_elements.append([idx for idx, element in enumerate(area_elements) if condition_overlap(0, element, 0, 2) and condition_overlap(1, element, 0, 2)])
        for i in range(1, len(overlap_bounds[1]) - 3, 2):
            list_subd_elements.append([idx for idx, element in enumerate(area_elements) if condition_overlap(0, element, 0, 2) and condition_overlap(1, element, i, i + 3)])
        list_subd_elements.append([idx for idx, element in enumerate(area_elements) if condition_overlap(0, element, 0, 2) and condition_overlap(1, element, -3, -1)])

        for i in range(1, len(overlap_bounds[0]) - 3, 2):
            list_subd_elements.append([idx for idx, element in enumerate(area_elements) if condition_overlap(0, element, i, i + 3) and condition_overlap(1, element, 0, 2)])
        for i in range(1, len(overlap_bounds[0]) - 3, 2):
            for j in range(1, len(overlap_bounds[1]) - 3, 2):
                list_subd_elements.append([idx for idx, element in enumerate(area_elements) if condition_overlap(0, element, i, i + 3) and condition_overlap(1, element, j, j + 3)])
        for i in range(1, len(overlap_bounds[0]) - 3, 2):
            list_subd_elements.append([idx for idx, element in enumerate(area_elements) if condition_overlap(0, element, i, i + 3) and condition_overlap(1, element, -3, -1)])

        list_subd_elements.append([idx for idx, element in enumerate(area_elements) if condition_overlap(0, element, -3, -1) and condition_overlap(1, element, 0, 2)])
        for i in range(1, len(overlap_bounds[1]) - 3, 2):
            list_subd_elements.append([idx for idx, element in enumerate(area_elements) if condition_overlap(0, element, -3, -1) and condition_overlap(1, element, i, i + 3)])
        list_subd_elements.append([idx for idx, element in enumerate(area_elements) if condition_overlap(0, element, -3, -1) and condition_overlap(1, element, -3, -1)])

    time_list.append(time.time() - init_time)

    init_time = time.time()
    for list_elements in list_subd_elements:
        list_full_subd_elements.append([area_elements[element] for element in list_elements])
    time_list.append(time.time() - init_time)

    init_time = time.time()
    for list_elements in list_full_subd_elements:
        list_subd_points.append(np.unique(list_elements))
    time_list.append(time.time() - init_time)

    init_time = time.time()
    for list_points in list_subd_points:
        list_subd_points_coords.append(np.array([area_points_coords[i].tolist() for i in list_points]))
    time_list.append(time.time() - init_time)

    init_time = time.time()
    for element, list_element_points in enumerate(area_elements):
        for point in list_element_points:
            if point in dict_elements_contain_point.keys():
                dict_elements_contain_point[point].append(element)
            else:
                dict_elements_contain_point[point] = [element]
    time_list.append(time.time() - init_time)

    init_time = time.time()
    for index_subd, list_elements in enumerate(list_subd_elements):
        temp_list = []
        temp_list_overlap = []
        dict_elements_contain_point_subd = {}
        
        for element in list_elements:
            for point in area_elements[element]:
                if point in dict_elements_contain_point_subd.keys():
                    dict_elements_contain_point_subd[point].append(element)
                else:
                    dict_elements_contain_point_subd[point] = [element]

        for point, list_elements in dict_elements_contain_point_subd.items():
            if list_elements != dict_elements_contain_point[point]:
                temp_list.append(point)
                temp_list_overlap.append(point)
        for point in area_boundary_points:
            if point in list_subd_points[index_subd]:
                temp_list.append(point)

        subd_boundary_points.append(list(set(temp_list)))
        subd_boundary_overlap_points.append(list(set(temp_list_overlap)))
    time_list.append(time.time() - init_time)

    return list_full_subd_elements, list_subd_elements, list_subd_points, list_subd_points_coords, subd_boundary_overlap_points, dict_elements_contain_point


def calculate_element_area(p_1, p_2, p_3):
    return abs((p_1[0] - p_3[0]) * (p_2[1] - p_3[1]) - (p_2[0] - p_3[0]) * (p_1[1] - p_3[1])) / 2

def create_barycentric_coords(element, points):
    x_points = np.array([points[element[i], 0] for i in range(3)])
    y_points = np.array([points[element[i], 1] for i in range(3)])

    a = np.array([x_points[1] * y_points[2] - x_points[2] * y_points[1], 
                    x_points[2] * y_points[0] - x_points[0] * y_points[2],
                    x_points[0] * y_points[1] - x_points[1] * y_points[0]] )
    b = np.array([y_points[1]-y_points[2], y_points[2]-y_points[0], y_points[0]-y_points[1]])
    c = np.array([x_points[2]-x_points[1], x_points[0]-x_points[2], x_points[1]-x_points[0]])

    A = 0.5 * np.linalg.det(np.vstack((np.ones_like(x_points), x_points, y_points)))
    dict_temp = {'barycentric_coords': np.vstack((a, b, c)), 'area_of_element': A}
    return dict_temp


if __name__ == "__main__":
    pass
