import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from itertools import groupby
import numpy as np
import scipy
from scipy.sparse import coo_matrix, lil_matrix
import math


def calculate_crit_convergence(u_current, u_previous, area_points_coords, dimTask, relation_PointsElements, coef_u):
    first_sum, second_sum, relative_error = 0, 0, 0
    for idx, value in enumerate(u_current):
        if value[1] and abs(value[0]) > abs(value[1]**2):
            relative_error = np.linalg.norm(value - u_previous[idx])**2 / np.linalg.norm(value)**2
            s = sum([calculate_local_matrix_stiffness(i, area_points_coords + u_current, dimTask)[1] for i in relation_PointsElements[idx]]) / 3
            first_sum += s * relative_error
            second_sum += s
        if value[1]:
            relative_error = np.linalg.norm(np.copy(value[1]) - np.copy(u_previous[idx, 1]))**2 / np.linalg.norm(np.copy(value[1]))**2
            s = sum([calculate_local_matrix_stiffness(i, area_points_coords + u_current * coef_u, dimTask)[1] for i in relation_PointsElements[idx]]) / 3
            first_sum += s * relative_error
            second_sum += s
    # for idx, value in enumerate(u_current):
    #     if any(abs(value) > 1e-9):
    #         relative_error = np.linalg.norm(value - u_previous[idx])**2 / np.linalg.norm(value)**2
    #         s = sum([calculate_local_matrix_stiffness(i, area_points_coords + u_current, dimTask)[1] for i in relation_PointsElements[idx]]) / 3
    #         first_sum += s * relative_error
    #         second_sum += s

    return math.sqrt(first_sum / second_sum)


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


def bound_condition_neumann(F, dict_neumann_points, points_coords, dim_task):
    points = list(dict_neumann_points.keys())
    len = np.linalg.norm(np.array(points_coords[points[1]]) - np.array(points_coords[points[0]]))
    for point, condition in dict_neumann_points.items():
        if math.isnan(condition[0]):
            F[point * dim_task + 1] += condition[1] * len / 2
        elif math.isnan(condition[1]):
            F[point * dim_task] += condition[0] * len / 2
        else:
            F[point * dim_task] += condition[0] * len / 2
            F[point * dim_task + 1] += condition[1] * len / 2
    return F


def bound_condition_dirichlet(K, F, dimTask, node, value, dim):
    '''
    K - матрица жёсткости \n
    F - матрица правой части \n
    dimTask - размерность задачи \n
    node - номер узла \n
    value - значение \n
    dim - по какой координате \n
    '''
    
    if scipy.sparse.issparse(K):
        K_col = K.getcol(node * dimTask + dim).toarray()
    else:
        K_col = K[:, node * dimTask + dim]
    F -= np.ravel(K_col) * value
    K[node * dimTask + dim, :] = 0
    K[:, node * dimTask + dim] = 0

    K[node * dimTask + dim, node * dimTask + dim] = 1
    F[node * dimTask + dim] = value

    return K, F


def calculate_subd_parameters(area_bounds, area_points_coords, area_elements, coef_overlap, cur_amnt_subds):
    temp_cond = lambda val, i: val[i] in [area_bounds[0, i], area_bounds[i + 1, i]]
    area_boundary_points = [idx for idx, val in enumerate(area_points_coords) if temp_cond(val, 0) or temp_cond(val, 1)]

    subd_bounds = []
    overlap_bounds = []
    subd_elements = []
    subd_points = []
    relation_points_elements = {}
    subd_boundary_overlap_points = []
    subd_boundary_points = []
    subd_internal_points = []
    subd_points_coords = []

    for idx, val in enumerate(cur_amnt_subds):
        temp_bounds = [area_bounds[idx, idx] + i*(area_bounds[idx + 1, idx] - area_bounds[idx, idx])/val for i in range(1, val)]
        subd_bounds.append([area_bounds[idx, idx]] + temp_bounds + [area_bounds[idx + 1, idx]])

    subd_limit = lambda x, func, type: func[x] + type * (func[x] - func[x - 1]) * coef_overlap
    for subd in subd_bounds:
        temp_bounds = sum([[subd_limit(x, subd, -1), subd_limit(x, subd, 1)] for x in range(1, len(subd)-1)], [])
        overlap_bounds.append([subd[0]] + temp_bounds + [subd[-1]])

    condition_overlap = lambda type, element, coef_1, coef_2: overlap_bounds[type][coef_1] < sum([area_points_coords[element[i], type] for i in range(3)])/3 < overlap_bounds[type][coef_2]

    if len(overlap_bounds[0]) == 2:
        subd_elements.append([element for element in area_elements if condition_overlap(1, element, 0, 2)])     
        for i in range(1, len(overlap_bounds[1]) - 3, 2):
            subd_elements.append([element for element in area_elements if condition_overlap(1, element, i, i + 3)])
        subd_elements.append([element for element in area_elements if condition_overlap(1, element, -3, -1)])
    elif len(overlap_bounds[1]) == 2:
        subd_elements.append([element for element in area_elements if condition_overlap(0, element, 0, 2)])
        for i in range(1, len(overlap_bounds[0]) - 3, 2):
            subd_elements.append([element for element in area_elements if condition_overlap(0, element, i, i + 3)])
        subd_elements.append([element for element in area_elements if condition_overlap(0, element, -3, -1)])
    else:
        
        subd_elements.append([element for element in area_elements if condition_overlap(0, element, 0, 2) and condition_overlap(1, element, 0, 2)])
        for i in range(1, len(overlap_bounds[1]) - 3, 2):
            subd_elements.append([element for element in area_elements if condition_overlap(0, element, 0, 2) and condition_overlap(1, element, i, i + 3)])
        subd_elements.append([element for element in area_elements if condition_overlap(0, element, 0, 2) and condition_overlap(1, element, -3, -1)])

        for i in range(1, len(overlap_bounds[0]) - 3, 2):
            subd_elements.append([element for element in area_elements if condition_overlap(0, element, i, i + 3) and condition_overlap(1, element, 0, 2)])
        for i in range(1, len(overlap_bounds[0]) - 3, 2):
            for j in range(1, len(overlap_bounds[1]) - 3, 2):
                subd_elements.append([element for element in area_elements if condition_overlap(0, element, i, i + 3) and condition_overlap(1, element, j, j + 3)])
        for i in range(1, len(overlap_bounds[0]) - 3, 2):
            subd_elements.append([element for element in area_elements if condition_overlap(0, element, i, i + 3) and condition_overlap(1, element, -3, -1)])

        subd_elements.append([element for element in area_elements if condition_overlap(0, element, -3, -1) and condition_overlap(1, element, 0, 2)])
        for i in range(1, len(overlap_bounds[1]) - 3, 2):
            subd_elements.append([element for element in area_elements if condition_overlap(0, element, -3, -1) and condition_overlap(1, element, i, i + 3)])
        subd_elements.append([element for element in area_elements if condition_overlap(0, element, -3, -1) and condition_overlap(1, element, -3, -1)])

    for subd in subd_elements:
        subd_points.append(np.unique(subd))

    for subd in subd_points:
        subd_points_coords.append(np.array([area_points_coords[i].tolist() for i in subd]))

    for idx, val in enumerate(area_points_coords):
        relation_points_elements[idx] = [list(element) for element in area_elements if idx in element]

    for idv, subd in enumerate(subd_elements):
        temp_list = []
        temp_list_overlap = []
        relation_points_elements_coords = {}
        for point in subd_points[idv]:
            relation_points_elements_coords[point] = [list(element) for element in subd if point in element]
        for idx, val in relation_points_elements_coords.items():
            if val != relation_points_elements[idx]:
                temp_list.append(idx)
                temp_list_overlap.append(idx)
        for point in area_boundary_points:
            if point in subd_points[idv]:
                temp_list.append(point)

        subd_boundary_points.append(list(set(temp_list)))
        subd_boundary_overlap_points.append(list(set(temp_list_overlap)))

    for idx, subd in enumerate(subd_elements):
        subd_internal_points.append(list(set(subd_points[idx]) - set(subd_boundary_points[idx])))
    
    return subd_elements, subd_points, subd_points_coords, subd_boundary_overlap_points, relation_points_elements


def calculate_element_area(p_1, p_2, p_3):
    return abs((p_1[0] - p_3[0]) * (p_2[1] - p_3[1]) - (p_2[0] - p_3[0]) * (p_1[1] - p_3[1])) / 2


def calculate_local_functions(element, points):
    def temp(chosen_point, i):
        '''
        element - элемент, в котором идёт расчёт \n
        points - массив координат \n
        dimTask - размерность \n
        '''
        x_points = np.array([points[element[i], 0] for i in range(3)])
        y_points = np.array([points[element[i], 1] for i in range(3)])

        a = np.array([x_points[1] * y_points[2] - x_points[2] * y_points[1], 
                      x_points[2] * y_points[0] - x_points[0] * y_points[2],
                      x_points[0] * y_points[1] - x_points[1] * y_points[0]] )
        b = np.array([y_points[1]-y_points[2], y_points[2]-y_points[0], y_points[0]-y_points[1]])
        c = np.array([x_points[2]-x_points[1], x_points[0]-x_points[2], x_points[1]-x_points[0]])

        A = 0.5 * np.linalg.det(np.vstack((np.ones_like(x_points), x_points, y_points)))
        return (a[i] + b[i] * chosen_point[0] + c[i] * chosen_point[1]) / 2 / A
    return temp


if __name__ == "__main__":
    pass
