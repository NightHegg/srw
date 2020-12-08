import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg, cgs
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from itertools import combinations, combinations_with_replacement
from itertools import groupby
import matplotlib.pyplot as plt
import sys
import os
import math
import time

def benchmark(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        u, Eps, Sigma, graph = func(*args, **kwargs)
        end_time = time.time()
        print(f"Время выполнения обычной задачи: {end_time - start_time}\n")
        return u, Eps, Sigma, graph
    return wrapper

def benchmark_schwarz(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        u, Eps, Sigma, graph, it = func(*args, **kwargs)
        end_time = time.time()
        print(f"Время выполнения методом Шварца: {end_time - start_time}\n"
              f"Количество выполненных итераций: {it}")
        return u, Eps, Sigma, graph, it
    return wrapper

def calculate_eps(area_elements, area_points_coords, dimTask, u_current, coef_u):
    Eps=[]
    for element in area_elements:
        B, _ = calculate_local_matrix_stiffness(element, area_points_coords, dimTask)
        Eps.append(np.dot(B, np.ravel(np.array([u_current[element[0]], u_current[element[1]], u_current[element[2]]])).transpose()))
    return np.array(Eps)

def calculate_matrix_stiffness(area_elements, area_points_coords, D, dimTask):
    K = np.zeros((len(area_points_coords) * 2, len(area_points_coords) * 2))
    for element in area_elements:
                B, A = calculate_local_matrix_stiffness(element, area_points_coords, dimTask)
                K_element = B.T @ D @ B * A
                for i in range(3):
                    for j in range(3):
                        K[element[i]*dimTask:element[i]*dimTask+2, element[j]*dimTask:element[j]*dimTask+2] += K_element[i*dimTask:i*dimTask+2,j*dimTask:j*dimTask+2]
    return K

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
    return row, col, data

def calculate_crit_convergence(u_current, u_previous, area_points_coords, dimTask, relation_PointsElements, coef_u):
    first_sum, second_sum, relative_error = 0, 0, 0
    for idx, val in enumerate(u_current):
        if val[1] and abs(val[0]) > abs(val[1]**2):
            relative_error == np.linalg.norm(np.copy(val) - np.copy(u_previous[idx]))**2 / np.linalg.norm(np.copy(val))**2
            s = sum([calculate_local_matrix_stiffness(i, area_points_coords + u_current, dimTask)[1] for i in relation_PointsElements[idx]]) / 3
            first_sum += s * relative_error
            second_sum += s
        if val[1]:
            relative_error = np.linalg.norm(np.copy(val[1]) - np.copy(u_previous[idx, 1]))**2 / np.linalg.norm(np.copy(val[1]))**2
            s = sum([calculate_local_matrix_stiffness(i, area_points_coords + u_current * coef_u, dimTask)[1] for i in relation_PointsElements[idx]]) / 3
            first_sum += s * relative_error
            second_sum += s

        
    return math.sqrt(first_sum / second_sum)

def plot_displacements(area_points_coords, area_elements, coef_u):
    def temp(u):
        fig, ax = plt.subplots()
        ax.triplot(area_points_coords[:,0], area_points_coords[:,1], area_elements.copy())
        ax.triplot(u[:, 0] * coef_u + area_points_coords[:, 0], u[:,1] * coef_u + area_points_coords[:, 1], area_elements.copy())
        ax.plot(area_points_coords[:,0] + u[:, 0] * coef_u, area_points_coords[:,1] + u[:, 1] * coef_u, 'o')

        fig.set_figwidth(10)
        fig.set_figheight(7)
        fig.set_facecolor('mintcream')

        plt.show()
    return temp

def calculate_local_matrix_stiffness(element, points, dimTask):
    '''
    element - элемент, в котором идёт расчёт \n
    points - массив координат \n
    dimTask - размерность \n
    '''
    B = np.zeros((3, 6))
    x_points = np.array([points[element[0], 0], points[element[1], 0], points[element[2], 0]])
    y_points = np.array([points[element[0], 1], points[element[1], 1], points[element[2], 1]])
    a = np.array([x_points[1] * y_points[2] - x_points[2] * y_points[1], 
                  x_points[2] * y_points[0] - x_points[0] * y_points[2],
                  x_points[0] * y_points[1] - x_points[1] * y_points[0]] )
    b = np.array([y_points[1]-y_points[2], y_points[2]-y_points[0], y_points[0]-y_points[1]])
    c = np.array([x_points[2]-x_points[1], x_points[0]-x_points[2], x_points[1]-x_points[0]])
    A = 0.5*np.linalg.det(np.array([[1, 1, 1], [x_points[0], x_points[1], x_points[2]], 
                                  [y_points[0], y_points[1], y_points[2]]]).transpose())
    for i in range(len(element)):
        B[:, i*dimTask:i*dimTask + 2] += np.array([[b[i], 0], [0, c[i]], [c[i]/2, b[i]/2]])/2/A
    return B, A

def bound_condition_neumann(F, element, neumann_points_Local, dimTask, value, points, dim):
    L = list(set(element) & set(neumann_points_Local))
    len = np.linalg.norm(np.array(points[L[0]]) - np.array(points[L[1]]))
    for node in L:
        F[node * dimTask + dim] += value * len / 2
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
    if sparse.issparse(K):
        K_col = K.getcol(node * dimTask + dim).toarray()
    else:
        K_col = K[:, node * dimTask + dim]
    F -= np.ravel(K_col) * value
    K[node * dimTask + dim, :] = 0
    K[:, node * dimTask + dim] = 0

    K[node * dimTask + dim, node * dimTask + dim] = 1
    F[node * dimTask + dim] = value

    return K, F

def read_task(cur_task):
    dirichlet_conditions = []
    neumann_conditions = []

    with open(f"tests/task_{cur_task}.dat") as f:
        dimTask = int(f.readline())
        coef_overlap = float(f.readline())
        E, nyu = list(map(float, f.readline().split()))
        coef_u, coef_sigma = list(map(float, f.readline().split()))
        for _ in range(int(f.readline())):
            dirichlet_conditions.append([int(val) if idx in [0, 1] else float(val) for idx, val in enumerate(f.readline().split())])
        for _ in range(int(f.readline())):
            neumann_conditions.append([int(val) if idx == 0 else float(val) for idx, val in enumerate(f.readline().split())])

    return dirichlet_conditions, neumann_conditions, dimTask, coef_overlap, E, nyu, coef_u, coef_sigma

def read_mesh(cur_mesh):
    bounds = []
    area_points_coords = []
    area_elements = []

    with open(f"tests/mesh/mesh_{cur_mesh}.dat") as f:
        for _ in range(int(f.readline())):
            bounds.append([float(x) for x in f.readline().split()])
        for _ in range(int(f.readline())):
            area_points_coords.append([float(val) for val in f.readline().split()])
        for _ in range(int(f.readline())):
            area_elements.append([int(val) for val in f.readline().split()])
    
    return np.array(bounds), np.array(area_points_coords), area_elements


def schwarz_multiplicative_method(    cur_task, cur_mesh, cur_amnt_subds, coef_convergence, func = sparse.linalg.spsolve, *args, **kwargs):
    """
    Программа возвращает:
    u_current, Eps, Sigma, graph, it
    """
    *arg, = read_task(cur_task)

    dirichlet_conditions = arg[0]
    neumann_conditions   = arg[1]
    dimTask              = arg[2]
    coef_overlap         = arg[3]
    E                    = arg[4]
    nyu                  = arg[5]
    coef_u               = arg[6]
    coef_sigma           = arg[7]

    area_bounds, area_points_coords, area_elements  = read_mesh(cur_mesh)

    area_boundary_points = [idx for idx, val in enumerate(area_points_coords) if val[0] in [area_bounds[0, 0], area_bounds[1, 0]] or val[1] in [area_bounds[0, 1], area_bounds[2, 1]]]
    
    area_limits = lambda cond, type: [area_bounds[cond[0], type], area_bounds[cond[0] + 1, type]]

    dirichlet_points = [[[idx for idx, val in enumerate(area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))], cond[1:]] for cond in dirichlet_conditions]

    neumann_points = [[[idx for idx, val in enumerate(area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))], cond[1], cond[2]] for cond in neumann_conditions]

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
        
        subd_elements.append([element for element in area_elements if condition_overlap(0, element, 0, 2) and condition_overlap(1, element, -3, -1)])
        for i in range(1, len(overlap_bounds[0]) - 3, 2):
            subd_elements.append([element for element in area_elements if condition_overlap(0, element, i, i + 3) and condition_overlap(1, element, -3, -1)])
        subd_elements.append([element for element in area_elements if condition_overlap(0, element, -3, -1) and condition_overlap(1, element, -3, -1)])

        for j in range(1, len(overlap_bounds[1]) - 3, 2):
            subd_elements.append([element for element in area_elements if condition_overlap(0, element, 0, 2) and condition_overlap(1, element, j, j + 3)])
        for i in range(1, len(overlap_bounds[0]) - 3, 2):
            for j in range(1, len(overlap_bounds[1]) - 3, 2):
                subd_elements.append([element for element in area_elements if condition_overlap(0, element, i, i + 3) and condition_overlap(1, element, j, j + 3)])
        for j in range(1, len(overlap_bounds[1]) - 3, 2):
            subd_elements.append([element for element in area_elements if condition_overlap(0, element, -3, -1) and condition_overlap(1, element, j, j + 3)])

        subd_elements.append([element for element in area_elements if condition_overlap(0, element, 0, 2) and condition_overlap(1, element, 0, 2)])
        for i in range(1, len(overlap_bounds[1]) - 3, 2):
            subd_elements.append([element for element in area_elements if condition_overlap(0, element, i, i + 3) and condition_overlap(1, element, 0, 2)])
        subd_elements.append([element for element in area_elements if condition_overlap(0, element, -3, -1) and condition_overlap(1, element, 0, 2)])

    for subd in subd_elements:
        subd_points.append([el for el, _ in groupby(sorted(sum([i for i in subd], [])))])
    
    for subd in subd_points:
        subd_points_coords.append(np.array([area_points_coords[i].tolist() for i in subd]))

    for idx, val in enumerate(area_points_coords):
        relation_points_elements[idx] = [element for element in area_elements if idx in element]
    
    for idv, subd in enumerate(subd_elements):
        temp_list = []
        temp_list_overlap = []
        relation_points_elements_coords = {}
        for point in subd_points[idv]:
            relation_points_elements_coords[point] = [element for element in subd if point in element]
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

    graph = plot_displacements(area_points_coords, area_elements, coef_u)
    amnt_iterations = 0
    u_current = np.zeros((area_points_coords.shape[0], 2))
    D = np.array([[1, nyu/(1-nyu), 0], [nyu/(1 - nyu), 1, 0], [0, 0, (1 - 2 * nyu) / 2 / (1-nyu)]]) * E * (1-nyu) / (1-2 * nyu) / (1+nyu)

    K_array = []
    for idv, subd in enumerate(subd_elements):
        ratioPoints_LocalGlobal = dict(zip(range(len(subd_points[idv])), subd_points[idv]))
        ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

        subdElements_Local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)
        row, col, data = calculate_sparse_matrix_stiffness(subdElements_Local, subd_points_coords[idv], D, dimTask)
        K = coo_matrix((data, (row, col)), shape = (subd_points_coords[idv].size, subd_points_coords[idv].size)).tolil()
        K_array.append(K)
    

    while True:
        u_previous = np.copy(u_current)
        for idv, subd in enumerate(subd_elements):
            ratioPoints_LocalGlobal = dict(zip(range(len(subd_points[idv])), subd_points[idv]))
            ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

            F = np.zeros(subd_points_coords[idv].size)
            subdElements_Local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)
            
            row, col, data = calculate_sparse_matrix_stiffness(subdElements_Local, subd_points_coords[idv], D, dimTask)

            K = coo_matrix((data, (row, col)), shape = (F.size, F.size)).tolil()
            
            for condition in dirichlet_points:
                listPoints = list(set(condition[0]) & set(subd_points[idv]))
                for node in listPoints:
                    if condition[1][0] == 2:
                        K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], 0)
                        K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][2], 1)
                    else:
                        K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], condition[1][0])

            rpGL_Change = lambda L: [ratioPoints_GlobalLocal[x] for x in L]

            for condition in neumann_points:
                listPoints = list(set(condition[0]) & set(subd_points[idv]))
                segmentPoints = list(combinations(listPoints, 2))
                for element in [element for element in subd for x in segmentPoints if x[0] in element and x[1] in element]:                
                    F = bound_condition_neumann(F, rpGL_Change(element), rpGL_Change(listPoints), dimTask, condition[1], subd_points_coords[idv], 0)
                    F = bound_condition_neumann(F, rpGL_Change(element), rpGL_Change(listPoints), dimTask, condition[2], subd_points_coords[idv], 1)

            listPoints_Schwarz = sum([list(set(subd_boundary_overlap_points[idv]) & set(subd)) for idx, subd in enumerate(subd_points) if idx != idv], [])

            for node in listPoints_Schwarz:
                K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], u_current[node, 0], dim = 0)
                K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], u_current[node, 1], dim = 1)

            [*arg,] = func(K.tocsr(), F)
            u_subd = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

            for x in list(ratioPoints_LocalGlobal.keys()):
                u_current[ratioPoints_LocalGlobal[x], :] = np.copy(u_subd[x, :])

        amnt_iterations += 1

        crit_convergence = calculate_crit_convergence(u_current, u_previous, area_points_coords, dimTask, relation_points_elements, coef_u)
        #print(f"Multiplicative CritConvergence = {critConvergence}", end = "\r")
        if crit_convergence < coef_convergence:
            break
    
    Eps = calculate_eps(area_elements, area_points_coords, dimTask, u_current, coef_u)
    Sigma = np.dot(D, Eps.transpose())

    return u_current, Eps, Sigma, graph, amnt_iterations

def schwarz_additive_method(          cur_task, cur_mesh, cur_amnt_subds, coef_convergence, func = sparse.linalg.spsolve, *args, **kwargs):
    """
    Программа возвращает:
    u_current, Eps, Sigma, graph, it
    """
    *arg, = read_task(cur_task)

    dirichlet_conditions = arg[0]
    neumann_conditions   = arg[1]
    dimTask              = arg[2]
    coef_overlap         = arg[3]
    E                    = arg[4]
    nyu                  = arg[5]
    coef_u               = arg[6]
    coef_sigma           = arg[7]

    coef_alpha = args[0][0]

    area_bounds, area_points_coords, area_elements  = read_mesh(cur_mesh)

    area_boundary_points = [idx for idx, val in enumerate(area_points_coords) if val[0] in [area_bounds[0, 0], area_bounds[1, 0]] or val[1] in [area_bounds[0, 1], area_bounds[2, 1]]]
    
    area_limits = lambda cond, type: [area_bounds[cond[0], type], area_bounds[cond[0] + 1, type]]
    dirichlet_points = [[[idx for idx, val in enumerate(area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))], cond[1:]] for cond in dirichlet_conditions]

    neumann_points = [[[idx for idx, val in enumerate(area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))], cond[1], cond[2]] for cond in neumann_conditions]

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
        
        subd_elements.append([element for element in area_elements if condition_overlap(0, element, 0, 2) and condition_overlap(1, element, -3, -1)])
        for i in range(1, len(overlap_bounds[0]) - 3, 2):
            subd_elements.append([element for element in area_elements if condition_overlap(0, element, i, i + 3) and condition_overlap(1, element, -3, -1)])
        subd_elements.append([element for element in area_elements if condition_overlap(0, element, -3, -1) and condition_overlap(1, element, -3, -1)])

        for j in range(1, len(overlap_bounds[1]) - 3, 2):
            subd_elements.append([element for element in area_elements if condition_overlap(0, element, 0, 2) and condition_overlap(1, element, j, j + 3)])
        for i in range(1, len(overlap_bounds[0]) - 3, 2):
            for j in range(1, len(overlap_bounds[1]) - 3, 2):
                subd_elements.append([element for element in area_elements if condition_overlap(0, element, i, i + 3) and condition_overlap(1, element, j, j + 3)])
        for j in range(1, len(overlap_bounds[1]) - 3, 2):
            subd_elements.append([element for element in area_elements if condition_overlap(0, element, -3, -1) and condition_overlap(1, element, j, j + 3)])

        subd_elements.append([element for element in area_elements if condition_overlap(0, element, 0, 2) and condition_overlap(1, element, 0, 2)])
        for i in range(1, len(overlap_bounds[1]) - 3, 2):
            subd_elements.append([element for element in area_elements if condition_overlap(0, element, i, i + 3) and condition_overlap(1, element, 0, 2)])
        subd_elements.append([element for element in area_elements if condition_overlap(0, element, -3, -1) and condition_overlap(1, element, 0, 2)])

    for subd in subd_elements:
        subd_points.append([el for el, _ in groupby(sorted(sum([i for i in subd], [])))])
    
    for subd in subd_points:
        subd_points_coords.append(np.array([area_points_coords[i].tolist() for i in subd]))

    for idx, val in enumerate(area_points_coords):
        relation_points_elements[idx] = [element for element in area_elements if idx in element]
    
    for idv, subd in enumerate(subd_elements):
        temp_list = []
        temp_list_overlap = []
        relation_points_elements_coords = {}
        for point in subd_points[idv]:
            relation_points_elements_coords[point] = [element for element in subd if point in element]
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

    graph = plot_displacements(area_points_coords, area_elements, coef_u)
    amnt_iterations = 0
    u_current = np.zeros((area_points_coords.shape[0], 2))
    D = np.array([[1, nyu/(1-nyu), 0], [nyu/(1-nyu), 1, 0], [0, 0, (1-2 * nyu) / 2 / (1-nyu)]]) * E * (1-nyu) / (1-2 * nyu) / (1+nyu)

    while True:
        u_previous = np.copy(u_current)
        u_current = np.zeros_like(u_previous)
        u_current_temp = np.copy(u_previous)
        u_sum = np.zeros_like(u_current)
        for idv, subd in enumerate(subd_elements):

            ratioPoints_LocalGlobal = dict(zip(range(len(subd_points[idv])), subd_points[idv]))
            ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

            F = np.zeros(subd_points_coords[idv].size)
            subdElements_Local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)
            
            row, col, data = calculate_sparse_matrix_stiffness(subdElements_Local, subd_points_coords[idv], D, dimTask)

            K = lil_matrix(coo_matrix((data, (row, col)), shape = (F.size, F.size)))

            for condition in dirichlet_points:
                listPoints = list(set(condition[0]) & set(subd_points[idv]))
                for node in listPoints:
                    if condition[1][0] == 2:
                        K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], 0)
                        K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][2], 1)
                    else:
                        K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], condition[1][0])

            rpGL_Change = lambda L: [ratioPoints_GlobalLocal[x] for x in L]

            for condition in neumann_points:
                listPoints = list(set(condition[0]) & set(subd_points[idv]))
                segmentPoints = list(combinations(listPoints, 2))
                for element in [element for element in subd for x in segmentPoints if x[0] in element and x[1] in element]:                
                    F = bound_condition_neumann(F, rpGL_Change(element), rpGL_Change(listPoints), dimTask, condition[1], subd_points_coords[idv], 0)
                    F = bound_condition_neumann(F, rpGL_Change(element), rpGL_Change(listPoints), dimTask, condition[2], subd_points_coords[idv], 1)

            listPoints_Schwarz = list(set(sum([list(set(subd_boundary_overlap_points[idv]) & set(subd)) for idx, subd in enumerate(subd_points) if idx != idv], [])))

            for node in listPoints_Schwarz:
                K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], u_previous[node, 0], dim = 0)
                K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], u_previous[node, 1], dim = 1)

            [*arg,] = func(K.tocsr(), F)
            u_subd = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

            for x in list(ratioPoints_LocalGlobal.keys()):
                u_current_temp[ratioPoints_LocalGlobal[x], :] = np.copy(u_subd[x, :])

            u_sum += (u_current_temp - u_previous)
            #graph(u_current_temp)
        amnt_iterations += 1
        
        u_current = np.copy(u_previous) + (coef_alpha * u_sum)
        #graph(u_current)
        crit_convergence = calculate_crit_convergence(u_current, u_previous, area_points_coords, dimTask, relation_points_elements, coef_u)
        #print(f"{crit_convergence}", end = "\r")
        if crit_convergence < coef_convergence:
            break
    
    Eps = calculate_eps(area_elements, area_points_coords, dimTask, u_current, coef_u)
    Sigma = np.dot(D, Eps.transpose())

    return u_current, Eps, Sigma, graph, amnt_iterations

def schwarz_two_level_additive_method(cur_task, cur_mesh, cur_amnt_subds, coef_convergence, func = sparse.linalg.spsolve, *args, **kwargs):
    """
    Программа возвращает:
    u_current, Eps, Sigma, graph, it
    """
    *arg, = read_task(cur_task)

    dirichlet_conditions = arg[0]
    neumann_conditions   = arg[1]
    dimTask              = arg[2]
    coef_overlap         = arg[3]
    E                    = arg[4]
    nyu                  = arg[5]
    coef_u               = arg[6]
    coef_sigma           = arg[7]

    coef_alpha = args[0][0]
    cur_coarse_mesh = args[0][1]

    area_bounds, area_points_coords, area_elements = read_mesh(cur_mesh)

    area_boundary_points = [idx for idx, val in enumerate(area_points_coords) if val[0] in [area_bounds[0, 0], area_bounds[1, 0]] or val[1] in [area_bounds[0, 1], area_bounds[2, 1]]]
    
    area_limits = lambda cond, type: [area_bounds[cond[0], type], area_bounds[cond[0] + 1, type]]
    dirichlet_points = [[[idx for idx, val in enumerate(area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))], cond[1:]] for cond in dirichlet_conditions]

    neumann_points = [[[idx for idx, val in enumerate(area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))], cond[1], cond[2]] for cond in neumann_conditions]

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
        subd_points.append([el for el, _ in groupby(sorted(sum([i for i in subd], [])))])
    
    for subd in subd_points:
        subd_points_coords.append(np.array([area_points_coords[i].tolist() for i in subd]))

    for idx, val in enumerate(area_points_coords):
        relation_points_elements[idx] = [element for element in area_elements if idx in element]
    
    for idv, subd in enumerate(subd_elements):
        temp_list = []
        temp_list_overlap = []
        relation_points_elements_coords = {}
        for point in subd_points[idv]:
            relation_points_elements_coords[point] = [element for element in subd if point in element]
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

    graph = plot_displacements(area_points_coords, area_elements, coef_u)
    amnt_iterations = 0
    u_current = np.zeros((area_points_coords.shape[0], 2))
    D = np.array([[1, nyu/(1-nyu), 0], [nyu/(1-nyu), 1, 0], [0, 0, (1-2 * nyu) / 2 / (1-nyu)]]) * E * (1-nyu) / (1-2 * nyu) / (1+nyu)

    while True:
        u_previous = np.copy(u_current)
        u_current = np.zeros_like(u_previous)
        u_current_temp = np.copy(u_previous)
        u_sum = np.zeros_like(u_current)
        for idv, subd in enumerate(subd_elements):
            ratioPoints_LocalGlobal = dict(zip(range(len(subd_points[idv])), subd_points[idv]))
            ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

            F = np.zeros(subd_points_coords[idv].size)
            subdElements_Local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)
            
            row, col, data = calculate_sparse_matrix_stiffness(subdElements_Local, subd_points_coords[idv], D, dimTask)

            K = lil_matrix(coo_matrix((data, (row, col)), shape = (F.size, F.size)))

            for condition in dirichlet_points:
                listPoints = list(set(condition[0]) & set(subd_points[idv]))
                for node in listPoints:
                    if condition[1][0] == 2:
                        K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], 0)
                        K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][2], 1)
                    else:
                        K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], condition[1][0])

            rpGL_Change = lambda L: [ratioPoints_GlobalLocal[x] for x in L]

            for condition in neumann_points:
                listPoints = list(set(condition[0]) & set(subd_points[idv]))
                segmentPoints = list(combinations(listPoints, 2))
                for element in [element for element in subd for x in segmentPoints if x[0] in element and x[1] in element]:                
                    F = bound_condition_neumann(F, rpGL_Change(element), rpGL_Change(listPoints), dimTask, condition[1], subd_points_coords[idv], 0)
                    F = bound_condition_neumann(F, rpGL_Change(element), rpGL_Change(listPoints), dimTask, condition[2], subd_points_coords[idv], 1)

            listPoints_Schwarz = sum([list(set(subd_boundary_overlap_points[idv]) & set(subd)) for idx, subd in enumerate(subd_points) if idx != idv], [])

            for node in listPoints_Schwarz:
                K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], u_previous[node, 0], dim = 0)
                K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], u_previous[node, 1], dim = 1)

            [*arg,] = func(K.tocsr(), F)
            u_subd = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

            for x in list(ratioPoints_LocalGlobal.keys()):
                u_current_temp[ratioPoints_LocalGlobal[x], :] = np.copy(u_subd[x, :])

            u_sum += (u_current_temp - u_previous)
        amnt_iterations += 1
        
        

        u_current = np.copy(u_previous) + (coef_alpha * u_sum)
        crit_convergence = calculate_crit_convergence(u_current, u_previous, area_points_coords, dimTask, relation_points_elements, coef_u)
        if crit_convergence < coef_convergence:
            break
    
    Eps = calculate_eps(area_elements, area_points_coords, dimTask, u_current, coef_u)
    Sigma = np.dot(D, Eps.transpose())

    return u_current, Eps, Sigma, graph, amnt_iterations

def basic_sparse_method(cur_task, cur_mesh, func = sparse.linalg.spsolve):
    """
    Программа возвращает:
    u_current, Eps, Sigma, graph
    """
    *arg, = read_task(cur_task)

    dirichlet_conditions = arg[0]
    neumann_conditions   = arg[1]
    dimTask              = arg[2]
    coef_overlap          = arg[3]
    E                    = arg[4]
    nyu                  = arg[5]
    coef_u               = arg[6]
    coef_sigma           = arg[7]

    bounds, area_points_coords, area_elements  = read_mesh(cur_mesh)

    D = np.array([[1, nyu/(1-nyu), 0], [nyu/(1-nyu), 1, 0], [0, 0, (1-2 * nyu) / 2 / (1-nyu)]]) * E * (1-nyu) / (1-2 * nyu) / (1+nyu)
    
    area_limits = lambda cond, type: [bounds[cond[0], type], bounds[cond[0] + 1, type]]

    dirichlet_points = [[[idx for idx, val in enumerate(area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))], cond[1:]] for cond in dirichlet_conditions]

    neumann_points = [[[idx for idx, val in enumerate(area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))], cond[1], cond[2]] for cond in neumann_conditions]

    areaBoundaryPoints = [idx for idx, val in enumerate(area_points_coords) if val[0] in [bounds[0, 0], bounds[1, 0]] or val[1] in [bounds[0, 1], bounds[2, 1]]]
    areaPoints = [x for x in range(len(area_points_coords))]

    #dirichlet_pointsAll = sorted(list(set(sum([side[0] for side in dirichlet_points], []))))
    #neumann_pointsAll = sorted(list(set(sum([side[0] for side in neumann_points], []))))
    #areaBoundaryPoints_Coords = [area_points_coords[x] for x in areaBoundaryPoints]

    F = np.zeros(area_points_coords.size)
    
    row, col, data = calculate_sparse_matrix_stiffness(area_elements, area_points_coords, D, dimTask)

    K = lil_matrix(coo_matrix((data, (row, col)), shape = (F.size, F.size)))

    for condition in dirichlet_points:
        for node in condition[0]:
            if condition[1][0] == 2:
                K, F = bound_condition_dirichlet(K, F, dimTask, node, condition[1][1], 0)
                K, F = bound_condition_dirichlet(K, F, dimTask, node, condition[1][2], 1)
            else:
                K, F = bound_condition_dirichlet(K, F, dimTask, node, condition[1][1], condition[1][0])
    
    for condition in neumann_points:
        for element in [element for element in area_elements for x in list(combinations(condition[0], 2)) if x[0] in element and x[1] in element]:
            F = bound_condition_neumann(F, element, condition[0], dimTask, condition[1], area_points_coords, 0)
            F = bound_condition_neumann(F, element, condition[0], dimTask, condition[2], area_points_coords, 1)
    
    *arg, = func(K.tocsr(), F)

    u = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

    Eps = calculate_eps(area_elements, area_points_coords, dimTask, u, coef_u)
    Sigma = np.dot(D, Eps.transpose())

    graph = plot_displacements(area_points_coords, area_elements, coef_u)

    return u, Eps, Sigma, graph

if __name__ == "__main__":
    cur_task = 1
    cur_mesh = 10
    cur_amnt_subds = [2, 1]
    coef_alpha = 0.5
    func = sparse.linalg.spsolve
    coef_convergence = 1e-3
    cur_coarse_mesh = 2

    @benchmark
    def basic_task(method, cur_task, cur_mesh, func):
        u, Eps, Sigma, graph = method(cur_task, cur_mesh, func)
        return u, Eps, Sigma, graph

    @benchmark_schwarz
    def schwarz_task(method, cur_task, cur_mesh, cur_amnt_subds, coef_convergence, func, *args):
        u, Eps, Sigma, graph, it = method(cur_task, cur_mesh, cur_amnt_subds, coef_convergence, func, args)
        print(f"min={abs(abs(min(Sigma[1])) - abs(2e+7)):.4e}, max={abs(abs(max(Sigma[1])) - abs(2e+7)):.4e}")
        graph(u)
        return u, Eps, Sigma, graph, it

    schwarz_task(schwarz_multiplicative_method, cur_task, cur_mesh, cur_amnt_subds, coef_convergence, func, coef_alpha, cur_coarse_mesh)
