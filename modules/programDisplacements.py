import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg, cgs
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from itertools import combinations, combinations_with_replacement
from itertools import groupby
import matplotlib.pyplot as plt


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

    D = np.array([[1, nyu/(1-nyu), 0], [nyu/(1 - nyu), 1, 0], [0, 0, (1 - 2 * nyu) / 2 / (1-nyu)]]) * E * (1-nyu) / (1-2 * nyu) / (1+nyu)

    area_bounds, area_points_coords, area_elements = read_mesh(cur_mesh)
    
    area_limits = lambda cond, type: [area_bounds[cond[0], type], area_bounds[cond[0] + 1, type]]
    dirichlet_points = [[[idx for idx, val in enumerate(area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))], cond[1:]] for cond in dirichlet_conditions]

    neumann_points = [[[idx for idx, val in enumerate(area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))], cond[1], cond[2]] for cond in neumann_conditions]

    *arg, = calculate_subd_parameters(area_bounds, area_points_coords, area_elements, coef_overlap)

    subd_elements                = arg[0]
    subd_points                  = arg[1]
    subd_points_coords           = arg[2]
    subd_boundary_overlap_points = arg[3]
    relation_points_elements     = arg[4]

    K_array = []
    for idv, subd in enumerate(subd_elements):
        ratioPoints_LocalGlobal = dict(zip(range(len(subd_points[idv])), subd_points[idv]))
        ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

        subd_elements_local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)
        row, col, data = calculate_sparse_matrix_stiffness(subd_elements_local, subd_points_coords[idv], D, dimTask)
        K = coo_matrix((data, (row, col)), shape = (subd_points_coords[idv].size, subd_points_coords[idv].size)).tolil()
        K_array.append(K)

    graph = plot_displacements(area_points_coords, area_elements, coef_u)

    amnt_iterations = 0
    u_current = np.zeros((area_points_coords.shape[0], 2))
    while True:
        u_previous = np.copy(u_current)
        for idv, subd in enumerate(subd_elements):
            ratioPoints_LocalGlobal = dict(zip(range(len(subd_points[idv])), subd_points[idv]))
            ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

            K = K_array[idv].copy()
            F = np.zeros(subd_points_coords[idv].size)
            
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

    *arg, = calculate_subd_parameters(area_bounds, area_points_coords, area_elements, coef_overlap)

    subd_elements                = arg[0]
    subd_points                  = arg[1]
    subd_points_coords           = arg[2]
    subd_boundary_overlap_points = arg[3]
    relation_points_elements     = arg[4]

    K_array = []
    for idv, subd in enumerate(subd_elements):
        ratioPoints_LocalGlobal = dict(zip(range(len(subd_points[idv])), subd_points[idv]))
        ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

        subd_elements_local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)
        row, col, data = calculate_sparse_matrix_stiffness(subd_elements_local, subd_points_coords[idv], D, dimTask)
        K = coo_matrix((data, (row, col)), shape = (subd_points_coords[idv].size, subd_points_coords[idv].size)).tolil()
        K_array.append(K)

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

            K = K_array[idv].copy()
            F = np.zeros(subd_points_coords[idv].size)

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

        amnt_iterations += 1
        
        u_current = np.copy(u_previous) + (coef_alpha * u_sum)

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

    D = np.array([[1, nyu/(1-nyu), 0], [nyu/(1-nyu), 1, 0], [0, 0, (1-2 * nyu) / 2 / (1-nyu)]]) * E * (1-nyu) / (1-2 * nyu) / (1+nyu)

    coef_alpha = args[0][0]
    cur_coarse_mesh = args[0][1]

    area_bounds, area_points_coords, area_elements = read_mesh(cur_mesh)
    _, area_coarse_points_coords, area_coarse_elements = read_mesh(cur_coarse_mesh)

    area_boundary_points = [idx for idx, val in enumerate(area_points_coords) if val[0] in [area_bounds[0, 0], area_bounds[1, 0]] or val[1] in [area_bounds[0, 1], area_bounds[2, 1]]]
    area_coarse_boundary_points = [idx for idx, val in enumerate(area_coarse_points_coords) if val[0] in [area_bounds[0, 0], area_bounds[1, 0]] or val[1] in [area_bounds[0, 1], area_bounds[2, 1]]]

    area_limits = lambda cond, type: [area_bounds[cond[0], type], area_bounds[cond[0] + 1, type]]
    dirichlet_points = [[[idx for idx, val in enumerate(area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))], cond[1:]] for cond in dirichlet_conditions]

    dirichlet_coarse_points = [[[idx for idx, val in enumerate(area_coarse_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))], cond[1:]] for cond in dirichlet_conditions]

    neumann_points = [[[idx for idx, val in enumerate(area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))], cond[1], cond[2]] for cond in neumann_conditions]

    *arg, = calculate_subd_parameters(area_bounds, area_points_coords, area_elements, coef_overlap)

    subd_bounds                  = arg[0]
    overlap_bounds               = arg[1]
    subd_elements                = arg[2]
    subd_points                  = arg[3]
    subd_points_coords           = arg[4]
    subd_boundary_overlap_points = arg[5]
    relation_points_elements     = arg[6]

    K_array = []
    for idv, subd in enumerate(subd_elements):
        ratioPoints_LocalGlobal = dict(zip(range(len(subd_points[idv])), subd_points[idv]))
        ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

        subd_elements_local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)
        row, col, data = calculate_sparse_matrix_stiffness(subd_elements_local, subd_points_coords[idv], D, dimTask)
        K = coo_matrix((data, (row, col)), shape = (subd_points_coords[idv].size, subd_points_coords[idv].size)).tolil()
        K_array.append(K)

    *arg_coarse, = calculate_subd_parameters(area_bounds, area_coarse_points_coords, area_coarse_elements, coef_overlap)
    
    subd_bounds_coarse                  = arg_coarse[0]
    overlap_bounds_coarse               = arg_coarse[1]
    subd_coarse_elements                = arg_coarse[2]
    subd_coarse_points                  = arg_coarse[3]
    subd_coarse_points_coords           = arg_coarse[4]
    subd_coarse_boundary_overlap_points = arg_coarse[5]
    relation_coarse_points_elements     = arg_coarse[6]

    K_coarse_array = []
    for idv, subd in enumerate(subd_coarse_elements):
        ratioPoints_LocalGlobal = dict(zip(range(len(subd_coarse_points[idv])), subd_coarse_points[idv]))
        ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

        subd_elements_local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)
        row, col, data = calculate_sparse_matrix_stiffness(subd_elements_local, subd_coarse_points_coords[idv], D, dimTask)
        K = coo_matrix((data, (row, col)), shape = (subd_coarse_points_coords[idv].size, subd_coarse_points_coords[idv].size)).tolil()
        K_coarse_array.append(K)
    
    u_current = np.zeros((area_points_coords.shape[0], 2))

    graph = plot_displacements(area_points_coords, area_bounds, area_elements, coef_u)

    plot_subd_boundaries(area_bounds, area_points_coords, subd_elements, subd_points_coords, subd_boundary_overlap_points, subd_bounds, overlap_bounds)

    amnt_iterations = 0
    while True:
        u_previous = np.copy(u_current)
        u_current = np.zeros_like(u_previous)

        u_current_temp = np.copy(u_previous)
        u_sum = np.zeros_like(u_current)
        u_special = np.ravel(np.zeros_like(u_current))
        for idv, subd in enumerate(subd_elements):
            
            ratioPoints_LocalGlobal = dict(zip(range(len(subd_points[idv])), subd_points[idv]))
            ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

            ratioPoints_LocalGlobal_coarse = dict(zip(range(len(subd_coarse_points[idv])), subd_coarse_points[idv]))
            ratioPoints_GlobalLocal_coarse = {v: k for k, v in ratioPoints_LocalGlobal_coarse.items()}
            K = K_array[idv].copy()
            F = np.zeros(subd_points_coords[idv].size)
            
            K_coarse = K_coarse_array[idv].copy()
            F_coarse = np.zeros(subd_coarse_points_coords[idv].size)

            for condition in dirichlet_points:
                listPoints = list(set(condition[0]) & set(subd_points[idv]))
                for node in listPoints:
                    if condition[1][0] == 2:
                        K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], 0)
                        K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][2], 1)
                    else:
                        K, F = bound_condition_dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], condition[1][0])

            for condition in dirichlet_coarse_points:
                listPoints = list(set(condition[0]) & set(subd_coarse_points[idv]))
                for node in listPoints:
                    if condition[1][0] == 2:
                        K_coarse, _ = bound_condition_dirichlet(K_coarse, F_coarse, dimTask, ratioPoints_GlobalLocal_coarse[node], condition[1][1], 0)
                        K_coarse, _ = bound_condition_dirichlet(K_coarse, F_coarse, dimTask, ratioPoints_GlobalLocal_coarse[node], condition[1][2], 1)
                    else:
                        K_coarse, _ = bound_condition_dirichlet(K_coarse, F_coarse, dimTask, ratioPoints_GlobalLocal_coarse[node], condition[1][1], condition[1][0])

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

            listPoints_Schwarz_coarse = sum([list(set(subd_coarse_boundary_overlap_points[idv]) & set(subd)) for idx, subd in enumerate(subd_coarse_points) if idx != idv], [])

            for node in listPoints_Schwarz_coarse:
                K_coarse, _ = bound_condition_dirichlet(K_coarse, F_coarse, dimTask, ratioPoints_GlobalLocal_coarse[node], u_previous[node, 0], dim = 0)
                K_coarse, _ = bound_condition_dirichlet(K_coarse, F_coarse, dimTask, ratioPoints_GlobalLocal_coarse[node], u_previous[node, 1], dim = 1)
            
            residual = F - np.dot(K.toarray(), np.ravel(u_previous[subd_points[idv]]))
            chosen_element = []
            for point in subd_coarse_points[idv]:
                chosen_element = []
                for element in subd_elements[idv]:
                    S_element = calculate_element_area(area_points_coords[element[0]], area_points_coords[element[1]], area_points_coords[element[2]])
                    S_1 = calculate_element_area(area_coarse_points_coords[point], area_points_coords[element[1]], area_points_coords[element[2]])
                    S_2 = calculate_element_area(area_points_coords[element[0]], area_coarse_points_coords[point], area_points_coords[element[2]])
                    S_3 = calculate_element_area(area_points_coords[element[0]], area_points_coords[element[1]], area_coarse_points_coords[point])
                    if S_1 + S_2 + S_3 - S_element < 1e-9:
                        chosen_element = element
                        break
                
                function = calculate_local_functions(chosen_element, area_points_coords, dimTask)
                for i in range(3):
                    temp = lambda point: ratioPoints_GlobalLocal_coarse[point] * dimTask
                    temp_2 = lambda point: ratioPoints_GlobalLocal[point] * dimTask
                    F_coarse[temp(point)] += function(area_coarse_points_coords[point], i) * residual[temp_2(chosen_element[i])]
                    F_coarse[temp(point) + 1] += function(area_coarse_points_coords[point], i) * residual[temp_2(chosen_element[i]) + 1]
            

            u_coarse = np.array(func(K_coarse.tocsr(), F_coarse))

            chosen_element = []
            for point in subd_points[idv]:
                for element in subd_coarse_elements[idv]:
                    S_element = calculate_element_area(area_coarse_points_coords[element[0]], area_coarse_points_coords[element[1]], area_coarse_points_coords[element[2]])
                    S_1 = calculate_element_area(area_points_coords[point], area_coarse_points_coords[element[1]], area_coarse_points_coords[element[2]])
                    S_2 = calculate_element_area(area_coarse_points_coords[element[0]], area_points_coords[point], area_coarse_points_coords[element[2]])
                    S_3 = calculate_element_area(area_coarse_points_coords[element[0]], area_coarse_points_coords[element[1]], area_points_coords[point])
                    if S_1 + S_2 + S_3 - S_element < 1e-9:
                        chosen_element = element
                        break
                function = calculate_local_functions(chosen_element, area_coarse_points_coords, dimTask)
                u_special[point * dimTask] = 0
                u_special[point * dimTask + 1] = 0
                for i in range(3):
                    temp = lambda point: ratioPoints_GlobalLocal_coarse[point] * dimTask
                    u_special[point * dimTask] += function(area_points_coords[point], i) * u_coarse[temp(chosen_element[i])]
                    u_special[point * dimTask + 1] += function(area_points_coords[point], i) * u_coarse[temp(chosen_element[i]) + 1]

            [*arg,] = func(K.tocsr(), F)
            u_subd = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

            for x in list(ratioPoints_LocalGlobal.keys()):
                u_current_temp[ratioPoints_LocalGlobal[x], :] = np.copy(u_subd[x, :])

            u_sum += (u_current_temp - u_previous)

        amnt_iterations += 1

        u_current = np.copy(u_previous) + (coef_alpha * u_sum) + (coef_alpha * u_special.reshape(-1, 2))
        crit_convergence = calculate_crit_convergence(u_current, u_previous, area_points_coords, dimTask, relation_points_elements, coef_u)
        print(f"TwoLevel CritConvergence = {crit_convergence}", end = "\r")
        if amnt_iterations > 0:
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