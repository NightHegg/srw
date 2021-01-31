import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import sparse

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

def plot_displacements(area_points_coords, area_bounds, area_elements, coef_u):
    def temp(u, plot_global_mesh = True):
        fig, ax = plt.subplots()

        ax.triplot(u[:, 0] * coef_u + area_points_coords[:, 0], u[:,1] * coef_u + area_points_coords[:, 1], area_elements.copy())
        ax.plot(area_points_coords[:,0] + u[:, 0] * coef_u, area_points_coords[:,1] + u[:, 1] * coef_u, 'o')
        ax.plot(area_bounds[:, 0], area_bounds[:, 1], color = "brown")
        if plot_global_mesh:
            ax.triplot(area_points_coords[:,0], area_points_coords[:,1], area_elements.copy())

        fig.set_figwidth(10)
        fig.set_figheight(7)
        fig.set_facecolor('mintcream')

        plt.show()
    return temp

def plot_subd_boundaries(area_bounds, area_points_coords, subd_elements, subd_points_coords, subd_boundary_overlap_points, subd_bounds, overlap_bounds):
    nrows = len(subd_elements) // 2
    fig, ax = plt.subplots(nrows, ncols = 2)
    
    for idx in range(len(subd_elements)):
        ax_spec = ax[idx] if nrows == 1 else ax[idx // 2, idx % 2]
            
        ax_spec.plot(area_bounds[:, 0], area_bounds[:, 1], color = "b")

        L = np.array([area_points_coords[point] for point in subd_boundary_overlap_points[idx]])
        ax_spec.plot(L[:, 0], L[:, 1], marker = "X", markersize = 15, linewidth = 0)

        for i in overlap_bounds[0][1:-1]:
            ax_spec.plot([i, i], [overlap_bounds[1][0], overlap_bounds[1][-1]], "k")
        for j in overlap_bounds[1][1:-1]:
            ax_spec.plot([overlap_bounds[0][0], overlap_bounds[0][-1]], [j, j], "k")

        ax_spec.plot(subd_points_coords[idx][:,0], subd_points_coords[idx][:,1], "o")
        ax_spec.triplot(area_points_coords[:,0], area_points_coords[:,1], subd_elements[idx].copy(), alpha = 0.4)

    fig.set_figwidth(20)
    fig.set_figheight(7)
    fig.set_facecolor('mintcream')

    plt.show()

def calculate_local_functions(element, points, dimTask):
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

def calculate_subd_parameters(area_bounds, area_points_coords, area_elements, coef_overlap):
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
    
    return subd_bounds, overlap_bounds, subd_elements, subd_points, subd_points_coords, subd_boundary_overlap_points, relation_points_elements

def calculate_element_area(p_1, p_2, p_3):
    return abs((p_1[0] - p_3[0]) * (p_2[1] - p_3[1]) - (p_2[0] - p_3[0]) * (p_1[1] - p_3[1])) / 2