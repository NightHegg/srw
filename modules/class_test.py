import scipy
from scipy.sparse import linalg
from scipy.sparse import lil_matrix, coo_matrix
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import os

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

    return lil_matrix(coo_matrix((data, (row, col)), shape = (area_points_coords.size, area_points_coords.size)))              

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

def read_task(cur_task):
    dirichlet_conditions = []
    neumann_conditions = []

    with open(f"modules/creation_input_files/input_files/task_{cur_task}.dat") as f:
        dimTask = int(f.readline())
        E, nyu = list(map(float, f.readline().split()))
        for _ in range(int(f.readline())):
            dirichlet_conditions.append([int(val) if idx in [0, 1] else float(val) for idx, val in enumerate(f.readline().split())])
        for _ in range(int(f.readline())):
            neumann_conditions.append([int(val) if idx == 0 else float(val) for idx, val in enumerate(f.readline().split())])
        coef_u, coef_sigma = list(map(float, f.readline().split()))
        coef_overlap = float(f.readline())

    return dimTask, E, nyu, dirichlet_conditions, neumann_conditions, coef_u, coef_sigma, coef_overlap

def read_mesh(cur_mesh):
    bounds = []
    area_points_coords = []
    area_elements = []

    with open(f"modules/creation_input_files/input_files/mesh/mesh_{cur_mesh}.dat") as f:
        for _ in range(int(f.readline())):
            bounds.append([float(x) for x in f.readline().split()])
        for _ in range(int(f.readline())):
            area_points_coords.append([float(val) for val in f.readline().split()])
        for _ in range(int(f.readline())):
            area_elements.append([int(val) for val in f.readline().split()])
    
    return np.array(bounds), np.array(area_points_coords), area_elements

class Test:
    def __init__(self, cur_task, cur_mesh, solve_function = linalg.spsolve):
        
        self.__solve_function = solve_function

        self.__area_bounds, self.__area_points_coords, self.__area_elements  = read_mesh(cur_mesh)

        *arg, = read_task(cur_task)

        self.__dimTask       = arg[0]
        E                    = arg[1]
        nyu                  = arg[2]
        dirichlet_conditions = arg[3]
        neumann_conditions   = arg[4]
        self.__coef_u        = arg[5]
        self.__coef_sigma    = arg[6]
        self.coef_overlap    = arg[7]

        self.D = np.array([[1, nyu/(1 - nyu), 0],
                           [nyu/(1 - nyu), 1, 0], 
                           [0, 0, (1 - 2 * nyu) / 2 / (1 - nyu)]]) * E * (1 - nyu) / (1 - 2 * nyu) / (1 + nyu)

        area_limits = lambda cond, type: [self.__area_bounds[cond[0], type], self.__area_bounds[cond[0] + 1, type]]

        self.__dirichlet_points = [[[idx for idx, val in enumerate(self.__area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))], cond[1:]] for cond in dirichlet_conditions]

        self.__neumann_points = [[[idx for idx, val in enumerate(self.__area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))], cond[1], cond[2]] for cond in neumann_conditions]


    def calculate_u(self):
        F = np.zeros(self.__area_points_coords.size)
        K = calculate_sparse_matrix_stiffness(self.__area_elements, self.__area_points_coords, self.D, self.__dimTask)

        for condition in self.__dirichlet_points:
            for node in condition[0]:
                if condition[1][0] == 2:
                    K, F = bound_condition_dirichlet(K, F, self.__dimTask, node, condition[1][1], 0)
                    K, F = bound_condition_dirichlet(K, F, self.__dimTask, node, condition[1][2], 1)
                else:
                    K, F = bound_condition_dirichlet(K, F, self.__dimTask, node, condition[1][1], condition[1][0])
    
            for condition in self.__neumann_points:
                for element in [element for element in self.__area_elements for x in list(combinations(condition[0], 2)) if x[0] in element and x[1] in element]:
                    F = bound_condition_neumann(F, element, condition[0], self.__dimTask, condition[1], self.__area_points_coords, 0)
                    F = bound_condition_neumann(F, element, condition[0], self.__dimTask, condition[2], self.__area_points_coords, 1)  

        *arg, = self.__solve_function(K.tocsr(), F)

        self.u = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))


    def calculate_eps(self):
        temp_array = []

        for element in self.__area_elements:
            B, _ = calculate_local_matrix_stiffness(element, self.__area_points_coords, self.__dimTask)
            temp_array.append(np.dot(B, np.ravel(np.array([self.u[element[0]], self.u[element[1]], self.u[element[2]]])).transpose()))

        self.Eps = np.array(temp_array)

    def calculate_sigma(self):
        self.Sigma = np.dot(self.D, self.Eps.transpose())

    def get_solution(self):
        self.calculate_u()
        self.calculate_eps()
        self.calculate_sigma()

    def plot_displacements(self):
        plot_displacements(self.__area_points_coords,self.__area_bounds, self.__area_elements, self.__coef_u)(self.u)