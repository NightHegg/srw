import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import linalg
from itertools import combinations
import meshio
import math
import scipy

import scr.functions as base_func


class basic_method:
    def __init__(self, data):
        init_time = time.time()
        self.name_method = "basic method"
        self.message = {}
        self.data = data

        self.solve_function = linalg.spsolve
        temp_contour = []

        with open(f'data/{self.data["area"]}/contour.dat', 'r') as f:
            for _ in range(int(f.readline())):
                temp_contour.append([float(x) for x in f.readline().split()])
            self.dim_task = int(f.readline())
            E, nyu = list(map(float, f.readline().split()))
            self.coef_u, self.coef_sigma = list(map(float, f.readline().split()))

        mesh = meshio.read(f'data/{self.data["area"]}/meshes/{self.data["mesh"]}.dat')
        
        self.contour_points = np.append(np.array(temp_contour), [temp_contour[0]], axis = 0)
        self.area_points_coords = mesh.points
        self.area_points = [num for num, _ in enumerate(self.area_points_coords)]
        self.area_elements = mesh.cells_dict["triangle"]

        self.D = np.array([[1, nyu/(1 - nyu), 0],
                        [nyu/(1 - nyu), 1, 0], 
                        [0, 0, (1 - 2 * nyu) / 2 / (1 - nyu)]]) * E * (1 - nyu) / (1 - 2 * nyu) / (1 + nyu)
 
        self.dirichlet_conditions = []
        self.neumann_conditions = []
        self.dict_area_dirichlet_points = {}
        self.dict_area_neumann_points = {}

        with open(f'data/{self.data["area"]}/tasks/{self.data["task"]}.dat', 'r') as f:
            for _ in range(int(f.readline())):
                self.dirichlet_conditions.append([int(val) if idx in [0, 1] else float(val) for idx, val in enumerate(f.readline().split())])
            for _ in range(int(f.readline())):
                self.neumann_conditions.append([int(val) if idx in [0, 1] else float(val) for idx, val in enumerate(f.readline().split())])

        for row in self.dirichlet_conditions:
            a, b = np.array(self.contour_points[row[0]]), np.array(self.contour_points[row[1]])
            for point in self.area_points:
                point_coords = np.array(self.area_points_coords[point])
                if abs(np.cross(b-a, point_coords - a)) < 1e-15 and np.dot(b-a, point_coords - a) >= 0 and np.dot(b-a, point_coords - a) < np.linalg.norm(a-b):
                    if point in self.dict_area_dirichlet_points:
                        if math.isnan(self.dict_area_dirichlet_points[point][0]) and not math.isnan(row[2]):
                            self.dict_area_dirichlet_points[point] = [row[2], self.dict_area_dirichlet_points[point][1]]
                        else:
                            self.dict_area_dirichlet_points[point] = [self.dict_area_dirichlet_points[point][0], row[3]]
                    else:
                        self.dict_area_dirichlet_points[point] = [row[2], row[3]]
                   
        for row in self.neumann_conditions:
            a, b = np.array(self.contour_points[row[0]]), np.array(self.contour_points[row[1]])
            for point in self.area_points:
                point_coords = np.array(self.area_points_coords[point])
                if abs(np.cross(b-a, point_coords - a)) < 1e-15 and np.dot(b-a, point_coords - a) >= 0 and np.dot(b-a, point_coords - a) < np.linalg.norm(a-b):
                    if point in self.dict_area_neumann_points:
                        if math.isnan(self.dict_area_neumann_points[point][0]) and not math.isnan(row[2]):
                            self.dict_area_neumann_points[point] = [row[2], self.dict_area_neumann_points[point][1]]
                        else:
                            self.dict_area_neumann_points[point] = [self.dict_area_neumann_points[point][0], row[3]]
                    else:
                        self.dict_area_neumann_points[point] = [row[2], row[3]]
        
        self.list_area_neumann_elements = []
        for element in self.area_elements:
            if len(set(element) & set(self.dict_area_neumann_points.keys())) == 2:
                self.list_area_neumann_elements.append(element)
        self.time_init = time.time() - init_time


    def set_condition_dirichlet(self, K, F, list_dirichlet_points, modifier = {}):
        for point in list_dirichlet_points:
            for idx, cur_condition in enumerate(self.dict_area_dirichlet_points[point]):
                modified_point = modifier[point] if modifier else point
                if not math.isnan(cur_condition):
                    indices = np.array(K.rows[modified_point * self.dim_task + idx])
                    F[indices] -= np.array(K.data[modified_point * self.dim_task + idx]) * cur_condition
                    for index in indices:
                        K[modified_point * self.dim_task + idx, index] = 0
                    for index in indices:
                        K[index, modified_point * self.dim_task + idx] = 0
                    K[modified_point * self.dim_task + idx, modified_point * self.dim_task + idx] = 1
                    F[modified_point * self.dim_task + idx] = cur_condition


    def set_condition_neumann(self, F, list_neumann_points, modifier = {}):
        for element in list_neumann_points:
            points = list(set(element) & set(self.dict_area_neumann_points.keys()))
            points_coords = [self.area_points_coords[point] for point in points]
            len = np.linalg.norm(np.array(points_coords[1]) - np.array(points_coords[0]))
            for point in points:
                for idx, cur_condition in enumerate(self.dict_area_neumann_points[point]):
                    modified_point = modifier[point] if modifier else point
                    if not math.isnan(cur_condition):
                        F[modified_point * self.dim_task + idx] += cur_condition * len / 2

        
    def calculate_u(self):
        self.K = base_func.calculate_sparse_matrix_stiffness(self.area_elements, self.area_points_coords, self.D, self.dim_task)
        self.F = np.zeros(self.area_points_coords.size)

        self.set_condition_dirichlet(self.K, self.F, self.dict_area_dirichlet_points.keys())
        self.set_condition_neumann(self.F, self.list_area_neumann_elements)

        *arg, = self.solve_function(self.K.tocsr(), self.F)
        self.u = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))


    def calculate_eps(self):
        temp_array = []

        for element in self.area_elements:
            B, _ = base_func.calculate_local_matrix_stiffness(element, self.area_points_coords, self.dim_task)
            # temp_array.append(np.dot(B, np.ravel(np.array([self.u[element[0]], self.u[element[1]], self.u[element[2]]])).T))
            temp_array.append(np.dot(B, np.ravel([self.u[element[idx]] for idx in range(len(element))]).T))

        self.Eps = np.array(temp_array)


    def calculate_sigma(self):
        self.Sigma = self.D @ self.Eps.T


    def get_solution(self):
        init_global = time.time()
        init_time = time.time()
        self.calculate_u()
        self.time_u = time.time() - init_time

        init_time = time.time()
        self.calculate_eps()
        self.calculate_sigma()
        self.time_eps_sigma = time.time() - init_time
        self.time_getting_solution = time.time() - init_global
        self.time_global = self.time_getting_solution + self.time_init

    def internal_plot_displacements(self, vector_u, area_points_coords, area_elements, plot_global_mesh = True):
        fig, ax = plt.subplots()
        
        new_points = lambda dimension: vector_u[:, dimension] * self.coef_u + area_points_coords[:, dimension]

        ax.triplot(new_points(0), new_points(1), area_elements.copy())
        ax.plot(new_points(0), new_points(1), 'o')
        ax.plot(self.contour_points[:, 0], self.contour_points[:, 1], color = "brown")

        if plot_global_mesh:
            ax.triplot(area_points_coords[:,0], area_points_coords[:,1], area_elements.copy())

        fig.set_figwidth(10)
        fig.set_figheight(7)
        fig.set_facecolor('mintcream')

        plt.show()


    def plot_displacements(self, plot_global_mesh = True):
        self.internal_plot_displacements(self.u, self.area_points_coords, self.area_elements, plot_global_mesh)


    def plot_init_mesh(self):
        fig, ax = plt.subplots()

        ax.triplot(self.area_points_coords[:, 0], self.area_points_coords[:, 1], self.area_elements.copy())
        ax.plot(self.contour_points[:, 0], self.contour_points[:, 1], color = "brown")

        fig.set_figwidth(10)
        fig.set_figheight(7)
        fig.set_facecolor('mintcream')

        plt.show()

    
    def construct_info(self):
        self.message['main_info'] = [
            f' area: {self.data["area"]}\n',
            f'task: {self.data["task"]}\n',
            f'mesh: {self.data["mesh"]}\n',
            f'method: {self.name_method}\n',
            f'{"-" * 5}\n'
        ]
        self.message['time'] = [
            f'Global time: {self.time_global:.2f}\n',
            f'Time of initialization: {self.time_init:.2f} ({self.time_init / self.time_global:.2%})\n',
            f'Time of calculation displacements: {self.time_u} ({self.time_u / self.time_global:.2%})\n',
            f'Time of calculation eps+sigma: {self.time_eps_sigma} ({self.time_eps_sigma / self.time_global:.2%})\n',
            f'{"-" * 5}\n'
        ]
        self.message['diff_stress'] = [
            f'Minimal difference for stress: {abs(abs(min(self.Sigma[1])) - 2e+7) / 2e+7:.2e}\n',
            f'Maximal difference for stress: {abs(abs(max(self.Sigma[1])) - 2e+7) / 2e+7:.2e}\n',
            f'{"-" * 5}\n'
        ]

    def get_info(self):
        self.construct_info()
        return sum(list(self.message.values()), [])


    def get_special_sigma(self):
        return f"min = {abs(abs(min(self.Sigma[1])) - 2e+7) / 2e+7:.2e}, max = {abs(abs(max(self.Sigma[1])) - 2e+7) / 2e+7:.2e}"


if __name__ == "__main__":
    pass