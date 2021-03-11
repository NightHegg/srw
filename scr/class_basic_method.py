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
from scipy.spatial import ConvexHull, convex_hull_plot_2d

import scr.functions as base_func

def check(a, b):
    return a

class basic_method:
    def __init__(self, data):
        self.name_method = "basic method"
        self.solve_function = linalg.spsolve
        temp_contour = []

        with open(f'data/{data["area"]}/contour.dat', 'r') as f:
            for _ in range(int(f.readline())):
                temp_contour.append([float(x) for x in f.readline().split()])
            self.dim_task = int(f.readline())
            E, nyu = list(map(float, f.readline().split()))
            self.coef_u, self.coef_sigma = list(map(float, f.readline().split()))

        mesh = meshio.read(f'data/{data["area"]}/meshes/{data["mesh"]}.dat')
        
        self.contour_points = np.append(np.array(temp_contour), [temp_contour[0]], axis = 0)
        # self.contour_points = np.array(temp_contour)
        self.area_points_coords = mesh.points
        self.area_points = [num for num, _ in enumerate(self.area_points_coords)]
        self.area_elements = mesh.cells_dict["triangle"]

        self.D = np.array([[1, nyu/(1 - nyu), 0],
                        [nyu/(1 - nyu), 1, 0], 
                        [0, 0, (1 - 2 * nyu) / 2 / (1 - nyu)]]) * E * (1 - nyu) / (1 - 2 * nyu) / (1 + nyu)
 
        self.dirichlet_conditions = []
        self.neumann_conditions = []
        self.dirichlet_points = {}
        self.neumann_points = {}

        with open(f'data/{data["area"]}/tasks/{data["task"]}.dat', 'r') as f:
            for _ in range(int(f.readline())):
                self.dirichlet_conditions.append([int(val) if idx in [0, 1] else float(val) for idx, val in enumerate(f.readline().split())])
            for _ in range(int(f.readline())):
                self.neumann_conditions.append([int(val) if idx in [0, 1] else float(val) for idx, val in enumerate(f.readline().split())])

        for row in self.dirichlet_conditions:
            a, b = np.array(self.contour_points[row[0]]), np.array(self.contour_points[row[1]])
            for point in self.area_points:
                point_coords = np.array(self.area_points_coords[point])
                if abs(np.cross(b-a, point_coords - a)) < 1e-15 and np.dot(b-a, point_coords - a) >= 0 and np.dot(b-a, point_coords - a) < np.linalg.norm(a-b):
                    if point in self.dirichlet_points:
                        if math.isnan(self.dirichlet_points[point][0]) and not math.isnan(row[2]):
                            self.dirichlet_points[point] = [row[2], self.dirichlet_points[point][1]]
                        else:
                            self.dirichlet_points[point] = [self.dirichlet_points[point][0], row[3]]
                    else:
                        self.dirichlet_points[point] = [row[2], row[3]]
                   
        for row in self.neumann_conditions:
            a, b = np.array(self.contour_points[row[0]]), np.array(self.contour_points[row[1]])
            for point in self.area_points:
                point_coords = np.array(self.area_points_coords[point])
                if abs(np.cross(b-a, point_coords - a)) < 1e-15 and np.dot(b-a, point_coords - a) >= 0 and np.dot(b-a, point_coords - a) < np.linalg.norm(a-b):
                    if point in self.neumann_points:
                        if math.isnan(self.neumann_points[point][0]) and not math.isnan(row[2]):
                            self.neumann_points[point] = [row[2], self.neumann_points[point][1]]
                        else:
                            self.neumann_points[point] = [self.neumann_points[point][0], row[3]]
                    else:
                        self.neumann_points[point] = [row[2], row[3]]


    def calculate_u(self):
        K = base_func.calculate_sparse_matrix_stiffness(self.area_elements, self.area_points_coords, self.D, self.dim_task)
        F = np.zeros(self.area_points_coords.size)

        for point, condition in self.dirichlet_points.items():
            if math.isnan(condition[0]):
                K, F = base_func.bound_condition_dirichlet(K, F, self.dim_task, point, condition[1], 1)
            elif math.isnan(condition[1]):
                K, F = base_func.bound_condition_dirichlet(K, F, self.dim_task, point, condition[0], 0)
            else:
                K, F = base_func.bound_condition_dirichlet(K, F, self.dim_task, point, condition[0], 0)
                K, F = base_func.bound_condition_dirichlet(K, F, self.dim_task, point, condition[1], 1)

        list_neumann_elements = [element for element in self.area_elements for x in combinations(self.neumann_points.keys(), 2) if all([i in element for i in x])]
        for element in list_neumann_elements:
            list_neumann_points = list(set(element) & set(self.neumann_points.keys()))
            dict_neumann_points = {point: self.neumann_points[point] for point in list_neumann_points}
            F = base_func.bound_condition_neumann(F, dict_neumann_points, self.area_points_coords, self.dim_task)

        *arg, = self.solve_function(K.tocsr(), F)
        self.u = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))


    def calculate_eps(self):
        temp_array = []

        for element in self.area_elements:
            B, _ = base_func.calculate_local_matrix_stiffness(element, self.area_points_coords, self.dim_task)
            temp_array.append(np.dot(B, np.ravel(np.array([self.u[element[0]], self.u[element[1]], self.u[element[2]]])).transpose()))

        self.Eps = np.array(temp_array)


    def calculate_sigma(self):
        self.Sigma = self.D @ self.Eps.T


    def get_solution(self):
        init_time = time.time()
        self.calculate_u()
        self.time_execution = time.time() - init_time

        self.calculate_eps()
        self.calculate_sigma()


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


    def get_info(self):
        message = (f"Method: {self.name_method}\n"
                   f"Time of execution: {self.time_execution}\n"
                   f"Minimal difference for stress: {abs(abs(min(self.Sigma[1])) - 2e+7):.2e}\n"
                   f"Maximal difference for stress: {abs(abs(max(self.Sigma[1])) - 2e+7):.2e}")
        return message


    def get_special_sigma(self):
        return f"min = {abs(abs(min(self.Sigma[1])) - 2e+7) / 2e+7:.2e}, max = {abs(abs(max(self.Sigma[1])) - 2e+7) / 2e+7:.2e}"


if __name__ == "__main__":
    pass