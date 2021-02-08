import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import linalg
from itertools import combinations

from scr.operations_input_files.read_input_files import read_mesh, read_task
import scr.functions as base_func

class basic_method:

    def __init__(self, cur_task, cur_mesh, solve_function = linalg.spsolve):
        self.name_method = "basic method"

        self.solve_function = solve_function
        self.area_bounds, self.area_points_coords, self.area_elements = read_mesh(cur_mesh)

        *arg, = read_task(cur_task)

        self.dimTask              = arg[0]
        E                         = arg[1]
        nyu                       = arg[2]
        self.dirichlet_conditions = arg[3]
        self.neumann_conditions        = arg[4]
        self.coef_u               = arg[5]
        self.coef_sigma           = arg[6]
        self.coef_overlap         = arg[7]

        self.D = np.array([[1, nyu/(1 - nyu), 0],
                           [nyu/(1 - nyu), 1, 0], 
                           [0, 0, (1 - 2 * nyu) / 2 / (1 - nyu)]]) * E * (1 - nyu) / (1 - 2 * nyu) / (1 + nyu)

        area_limits = lambda cond, type: [self.area_bounds[cond[0], type], self.area_bounds[cond[0] + 1, type]]

        self.dirichlet_points = {tuple(idx for idx, val in enumerate(self.area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))) : cond[1:] for cond in self.dirichlet_conditions}

        self.neumann_points = {tuple(idx for idx, val in enumerate(self.area_points_coords) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))) : cond[1:] for cond in self.neumann_conditions}


    def calculate_u(self):
        K = base_func.calculate_sparse_matrix_stiffness(self.area_elements, self.area_points_coords, self.D, self.dimTask)
        F = np.zeros(self.area_points_coords.size)
        
        for list_points, condition in self.dirichlet_points.items():
            for point in list_points:
                if condition[0] == 2:
                    K, F = base_func.bound_condition_dirichlet(K, F, self.dimTask, point, condition[1], 0)
                    K, F = base_func.bound_condition_dirichlet(K, F, self.dimTask, point, condition[2], 1)
                else:
                    K, F = base_func.bound_condition_dirichlet(K, F, self.dimTask, point, condition[1], condition[0])
    
        for list_points, stress in self.neumann_points.items():
            list_elements = [element for element in self.area_elements for x in combinations(list_points, 2) if all([i in element for i in x])]
            for element in list_elements:
                list_neumann_points = list(set(element) & set(list_points))
                for dim in range(self.dimTask):
                    F = base_func.bound_condition_neumann(F, list_neumann_points, self.dimTask, stress[dim], self.area_points_coords, dim)

        *arg, = self.solve_function(K.tocsr(), F)
        self.u = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))


    def calculate_eps(self):
        temp_array = []

        for element in self.area_elements:
            B, _ = base_func.calculate_local_matrix_stiffness(element, self.area_points_coords, self.dimTask)
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


    def plot_displacements(self, plot_global_mesh = True):
        fig, ax = plt.subplots()

        new_points = lambda dimension: self.u[:, dimension] * self.coef_u + self.area_points_coords[:, dimension]

        ax.triplot(new_points(0), new_points(1), self.area_elements.copy())
        ax.plot(new_points(0), new_points(1), 'o')
        ax.plot(self.area_bounds[:, 0], self.area_bounds[:, 1], color = "brown")

        if plot_global_mesh:
            ax.triplot(self.area_points_coords[:,0], self.area_points_coords[:,1], self.area_elements.copy())

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
        return f"min = {abs(abs(min(self.Sigma[1])) - 2e+7):.2e}, max = {abs(abs(max(self.Sigma[1])) - 2e+7):.2e}"


if __name__ == "__main__":
    pass