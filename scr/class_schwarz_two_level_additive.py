import os
import sys
import math

from scipy.sparse.linalg import dsolve
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from scipy.sparse import linalg
import meshio
import time

import scr.functions as base_func
from scr.class_schwarz_additive import schwarz_additive


def check_point_in_elements(point_coords):
    def temp(element, area_points_coords):
        S_element = base_func.calculate_element_area(area_points_coords[element[0]], area_points_coords[element[1]], area_points_coords[element[2]])
        S_1 = base_func.calculate_element_area(point_coords, area_points_coords[element[1]], area_points_coords[element[2]])
        S_2 = base_func.calculate_element_area(area_points_coords[element[0]], point_coords, area_points_coords[element[2]])
        S_3 = base_func.calculate_element_area(area_points_coords[element[0]], area_points_coords[element[1]], point_coords)
        return S_1 + S_2 + S_3 - S_element < 1e-9
    return temp


class schwarz_two_level_additive(schwarz_additive):
    def __init__(self, data):
        init_time = time.time()
        super().__init__(data)

        self.name_method = "schwarz additive two level method"

        coarse_mesh = meshio.read(f'data/{data["area"]}/meshes/{data["coarse_mesh"]}.dat')

        self.area_coarse_points_coords = coarse_mesh.points
        self.area_coarse_points = [num for num, _ in enumerate(self.area_coarse_points_coords)]
        self.area_coarse_elements = coarse_mesh.cells_dict["triangle"]

        self.dirichlet_coarse_points = {}

        for row in self.dirichlet_conditions:
            a, b = np.array(self.contour_points[row[0]]), np.array(self.contour_points[row[1]])
            for point in self.area_coarse_points:
                point_coords = np.array(self.area_coarse_points_coords[point])
                if abs(np.cross(b-a, point_coords - a)) < 1e-15 and np.dot(b-a, point_coords - a) >= 0 and np.dot(b-a, point_coords - a) < np.linalg.norm(a-b):
                    if point in self.dirichlet_coarse_points:
                        if math.isnan(self.dirichlet_coarse_points[point][0]) and not math.isnan(row[2]):
                            self.dirichlet_coarse_points[point] = [row[2], self.dirichlet_coarse_points[point][1]]
                        else:
                            self.dirichlet_coarse_points[point] = [self.dirichlet_coarse_points[point][0], row[3]]
                    else:
                        self.dirichlet_coarse_points[point] = [row[2], row[3]]

        self.K_special = base_func.calculate_sparse_matrix_stiffness(self.area_elements, self.area_points_coords, self.D, self.dim_task)
        self.F_special = np.zeros(self.area_points_coords.size)

        self.set_condition_dirichlet(self.K_special, self.F_special, self.dirichlet_points)
        self.set_condition_neumann(self.F_special)

        self.K_coarse_special = base_func.calculate_sparse_matrix_stiffness(self.area_coarse_elements, self.area_coarse_points_coords, self.D, self.dim_task)
        self.F_coarse_special = np.zeros(self.area_coarse_points_coords.size)

        self.barycentric_coords_for_coarse_elements = {}
        for num_element, coarse_element in enumerate(self.area_coarse_elements):
            self.barycentric_coords_for_coarse_elements[num_element] = base_func.create_barycentric_coords(coarse_element, self.area_coarse_points_coords)

        self.dict_point_in_coarse_elements = {}
        for num_point, point_coords in enumerate(self.area_points_coords):
            bool_check = check_point_in_elements(point_coords)
            for num_element, coarse_element in enumerate(self.area_coarse_elements):
                if bool_check(coarse_element, self.area_coarse_points_coords):
                    self.dict_point_in_coarse_elements[num_point] = num_element
                    break
        self.time_init = time.time() - init_time


    def interal_final_calculate_u(self):
        u_special = np.ravel(np.zeros_like(self.u))

        F_test = self.K_special.dot(np.ravel(self.u_previous))
        residual = self.F_special - F_test

        # init_time = time.time()
        K_coarse = self.K_coarse_special.copy()
        F_coarse = self.F_coarse_special.copy()
        # self.time_final_1 += time.time() - init_time

        # init_time = time.time()
        for point, element in self.dict_point_in_coarse_elements.items():
            point_coords = self.area_points_coords[point]

            barycentric_data = self.barycentric_coords_for_coarse_elements[element]
            bar_coords = barycentric_data['barycentric_coords']
            A = barycentric_data['area_of_element']

            for i in range(3):
                value = (bar_coords[0, i] + bar_coords[1, i] * point_coords[0] + bar_coords[2, i] * point_coords[1]) / 2 / A
                F_coarse[self.area_coarse_elements[element][i] * self.dim_task : (self.area_coarse_elements[element][i] + 1) * self.dim_task] += value * residual[point * self.dim_task : (point + 1) * self.dim_task]
        # self.time_final_2 += time.time() - init_time

        # init_time = time.time()
        self.set_condition_dirichlet(K_coarse, F_coarse, self.dirichlet_coarse_points)
        [*arg,] = self.solve_function(K_coarse.tocsr(), F_coarse)
        u_coarse = np.ravel(np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2)))
        # self.time_final_3 += time.time() - init_time

        # init_time = time.time()
        for point, element in self.dict_point_in_coarse_elements.items():
            point_coords = self.area_points_coords[point]

            barycentric_data = self.barycentric_coords_for_coarse_elements[element]
            bar_coords = barycentric_data['barycentric_coords']
            A = barycentric_data['area_of_element']

            for i in range(3):
                value = (bar_coords[0, i] + bar_coords[1, i] * point_coords[0] + bar_coords[2, i] * point_coords[1]) / 2 / A
                u_special[point * self.dim_task: (point + 1) * self.dim_task] += value * u_coarse[self.area_coarse_elements[element][i] * self.dim_task : (self.area_coarse_elements[element][i] + 1) * self.dim_task]

        self.u = np.copy(self.u_previous) + (self.coef_alpha * self.u_sum) + (self.coef_alpha * u_special.reshape(-1, 2))
        # self.time_final_4 += time.time() - init_time


    def plot_init_mesh(self):
        fig, ax = plt.subplots()

        ax.triplot(self.area_points_coords[:, 0], self.area_points_coords[:, 1], self.area_elements.copy())

        ax.triplot(self.area_coarse_points_coords[:, 0], self.area_coarse_points_coords[:, 1], self.area_coarse_elements.copy())
        ax.plot(self.area_coarse_points_coords[:, 0], self.area_coarse_points_coords[:, 1], 'o', markersize = 10)

        ax.plot(self.contour_points[:, 0], self.contour_points[:, 1], color = "brown")

        fig.set_figwidth(10)
        fig.set_figheight(7)
        fig.set_facecolor('mintcream')

        plt.show()
        

if __name__ == "__main__":
    pass
