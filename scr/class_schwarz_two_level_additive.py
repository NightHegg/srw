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
        return np.isclose(S_1 + S_2 + S_3, S_element)
    return temp


def caution_check_point_in_elements(point_coords):
    def temp(element, area_points_coords):
        S_element = base_func.calculate_element_area(area_points_coords[element[0]], area_points_coords[element[1]], area_points_coords[element[2]])
        S_1 = base_func.calculate_element_area(point_coords, area_points_coords[element[1]], area_points_coords[element[2]])
        S_2 = base_func.calculate_element_area(area_points_coords[element[0]], point_coords, area_points_coords[element[2]])
        S_3 = base_func.calculate_element_area(area_points_coords[element[0]], area_points_coords[element[1]], point_coords)
        return S_1 + S_2 + S_3 - S_element
    return temp


class schwarz_two_level_additive(schwarz_additive):
    def __init__(self, data):
        init_time = time.time()
        super().__init__(data)

        self.name_method = "schwarz_additive_two_level"
        
        # coarse_area = "simplified_cylinder"
        coarse_area = data["area"]
        coarse_mesh = meshio.read(f'data/meshes/coarse/{coarse_area}/{data["coarse_mesh"]:.3e}.dat')

        self.area_coarse_points_coords = coarse_mesh.points
        self.area_coarse_points = np.array([num for num, _ in enumerate(self.area_coarse_points_coords)])
        self.area_coarse_elements = coarse_mesh.cells_dict["triangle"]

        self.dict_area_coarse_dirichlet_points = {}
        self.dict_area_coarse_neumann_points = {}

        for ind, row in enumerate(self.dirichlet_conditions):
            if ind in [0, 1]:
                radius, displacement = row[0], row[1]
                for point in self.area_coarse_points:
                    point_coords = np.array(self.area_coarse_points_coords[point])
                    if np.isclose(np.linalg.norm(point_coords), radius):
                        self.dict_area_coarse_dirichlet_points[point] = point_coords * (displacement / radius)
            else:
                a, b = np.array(self.contour_points[row[0]]), np.array(self.contour_points[row[1]])
                for point in self.area_coarse_points:
                    point_coords = np.array(self.area_coarse_points_coords[point])
                    if (abs(np.cross(b-a, point_coords - a)) < 1e-15 and np.dot(b-a, point_coords - a) >= 0 and np.dot(b-a, point_coords - a) < np.linalg.norm(a-b)) or np.allclose(point_coords, a) or np.allclose(point_coords, b):
                        if point in self.dict_area_coarse_dirichlet_points:
                            if math.isnan(self.dict_area_coarse_dirichlet_points[point][0]) and math.isnan(self.dict_area_coarse_dirichlet_points[point][1]):
                                self.dict_area_coarse_dirichlet_points[point] = [row[2], row[3]]
                            else:
                                if math.isnan(row[2]):
                                    self.dict_area_coarse_dirichlet_points[point] = [self.dict_area_coarse_dirichlet_points[point][0], row[3]]
                                elif math.isnan(row[3]):
                                    self.dict_area_coarse_dirichlet_points[point] = [row[2], self.dict_area_coarse_dirichlet_points[point][1]]
                        else:
                            self.dict_area_coarse_dirichlet_points[point] = [row[2], row[3]]
        for row in self.neumann_conditions:
            radius, pressure = row[0], row[1]
            for point in self.area_coarse_points:
                point_coords = np.array(self.area_coarse_points_coords[point])
                if np.isclose(np.linalg.norm(point_coords), radius):
                    self.dict_area_coarse_neumann_points[point] = pressure

        self.list_area_coarse_neumann_elements = []
        for index_element, element in enumerate(self.area_coarse_elements):
            if len(set(element) & set(self.dict_area_coarse_neumann_points.keys())) == 2:
                self.list_area_coarse_neumann_elements.append(index_element)

        self.K_special = base_func.calculate_sparse_matrix_stiffness(self.area_elements, self.area_points_coords, self.area_points.size, self.D, self.dim_task)
        self.F_special = np.zeros(self.area_points_coords.size)

        self.set_condition_neumann(self.F_special, self.list_area_neumann_elements, self.area_points_coords, self.dict_area_neumann_points)
        self.set_condition_dirichlet(self.K_special, self.F_special, self.dict_area_dirichlet_points, self.dict_area_dirichlet_points.keys())

        self.K_coarse_special = base_func.calculate_sparse_matrix_stiffness(self.area_coarse_elements, self.area_coarse_points_coords, self.area_coarse_points.size, self.D, self.dim_task)
        self.F_coarse_special = np.zeros(self.area_coarse_points_coords.size)

        self.barycentric_coords_for_coarse_elements = {}
        for num_element, coarse_element in enumerate(self.area_coarse_elements):
            self.barycentric_coords_for_coarse_elements[num_element] = base_func.create_barycentric_coords(coarse_element, self.area_coarse_points_coords)

        self.dict_point_in_coarse_elements = {}
        for num_point, point_coords in enumerate(self.area_points_coords):
            bool_check = check_point_in_elements(point_coords)
            bool_element = False
            for num_element, coarse_element in enumerate(self.area_coarse_elements):
                bool_element = bool_check(coarse_element, self.area_coarse_points_coords)
                if bool_element:
                    self.dict_point_in_coarse_elements[num_point] = num_element
                    break
            if not bool_element:
                caution_check = caution_check_point_in_elements(point_coords)
                dict_caution = {caution_check(elem, self.area_coarse_points_coords): num for num, elem in enumerate(self.area_coarse_elements)}
                self.dict_point_in_coarse_elements[num_point] = dict_caution[min(dict_caution)]
        # test_dict = {}
        # for key, value in self.dict_point_in_coarse_elements.items():
        #     if value in test_dict.keys():
        #         test_dict[value].append(key)
        #     else:
        #         test_dict[value] = [key]
        # for key, value in test_dict.items():
        #     self.internal_plot_displacements_coarse(self.area_points_coords[value], self.area_coarse_points_coords, self.area_coarse_elements)
        self.time_init = time.time() - init_time


    def get_displacements(self):
        u_special = np.ravel(np.zeros_like(self.u))

        F_previous = self.K_special.dot(np.ravel(self.u_previous))
        F_residual = self.F_special.copy() - F_previous

        K_coarse = self.K_coarse_special.copy()
        F_coarse = self.F_coarse_special.copy()

        for point, element in self.dict_point_in_coarse_elements.items():
            point_coords = self.area_points_coords[point]
            [a, b, c], A = self.barycentric_coords_for_coarse_elements[element]
            value = (a + b * point_coords[0] + c * point_coords[1]) / 2 / A

            F_coarse[self.area_coarse_elements[element] * self.dim_task] += np.dot(value, F_residual[point * self.dim_task])
            F_coarse[self.area_coarse_elements[element] * self.dim_task + 1] += np.dot(value, F_residual[point * self.dim_task + 1])

        self.set_condition_dirichlet(K_coarse, F_coarse, self.dict_area_coarse_dirichlet_points, self.dict_area_coarse_dirichlet_points.keys())
        [*arg,] = self.solve_function(K_coarse.tocsr(), F_coarse)
        u_coarse = np.ravel(np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2)))

        for point, element in self.dict_point_in_coarse_elements.items():
            point_coords = self.area_points_coords[point]
            [a, b, c], A = self.barycentric_coords_for_coarse_elements[element]
            value = (a + b * point_coords[0] + c * point_coords[1]) / 2 / A

            u_special[point * self.dim_task] += np.dot(value, u_coarse[self.area_coarse_elements[element] * self.dim_task])
            u_special[point * self.dim_task + 1] += np.dot(value, u_coarse[self.area_coarse_elements[element] * self.dim_task + 1])

        self.u = self.u_previous.copy() + (self.coef_alpha * self.u_sum.copy()) + (self.coef_alpha * u_special.reshape(-1, 2))

    def plot_init_coarse_mesh(self):
        self.internal_plot_displacements(self.area_coarse_points_coords, self.area_coarse_elements)

if __name__ == "__main__":
    pass
