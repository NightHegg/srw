import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from scipy.sparse import linalg
import meshio
import math

import scr.functions as base_func
import scr._class_visualisation as class_visual


class class_template(class_visual.class_visualisation):
    def __init__(self, data):
        self.message = {}
        self.data = data

        self.solve_function = linalg.spsolve
        temp_contour = []

        with open(f'data/{self.data["fine_area"]}/area_info.dat', 'r') as f:
            for _ in range(int(f.readline())):
                temp_contour.append([float(x) for x in f.readline().split()])
            self.dim_task = int(f.readline())
            self.E, self.nyu = list(map(float, f.readline().split()))
            self.coef_u, self.coef_sigma = list(map(float, f.readline().split()))

        mesh = meshio.read(f'data/{self.data["fine_area"]}/meshes_fine/{self.data["fine_mesh"]:.3e}.msh')

        self.contour_points = np.append(np.array(temp_contour), [temp_contour[0]], axis = 0)
        self.area_points_coords = mesh.points
        self.area_points_coords = np.delete(self.area_points_coords, -1, axis = 1)
        self.area_points = np.array([num for num, _ in enumerate(self.area_points_coords)])
        self.area_elements = mesh.cells_dict["triangle"]

        self.D = np.array(
            [
                [1, self.nyu/(1 - self.nyu), 0],
                [self.nyu/(1 - self.nyu), 1, 0], 
                [0, 0, (1 - 2 * self.nyu) * 2 / (1 - self.nyu)]
            ]
        ) * self.E * (1 - self.nyu) / ((1 - 2 * self.nyu) * (1 + self.nyu))
 
        self.dirichlet_conditions = []
        self.neumann_conditions = []
        self.dict_area_dirichlet_points = {}
        self.dict_area_neumann_points = {}

        with open(f'data/{self.data["fine_area"]}/tasks/{self.data["task"]}.dat', 'r') as f:
            if self.data["fine_area"] == 'rectangle':
                for _ in range(int(f.readline())):
                    self.dirichlet_conditions.append(np.array([int(val) if ind in [0, 1] else float(val) for ind, val in enumerate(f.readline().split())]))
                for _ in range(int(f.readline())):
                    self.neumann_conditions.append(np.array([int(val) if ind in [0, 1] else float(val) for ind, val in enumerate(f.readline().split())]))
                self.table_pressure = self.neumann_conditions[-1][-1]
            else:
                self.inner_radius = self.contour_points[0, 0]
                self.outer_radius = self.contour_points[1, 0]

                self.inner_displacement = float(f.readline())
                self.outer_displacement = float(f.readline())
                self.dirichlet_conditions.append([self.inner_radius, self.inner_displacement])
                self.dirichlet_conditions.append([self.outer_radius, self.outer_displacement])

                for _ in range(int(f.readline())):
                    self.dirichlet_conditions.append(np.array([int(val) if ind in [0, 1] else float(val) for ind, val in enumerate(f.readline().split())]))

                self.inner_pressure = float(f.readline())
                self.outer_pressure = float(f.readline())
                self.neumann_conditions.append(np.array([self.inner_radius, self.inner_pressure]))
                self.neumann_conditions.append(np.array([self.outer_radius, self.outer_pressure]))

                self.inner_radius_points = np.isclose(self.inner_radius, np.linalg.norm(self.area_points_coords, axis = 1))
                self.outer_radius_points = np.isclose(self.outer_radius, np.linalg.norm(self.area_points_coords, axis = 1))

        if self.data["fine_area"] == 'rectangle':
            for point_num, point_coords in enumerate(self.area_points_coords):
                for row in self.dirichlet_conditions:
                    contour_points = self.contour_points[row[:2].astype(int)]
                    if np.isclose(abs(np.cross(np.diff(contour_points, axis = 0), point_coords - contour_points[0])), 0):
                        if point_num in self.dict_area_dirichlet_points:
                            template_nan = np.isnan(self.dict_area_dirichlet_points[point_num])
                            self.dict_area_dirichlet_points[point_num][template_nan] = np.copy(row)[2:][template_nan]
                        else:
                            self.dict_area_dirichlet_points[point_num] = np.copy(row)[2:]
                
                for row in self.neumann_conditions:
                    contour_points = self.contour_points[row[:2].astype(int)]
                    if np.isclose(abs(np.cross(np.diff(contour_points, axis = 0), point_coords - contour_points[0])), 0):
                        self.dict_area_neumann_points[point_num] = row[2:]
        else:
            for point_num, point_coords in enumerate(self.area_points_coords):
                for index, row in enumerate(self.dirichlet_conditions):
                    if index in [0, 1]:
                        radius, displacement = row[0], row[1]
                        if np.isclose(np.linalg.norm(point_coords), radius):
                            self.dict_area_dirichlet_points[point_num] = point_coords * (displacement / radius)
                    else:
                        contour_points = self.contour_points[row[:2].astype(int)]
                        if np.isclose(abs(np.cross(np.diff(contour_points, axis = 0), point_coords - contour_points[0])), 0):
                            if point_num in self.dict_area_dirichlet_points:
                                template_nan = np.isnan(self.dict_area_dirichlet_points[point_num])
                                self.dict_area_dirichlet_points[point_num][template_nan] = np.copy(row)[2:][template_nan]
                            else:
                                self.dict_area_dirichlet_points[point_num] = np.copy(row)[2:]
                for row in self.neumann_conditions:
                    radius, pressure = row[0], row[1]
                    if np.isclose(np.linalg.norm(point_coords), radius):
                        self.dict_area_neumann_points[point_num] = pressure

        self.list_area_neumann_elements = []
        if self.data["fine_area"] == 'rectangle':
            for element in self.area_elements:
                points = list(set(element) & set(list(self.dict_area_neumann_points.keys())))
                if len(points) == 2:
                    self.list_area_neumann_elements.append(points)
        else:
            for element in self.area_elements:
                inner_points = list(set(element) & set(self.area_points[self.inner_radius_points]))
                outer_points = list(set(element) & set(self.area_points[self.outer_radius_points]))
                if len(inner_points) == 2:
                    self.list_area_neumann_elements.append(inner_points)
                elif len(outer_points) == 2:
                    self.list_area_neumann_elements.append(outer_points)

        self.list_area_of_elements = np.array([base_func.calculate_local_matrix_stiffness(i, self.area_points_coords, self.dim_task)[1] for i in self.area_elements])

        self.dict_elements_contain_point = {}
        for element, list_element_points in enumerate(self.area_elements):
            for point in list_element_points:
                if point in self.dict_elements_contain_point.keys():
                    self.dict_elements_contain_point[point] = np.append(self.dict_elements_contain_point[point], element)
                else:
                    self.dict_elements_contain_point[point] = np.array([element])
        self.dict_elements_contain_point = dict(sorted(self.dict_elements_contain_point.items(), key=lambda x: x[0]))

        self.element_centroid_points_coords = np.array(list(map(lambda x: np.mean(self.area_points_coords[x], axis = 0), self.area_elements)))
        self.element_centroid_points_angles = np.array(list(map(lambda x: np.arctan2(x[1], x[0]), self.element_centroid_points_coords)))


    def set_condition_dirichlet(self, K, F, dict, list_dirichlet_points, modifier = {}):
        for point in list_dirichlet_points:
            conditions = dict[point]
            modified_point = modifier[point] if modifier else point
            for cur_dimension, cur_condition in enumerate(conditions):
                if not math.isnan(cur_condition):
                    indices = np.array(K.rows[modified_point * self.dim_task + cur_dimension])
                    F[indices] -= np.array(K.data[modified_point * self.dim_task + cur_dimension]) * cur_condition
                    for index in indices:
                        K[modified_point * self.dim_task + cur_dimension, index] = 0
                    for index in indices:
                        K[index, modified_point * self.dim_task + cur_dimension] = 0
                    K[modified_point * self.dim_task + cur_dimension, modified_point * self.dim_task + cur_dimension] = 1
                    F[modified_point * self.dim_task + cur_dimension] = cur_condition


    def set_condition_neumann(self, F, list_neumann_elements, area_points_coords, dict1, modifier = {}):
        for points in list_neumann_elements:
            points = np.array(points)
            len = np.linalg.norm(np.diff(area_points_coords[points], axis = 0))
            if self.data["fine_area"] == 'rectangle':
                for point in points:
                    modified_point = modifier[point] if modifier else point
                    pressure = dict1[point]
                    F[modified_point * self.dim_task: (modified_point + 1) * self.dim_task] += pressure * len / 2
            else:
                value = np.flip(np.ravel(np.abs((np.diff(area_points_coords[points], axis = 0)))))
                for point in points:
                    modified_point = modifier[point] if modifier else point
                    pressure = dict1[point]
                    F[modified_point * self.dim_task: (modified_point + 1) * self.dim_task] += value * (pressure / len) * len / 2


    def calculate_eps(self):
        temp_array = []
        for element in self.area_elements:
            B, _ = base_func.calculate_local_matrix_stiffness(element, self.area_points_coords, self.dim_task)
            result = np.dot(B, np.ravel(self.u[element]))
            temp_array.append(result)

        self.eps = np.array(temp_array)
        self.eps[:, -1] /= 2


    def calculate_sigma(self):
        self.sigma = np.dot(self.eps, self.D)
        self.sigma_points = np.array(list(map(lambda x: np.mean(self.sigma[x, :], axis = 0), self.dict_elements_contain_point.values())))
        self.sigma_mean = np.mean(self.sigma, axis = 1)


    def calculate_polar_variables(self):
        self.u_polar = np.hstack(
            (
                np.linalg.norm(self.u, axis = 1).reshape(-1, 1) * np.sign(self.u[np.arange(self.u.shape[0]), np.argmax(np.abs(self.u), axis = 1)]),
                np.arctan2(self.area_points_coords_modified[:, 1], self.area_points_coords_modified[:, 0]).reshape(-1, 1)
            )
        )
        A = []
        for idx, value in enumerate(self.area_elements):
            point_coords = np.mean(self.area_points_coords[value], axis = 0)
            phi = np.arctan2(point_coords[1], point_coords[0])
            M = np.array([[np.cos(phi), np.sin(phi)], [np.sin(phi), np.cos(phi)]]) ** 2
            result = np.dot(self.sigma[idx, :-1], M) + np.array([+1, -1]) * self.sigma[idx, -1] * np.sin(2 * phi)
            A.append(result)
        self.sigma_polar = np.array(A)

        A = []
        for idx, value in enumerate(self.area_points_coords):
            phi = np.arctan2(value[1], value[0])
            M = np.array([[np.cos(phi), np.sin(phi)], [np.sin(phi), np.cos(phi)]]) ** 2
            result = np.dot(self.sigma_points[idx, :-1], M) + np.array([+1, -1]) * self.sigma_points[idx, -1] * np.sin(2 * phi)
            A.append(result)
        self.sigma_points_polar = np.array(A)


    def calculate_exact_variables(self):
        r_point = np.linalg.norm(self.area_points_coords, axis = 1)
        r_element = np.linalg.norm(self.element_centroid_points_coords, axis = 1)

        p_1, p_2 = abs(self.inner_pressure), abs(self.outer_pressure)
        a, b = self.inner_radius, self.outer_radius
        A = (p_1*a**2 - p_2*b**2) / (b**2 - a**2)
        B = (p_1-p_2)*(a*b)**2 / (b**2-a**2)
        
        u_exact = (A * (1 + self.nyu) * (1 - 2 * self.nyu) / self.E) * r_point + (B * (1 + self.nyu) / self.E) / r_point
        self.u_modified_exact = u_exact * self.coef_u + r_point

        self.sigma_exact = (A + B * np.outer(1 / r_element**2, np.array([-1, +1])))


    def get_solution(self):
        init_global = time.time()

        init_time = time.time()
        self.calculate_u()
        self.area_points_coords_modified = self.u * self.coef_u + self.area_points_coords
        self.time_u = time.time() - init_time

        init_time = time.time()

        self.calculate_eps()
        self.calculate_sigma()

        # if not self.data['fine_area'] == 'rectangle':
        #     self.calculate_exact_variables()
        #     self.calculate_polar_variables()
        
        self.time_eps_sigma = time.time() - init_time
        self.time_getting_solution = time.time() - init_global
        self.time_global = self.time_getting_solution + self.time_init


    def calculate_error(self, exact_solution, num_solution, type_value):
        dict_nonzero = {idx : value for idx, value in enumerate(exact_solution) if not np.all((np.isclose(value, np.zeros_like(value))))}
        divisible, divisor, relative_error = 0, 0, 0
        for idx, value in dict_nonzero.items():
            relative_error = (np.linalg.norm(value - num_solution[idx]) / np.linalg.norm(value)) ** 2
            if type_value == 'point':
                area_value = np.sum(self.list_area_of_elements[self.dict_elements_contain_point[idx]]) / 3
            elif type_value == 'element':
                area_value = self.list_area_of_elements[idx]

            divisible += area_value * relative_error
            divisor += area_value
        return math.sqrt(divisible / divisor)


    def get_numerical_error_displacement(self):
        value = self.calculate_error(self.u_modified_exact, self.u_polar[:, 0], 'point')
        return value


    def get_numerical_error_sigma(self):
        value = list(map(lambda x: self.calculate_error(self.sigma_exact[:, x], self.sigma_polar[:, x], 'element'), range(2)))
        return value


    def conjugate_method(self, A, b, x = None):
        init_time = time.time()
        amnt_iters_cg = 0
        n = len(b)
        if x is None:
            x = np.ones(n)
        
        r = b - A.dot(x)
        r_0_norm = np.linalg.norm(b)
        if np.linalg.norm(r) < 1e-10:
            return x
        z = r
        while True:
            mult = A.dot(z)
            r_previous_norm = np.dot(r, r)
            alpha = r_previous_norm / np.dot(mult, z)

            x += alpha * z
            r -= alpha * mult
            coef_convergence = np.linalg.norm(r) / r_0_norm
            if coef_convergence < 1e-8:
                time_cg = time.time() - init_time
                break
            else:
                r_current_norm = np.dot(r, r)
                beta = r_current_norm / r_previous_norm
                z = r + beta * z
                amnt_iters_cg += 1
        return x, amnt_iters_cg, time_cg


    def plot_displacement(self):
        self.internal_plot_displacement(self.area_points_coords_modified, self.area_elements)


    def plot_area_init_mesh(self):
        self.internal_plot_displacement(self.area_points_coords, self.area_elements)

    
    def analysis_time(self):
        print(f'Initialization: {self.time_init:.3f}')
        print(f'Time 1: {self.time_1:.3f} {self.time_1 / self.time_init:.2%}')
        print(f'Time 2: {self.time_2:.3f} {self.time_2 / self.time_init:.2%}')
        print(f'Time 3: {self.time_3:.3f} {self.time_3 / self.time_init:.2%}')
        print(f'Time 4: {self.time_4:.3f} {self.time_4 / self.time_init:.2%}')
        print(f'Time 5: {self.time_5:.3f} {self.time_5 / self.time_init:.2%}')
        print(f'Time 6: {self.time_6:.3f} {self.time_6 / self.time_init:.2%}')
        print(f'Time 7: {self.time_7:.3f} {self.time_7 / self.time_init:.2%}')
        print(f'Time 8: {self.time_8:.3f} {self.time_8 / self.time_init:.2%}')
        # print(f'Time 9: {self.time_9} {self.time_9 / self.time_init:.2%}')


if __name__ == "__main__":
    pass