import enum
import os
import sys
import time

from ray.worker import init
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from scipy.sparse import linalg
from scipy.sparse import coo_matrix
import meshio
import math

import scr.functions as base_func
import scr._class_visualisation as class_visual


class class_template(class_visual.class_visualisation):
    def __init__(self, data):
        self.time1 = 0
        self.time2 = 0
        self.time3 = 0
        self.time4 = 0
        self.time5 = 0
        self.time6 = 0
        self.time7 = 0
        self.time8 = 0
        self.time9 = 0
        self.time10 = 0
        self.time11 = 0
        self.time12 = 0
        self.time13 = 0
        self.time14 = 0
        self.time15 = 0
        self.time16 = 0
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
        self.area_elements_indices = np.arange(self.area_elements.shape[0])

        self.N = self.area_points_coords.size

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

        init_time_2 = time.time()

        if self.data["fine_area"] == 'rectangle':
            for point_num, point_coords in enumerate(self.area_points_coords):
                init_time = time.time()

                for row in self.dirichlet_conditions:
                    contour_points = self.contour_points[row[:2].astype(int)]
                    if base_func.func(contour_points[0], contour_points[1], point_coords):
                        if point_num in self.dict_area_dirichlet_points:      
                            template_nan = np.isnan(self.dict_area_dirichlet_points[point_num])
                            self.dict_area_dirichlet_points[point_num][template_nan] = np.copy(row)[2:][template_nan]  
                        else:
                            self.dict_area_dirichlet_points[point_num] = np.copy(row)[2:]
                        

                self.time14 += time.time() - init_time
                init_time = time.time()

                for row in self.neumann_conditions:
                    contour_points = self.contour_points[row[:2].astype(int)]
                    if base_func.func(contour_points[0], contour_points[1], point_coords):
                        self.dict_area_neumann_points[point_num] = row[2:]
                self.time15 += time.time() - init_time

        else:
            for point_num, point_coords in enumerate(self.area_points_coords):
                for index, row in enumerate(self.dirichlet_conditions):
                    if index in [0, 1]:
                        radius, displacement = row[0], row[1]
                        if np.isclose(np.linalg.norm(point_coords), radius):
                            self.dict_area_dirichlet_points[point_num] = point_coords * (displacement / radius)
                    else:
                        contour_points = self.contour_points[row[:2].astype(int)]
                        cond = np.isclose(abs(np.cross(np.diff(contour_points, axis = 0), point_coords - contour_points[0])), 0)
                        # cond = base_func.func(contour_points[0], contour_points[1], point_coords)
                        if cond:
                            if point_num in self.dict_area_dirichlet_points:
                                template_nan = np.isnan(self.dict_area_dirichlet_points[point_num])
                                self.dict_area_dirichlet_points[point_num][template_nan] = np.copy(row)[2:][template_nan]
                            else:
                                self.dict_area_dirichlet_points[point_num] = np.copy(row)[2:]
                for row in self.neumann_conditions:
                    radius, pressure = row[0], row[1]
                    if np.isclose(np.linalg.norm(point_coords), radius):
                        self.dict_area_neumann_points[point_num] = pressure

        self.time7 = time.time() - init_time_2
        init_time = time.time()

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

        self.time8 = time.time() - init_time
        init_time = time.time()

        self.lst_B, self.lst_A = self.calculate_B_A(self.area_elements, self.area_points_coords)

        self.time9 = time.time() - init_time
        init_time = time.time()

        self.dict_elements_contain_point = {}
        for index in range(self.area_elements.shape[0]):
            for point in self.area_elements[index]:
                if point in self.dict_elements_contain_point.keys():
                    self.dict_elements_contain_point[point] = np.append(self.dict_elements_contain_point[point], index)
                else:
                    self.dict_elements_contain_point[point] = np.array([index])
        self.dict_elements_contain_point = dict(sorted(self.dict_elements_contain_point.items(), key=lambda x: x[0]))

        self.time10 = time.time() - init_time
        init_time = time.time()

        self.special_area = []
        for point in self.area_points:
            self.special_area.append(sum(self.lst_A[self.dict_elements_contain_point[point]])/3)

        self.time11 = time.time() - init_time
        init_time = time.time()

        self.element_centroid_points_coords = np.array(list(map(lambda x: np.mean(self.area_points_coords[x], axis = 0), self.area_elements)))

        self.time12 = time.time() - init_time
        init_time = time.time()

        self.element_centroid_points_angles = np.array(list(map(lambda x: np.arctan2(x[1], x[0]), self.element_centroid_points_coords)))

        self.time13 = time.time() - init_time


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
        for index, element in enumerate(self.area_elements):
            B = self.lst_B[index]
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
        arr_nonzero = np.where(~np.all(np.isclose(exact_solution, np.zeros_like(exact_solution)), axis=1))[0]
        vec = (np.linalg.norm(exact_solution[arr_nonzero] - num_solution[arr_nonzero], axis=1) / np.linalg.norm(exact_solution[arr_nonzero], axis=1)) ** 2
        divisible, divisor, relative_error = 0, 0, 0
        for idx in range(len(arr_nonzero)):
            relative_error = vec[idx]

            if type_value == 'point':
                area_value = self.special_area[arr_nonzero[idx]]
            elif type_value == 'element':
                area_value = self.list_area_of_elements[arr_nonzero[idx]]

            divisible += area_value * relative_error
            divisor += area_value
        return math.sqrt(divisible / divisor)


    def get_numerical_error_displacement(self):
        value = self.calculate_error(self.u_modified_exact, self.u_polar[:, 0], 'point')
        return value


    def get_numerical_error_sigma(self):
        value = list(map(lambda x: self.calculate_error(self.sigma_exact[:, x], self.sigma_polar[:, x], 'element'), range(2)))
        return value


    def conjugate_method(self, A, b, crit_convergence = 1e-5, x = None):
        init_time = time.time()
        local_crit = 1e-3 if crit_convergence >= 1 else crit_convergence * 1e-3
        
        amnt_iters_cg = 0
        if x is None:
            x = np.ones_like(b)
        
        r = b - A.dot(x)
        r_0_norm = np.linalg.norm(b)
        z = r
        while True:
            mult = A.dot(z)
            r_previous_norm = np.dot(r, r)
            alpha = r_previous_norm / np.dot(mult, z)

            x += alpha * z
            r -= alpha * mult
            coef_convergence = np.linalg.norm(r) / r_0_norm
            if coef_convergence < local_crit:
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


    def calculate_B_A(self, elements, points):
        arr_B = []
        arr_A = []
        for element in elements:
            M = np.hstack((np.ones((points[element].shape[0], 1)), points[element])).T
            A = 0.5*np.linalg.det(M)
            M_inv = np.linalg.inv(M)
            a, b, c = M_inv[:, 0], M_inv[:, 1], M_inv[:, 2]
            M_new = np.array([
                [a[0], 0, a[1], 0, a[2], 0],
                [b[0], 0, b[1], 0, b[2], 0],
                [c[0], 0, c[1], 0, c[2], 0],
                [0, a[0], 0, a[1], 0, a[2]],
                [0, b[0], 0, b[1], 0, b[2]],
                [0, c[0], 0, c[1], 0, c[2]]
            ])
            A_form = np.array([
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0.5, 0, 0.5, 0]
            ])
            B = np.dot(A_form, M_new)
            arr_B.append(B)
            arr_A.append(A)
        return arr_B, np.array(arr_A)


    def calculate_sparse_matrix_stiffness(self, indices_elements, area_elements, lst_B, lst_A, amnt_area_points, modifier = {}):
        row, col, data = [], [], []
        for index in indices_elements:
            K_element = lst_B[index].T @ self.D @ lst_B[index] * lst_A[index]
            for i in range(3):
                for j in range(3):
                    point_1 = modifier[area_elements[index][i]] if modifier else area_elements[index][i]
                    point_2 = modifier[area_elements[index][j]] if modifier else area_elements[index][j]
                    for k in range(self.dim_task):
                        for z in range(self.dim_task):
                            row.append(point_1 * self.dim_task + k)
                            col.append(point_2 * self.dim_task + z)
                            data.append(K_element[i * self.dim_task + k, j * self.dim_task + z])
        return coo_matrix((data, (row, col)), shape = (amnt_area_points * self.dim_task, amnt_area_points * self.dim_task)).tolil()

    
    def analysis_time(self):
        print(f'u: {self.time_u:.3f} {self.time_u / (self.time_u + self.time_init):.2%}')
        print(f'Time 1: {self.time1:.3f} {self.time1 / self.time_u:.2%}')
        print(f'Time 2: {self.time2:.3f} {self.time2 / self.time_u:.2%}')
        print(f'Time 3: {self.time3:.3f} {self.time3 / self.time_u:.2%}')
        print(f'Time 4: {self.time4:.3f} {self.time4 / self.time_u:.2%}')
        print(f'Time 5: {self.time5:.3f} {self.time5 / self.time_u:.2%}')
        print(f'Time 6: {self.time6:.3f} {self.time6 / self.time_u:.2%}')
        print()
        print(f'Initialization: {self.time_init:.3f} {self.time_init / (self.time_u + self.time_init):.2%}')
        print(f'Time 7: {self.time7:.3f} {self.time7 / self.time_init:.2%}')
        print(f'Time 8: {self.time8:.3f} {self.time8 / self.time_init:.2%}')
        print(f'Time 9: {self.time9:.3f} {self.time9 / self.time_init:.2%}')
        print(f'Time 10: {self.time10:.3f} {self.time10 / self.time_init:.2%}')
        print(f'Time 11: {self.time11:.3f} {self.time11 / self.time_init:.2%}')
        print(f'Time 12: {self.time12:.3f} {self.time12 / self.time_init:.2%}')
        print(f'Time 13: {self.time13:.3f} {self.time13 / self.time_init:.2%}')
        print(f'Time 14: {self.time14:.3f} {self.time14 / self.time_init:.2%}')
        print(f'Time 15: {self.time15:.3f} {self.time15 / self.time_init:.2%}')
        print(f'Time 16: {self.time16:.3f} {self.time16 / self.time_init:.2%}')


if __name__ == "__main__":
    pass