import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import linalg
import meshio
import math
import matplotlib.patches as mpatches

import scr.functions as base_func


class class_template:
    def __init__(self, data):
        self.message = {}
        self.data = data

        self.solve_function = linalg.spsolve
        temp_contour = []

        with open(f'data/{self.data["area"]}/area_info.dat', 'r') as f:
            for _ in range(int(f.readline())):
                temp_contour.append([float(x) for x in f.readline().split()])
            self.dim_task = int(f.readline())
            self.E, self.nyu = list(map(float, f.readline().split()))
            self.coef_u, self.coef_sigma = list(map(float, f.readline().split()))

        mesh = meshio.read(f'data/{self.data["area"]}/meshes_fine/{self.data["mesh"]:.3e}.msh')
        
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

        with open(f'data/{self.data["area"]}/tasks/{self.data["task"]}.dat', 'r') as f:
            if self.data["area"] == 'rectangle':
                for _ in range(int(f.readline())):
                    self.dirichlet_conditions.append(np.array([int(val) if ind in [0, 1] else float(val) for ind, val in enumerate(f.readline().split())]))
                for _ in range(int(f.readline())):
                    self.neumann_conditions.append(np.array([int(val) if ind in [0, 1] else float(val) for ind, val in enumerate(f.readline().split())]))
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

        if self.data["area"] == 'rectangle':
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
        if self.data["area"] == 'rectangle':
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
            if self.data["area"] == 'rectangle':
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


    def calculate_polar_variables(self):
        self.u_polar = np.hstack(
            (
                np.linalg.norm(self.area_points_coords_modified, axis = 1).reshape(-1, 1),
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

        if not self.data['area'] == 'rectangle':
            self.calculate_exact_variables()
            self.calculate_polar_variables()
        
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


    def internal_plot_displacements(self, point_coords, elements, special_points = None, draw_points = False):
        fig, ax = plt.subplots()

        ax.triplot(point_coords[:, 0], point_coords[:, 1], elements)
        if draw_points:
            ax.plot(point_coords[special_points, 0], point_coords[special_points, 1], 'o')

        if self.data['area'] == 'rectangle':
            ax.plot(self.contour_points[:, 0], self.contour_points[:, 1], color = "brown")
        elif self.data['area'] == 'thick_walled_cylinder':
            inner_circle = mpatches.Arc([0, 0], self.inner_radius * 2, self.inner_radius * 2, angle = 0, theta1 = 0, theta2 = 90)
            outer_circle = mpatches.Arc([0, 0], self.outer_radius * 2, self.outer_radius * 2, angle = 0, theta1 = 0, theta2 = 90)

            ax.add_patch(inner_circle)
            ax.add_patch(outer_circle)

            ax.plot(self.contour_points[:2, 0], self.contour_points[:2, 1], color = "k")
            ax.plot(self.contour_points[2:-1, 0], self.contour_points[2:-1, 1], color = "k")

            ax.set_xlim([-self.outer_radius/8, self.outer_radius * 9 / 8])
            ax.set_ylim([-self.outer_radius/8, self.outer_radius * 9 / 8])

        fig.set_figwidth(12)
        fig.set_figheight(12)
        fig.set_facecolor('mintcream')

        plt.show()

    def internal_plot_displacements_coarse(self, points_coords, coarse_element, elements):
        fig, ax = plt.subplots()
        ax.plot(points_coords[:, 0], points_coords[:, 1], 'o')
        ax.triplot(coarse_element[:, 0], coarse_element[:, 1], elements.copy())

        if self.data['area'] == 'rectangle':
            ax.plot(self.contour_points[:, 0], self.contour_points[:, 1], color = "brown")
        elif self.data['area'] == 'thick_walled_cylinder':
            inner_circle = mpatches.Arc([0, 0], self.inner_radius * 2, self.inner_radius * 2, angle = 0, theta1 = 0, theta2 = 90)
            outer_circle = mpatches.Arc([0, 0], self.outer_radius * 2, self.outer_radius * 2, angle = 0, theta1 = 0, theta2 = 90)

            ax.add_patch(inner_circle)
            ax.add_patch(outer_circle)

            ax.plot(self.contour_points[:2, 0], self.contour_points[:2, 1], color = "k")
            ax.plot(self.contour_points[2:-1, 0], self.contour_points[2:-1, 1], color = "k")

            ax.set_xlim([-self.outer_radius/8, self.outer_radius * 9 / 8])
            ax.set_ylim([-self.outer_radius/8, self.outer_radius * 9 / 8])

        fig.set_figwidth(10)
        fig.set_figheight(10)
        fig.set_facecolor('mintcream')

        plt.show()


    def plot_displacements(self, save_route = None):
        self.internal_plot_displacements(self.area_points_coords_modified, self.area_elements)


    def plot_init_mesh(self, save_route = None):
        self.internal_plot_displacements(self.area_points_coords, self.area_elements)


    def plot_polar(self):
        plt.polar(self.u_polar[self.inner_radius_points, 1], self.u_polar[self.inner_radius_points, 0], 'o')
        plt.polar(self.u_polar[self.outer_radius_points, 1], self.u_polar[self.outer_radius_points, 0], 'o')
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
            f'Minimal difference for stress: {abs(abs(min(self.sigma[1])) - 2e+7) / 2e+7:.2e}\n',
            f'Maximal difference for stress: {abs(abs(max(self.sigma[1])) - 2e+7) / 2e+7:.2e}\n',
            f'{"-" * 5}\n'
        ]


    def get_info(self):
        self.construct_info()
        return sum(list(self.message.values()), [])


    def pcolormesh(self):
        triang = mtri.Triangulation(self.area_points_coords[:, 0], self.area_points_coords[:, 1], self.area_elements)
        plt.tricontourf(triang, np.linalg.norm(self.u, axis = 1) * self.coef_u)
        plt.colorbar()
        plt.axis('equal')
        plt.show()


    def conjugate_method(self, A, b, x = None):
        n = len(b)
        if not x:
            x = np.ones(n)
        
        r = b - A.dot(x)
        if np.linalg.norm(r) < 1e-8:
            return x
        z = r
        for i in range(2*n):
            r_previous_norm = np.dot(r, r)
            alpha = r_previous_norm / np.dot(A.dot(z), z)
            x += alpha * z
            r -= alpha * A.dot(z)
            coef_convergence = np.linalg.norm(r) / np.linalg.norm(b)
            if coef_convergence < 1e-8:
                # print(f'Amount of iterations: {i}\nN: {n}')
                break
            else:
                r_current_norm = np.dot(r, r)
                beta = r_current_norm / r_previous_norm
                z = r + beta * z
                # print(f'{coef_convergence:.3e}')
        return x

if __name__ == "__main__":
    pass