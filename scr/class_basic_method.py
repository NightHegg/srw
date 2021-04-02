import os
import sys
from numpy.core.defchararray import index

from numpy.core.numeric import outer
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import linalg
from itertools import combinations
import meshio
import math
import scipy
import matplotlib.patches as mpatches

import scr.functions as base_func


class basic_method:
    def __init__(self, data):
        init_time = time.time()
        self.name_method = "basic method"
        self.message = {}
        self.data = data

        self.solve_function = linalg.spsolve
        temp_contour = []

        with open(f'data/{self.data["area"]}/area_info.dat', 'r') as f:
            for _ in range(int(f.readline())):
                temp_contour.append([float(x) for x in f.readline().split()])
            self.dim_task = int(f.readline())
            E, nyu = list(map(float, f.readline().split()))
            self.coef_u, self.coef_sigma = list(map(float, f.readline().split()))

        mesh = meshio.read(f'data/{self.data["area"]}/meshes/fine_meshes/{self.data["mesh"]}.dat')
        
        self.contour_points = np.append(np.array(temp_contour), [temp_contour[0]], axis = 0)
        self.area_points_coords = mesh.points
        self.area_points = [num for num, _ in enumerate(self.area_points_coords)]
        self.area_elements = mesh.cells_dict["triangle"]

        self.D = np.array([[1, nyu/(1 - nyu), 0],
                        [nyu/(1 - nyu), 1, 0], 
                        [0, 0, (1 - 2 * nyu)  / 2 / (1 - nyu)]]
                        ) * E * (1 - nyu) / (1 - 2 * nyu) / (1 + nyu)
 
        dirichlet_conditions = []
        neumann_conditions = []
        self.dict_area_dirichlet_points = {}
        self.dict_area_neumann_points = {}

        with open(f'data/{self.data["area"]}/tasks/{self.data["task"]}.dat', 'r') as f:
            if self.data['area'] == 'rectangle':
                for _ in range(int(f.readline())):
                    dirichlet_conditions.append([int(val) if ind in [0, 1] else float(val) for ind, val in enumerate(f.readline().split())])

                for _ in range(int(f.readline())):
                    neumann_conditions.append([int(val) if ind in [0, 1] else float(val) for ind, val in enumerate(f.readline().split())])
            elif self.data['area'] == 'thick_walled_cylinder':
                self.inner_radius = self.contour_points[0, 0]
                self.outer_radius = self.contour_points[1, 0]

                self.inner_displacement = float(f.readline())
                self.outer_displacement = float(f.readline())
                dirichlet_conditions.append([self.inner_radius, self.inner_displacement])
                dirichlet_conditions.append([self.outer_radius, self.outer_displacement])

                for _ in range(int(f.readline())):
                    dirichlet_conditions.append([int(val) if ind in [0, 1] else float(val) for ind, val in enumerate(f.readline().split())])

                self.inner_pressure = float(f.readline())
                self.outer_pressure = float(f.readline())
                neumann_conditions.append([self.inner_radius, self.inner_pressure])
                neumann_conditions.append([self.outer_radius, self.outer_pressure])
        
        self.inner_radius_points = np.isclose(self.inner_radius, np.linalg.norm(self.area_points_coords, axis = 1))
        self.outer_radius_points = np.isclose(self.outer_radius, np.linalg.norm(self.area_points_coords, axis = 1))

        dirichlet_radius_points = []
        if self.data['area'] == 'rectangle':
            for row in dirichlet_conditions:
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
            for row in neumann_conditions:
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
        elif self.data['area'] == 'thick_walled_cylinder':    
            for ind, row in enumerate(dirichlet_conditions):
                if ind in [0, 1]:
                    radius, displacement = row[0], row[1]
                    for point in self.area_points:
                        point_coords = np.array(self.area_points_coords[point])
                        if np.isclose(np.linalg.norm(point_coords), radius):
                            self.dict_area_dirichlet_points[point] = point_coords * (displacement / radius)
                else:
                    a, b = np.array(self.contour_points[row[0]]), np.array(self.contour_points[row[1]])
                    for point in self.area_points:
                        point_coords = np.array(self.area_points_coords[point])
                        if (abs(np.cross(b-a, point_coords - a)) < 1e-15 and np.dot(b-a, point_coords - a) >= 0 and np.dot(b-a, point_coords - a) < np.linalg.norm(a-b)) or np.allclose(point_coords, a) or np.allclose(point_coords, b):
                            if point in self.dict_area_dirichlet_points:
                                if math.isnan(self.dict_area_dirichlet_points[point][0]) and math.isnan(self.dict_area_dirichlet_points[point][1]):
                                    self.dict_area_dirichlet_points[point] = [row[2], row[3]]
                                else:
                                    if math.isnan(row[2]):
                                        self.dict_area_dirichlet_points[point] = [self.dict_area_dirichlet_points[point][0], row[3]]
                                    elif math.isnan(row[3]):
                                        self.dict_area_dirichlet_points[point] = [row[2], self.dict_area_dirichlet_points[point][1]]
                            else:
                                self.dict_area_dirichlet_points[point] = [row[2], row[3]]
            for row in neumann_conditions:
                radius, pressure = row[0], row[1]
                for point in self.area_points:
                    point_coords = np.array(self.area_points_coords[point])
                    if np.isclose(np.linalg.norm(point_coords), radius):
                        self.dict_area_neumann_points[point] = pressure
        
        coef_lambda = nyu * E / (1 + nyu) / (1 - 2 * nyu)
        coef_myu = nyu / 2 / (1 + nyu)
        # A = (self.inner_pressure * self.inner_radius ** 2 - self.outer_pressure * self.outer_radius ** 2) / 2 / (coef_lambda + coef_myu) / (self.outer_radius ** 2 - self.inner_radius ** 2)
        # B = (self.inner_pressure - self.outer_pressure) * self.inner_radius ** 2 * self.outer_radius ** 2 / 2 / coef_myu / (self.outer_radius ** 2 - self.inner_radius ** 2)

        # self.u_exact = (A + 1/B) * np.linalg.norm(self.area_points_coords, axis = 1)            

        self.list_area_neumann_elements = []
        for index_element, element in enumerate(self.area_elements):
            if len(set(element) & set(self.dict_area_neumann_points.keys())) == 2:
                self.list_area_neumann_elements.append(index_element)

        self.list_sum_elements = [base_func.calculate_local_matrix_stiffness(i, self.area_points_coords, self.dim_task)[1] for i in self.area_elements]
        self.time_init = time.time() - init_time


    def set_condition_dirichlet(self, K, F, list_dirichlet_points, modifier = {}):
        for point in list_dirichlet_points:
            for cur_dimension, cur_condition in enumerate(self.dict_area_dirichlet_points[point]):
                modified_point = modifier[point] if modifier else point
                if not math.isnan(cur_condition):
                    indices = np.array(K.rows[modified_point * self.dim_task + cur_dimension])
                    F[indices] -= np.array(K.data[modified_point * self.dim_task + cur_dimension]) * cur_condition
                    for index in indices:
                        K[modified_point * self.dim_task + cur_dimension, index] = 0
                    for index in indices:
                        K[index, modified_point * self.dim_task + cur_dimension] = 0
                    K[modified_point * self.dim_task + cur_dimension, modified_point * self.dim_task + cur_dimension] = 1
                    F[modified_point * self.dim_task + cur_dimension] = cur_condition


    def set_condition_neumann(self, F, list_neumann_elements, modifier = {}):
        for index_element in list_neumann_elements:
            element = self.area_elements[index_element]   
            points = list(set(element) & set(self.dict_area_neumann_points.keys()))
            len = np.linalg.norm(np.diff(self.area_points_coords[points], axis = 0))
            value = np.flip(np.ravel(np.abs((np.diff(self.area_points_coords[points], axis = 0)))))
            for point in points:
                modified_point = modifier[point] if modifier else point
                pressure = self.dict_area_neumann_points[point]
                F[modified_point * self.dim_task: (modified_point + 1) * self.dim_task] += value * (pressure / len) * len / 2


    def calculate_u(self):
        K = base_func.calculate_sparse_matrix_stiffness(self.area_elements, self.area_points_coords, self.D, self.dim_task)
        F = np.zeros(self.area_points_coords.size)

        # self.set_condition_neumann(F, self.list_area_neumann_elements)
        self.set_condition_dirichlet(K, F, self.dict_area_dirichlet_points.keys())

        *arg, = self.solve_function(K.tocsr(), F)
        self.u = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

        self.area_point_coords_modified = self.u * self.coef_u + self.area_points_coords
        self.u_polar = np.hstack(
            (
                np.linalg.norm(self.area_point_coords_modified, axis = 1).reshape(-1, 1), 
                np.arctan2(self.area_point_coords_modified[:, 1], self.area_point_coords_modified[:, 0]).reshape(-1, 1)
            )
        )

        print(self.u_polar[self.inner_radius_points, 0])
        # assert np.allclose(
        #     self.u_polar[self.outer_radius_points, 0],
        #     np.amax(self.u_polar[self.outer_radius_points, 0])
        #     ),'Error: unsymmetric outer boundary'

        # assert np.allclose(
        #     self.u_polar[self.inner_radius_points, 0],
        #     np.amax(self.u_polar[self.inner_radius_points, 0])
        #     ),'Error: unsymmetric inner boundary'
                

    def calculate_eps(self):
        temp_array = []
        for element in self.area_elements:
            B, _ = base_func.calculate_local_matrix_stiffness(element, self.area_points_coords, self.dim_task)
            temp_array.append(np.dot(B, np.ravel([self.u[element[index]] for index in range(len(element))]).T))
        self.eps = np.array(temp_array)


    def calculate_sigma(self):
        self.sigma = self.D @ self.eps.T


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

        if self.data['area'] == 'rectangle':
            ax.plot(self.contour_points[:, 0], self.contour_points[:, 1], color = "brown")
        elif self.data['area'] == 'thick_walled_cylinder':
            inner_radius = self.contour_points[0, 0] * 2
            outer_radius = self.contour_points[1, 0] * 2

            inner_circle = mpatches.Arc([0, 0], inner_radius, inner_radius, angle = 0, theta1 = 0, theta2 = 90)
            outer_circle = mpatches.Arc([0, 0], outer_radius, outer_radius, angle = 0, theta1 = 0, theta2 = 90)

            ax.add_patch(outer_circle)
            ax.add_patch(inner_circle)

            ax.plot(self.contour_points[:2, 0], self.contour_points[:2, 1], color = "k")
            ax.plot(self.contour_points[2:-1, 0], self.contour_points[2:-1, 1], color = "k")

            ax.set_xlim([0 - outer_radius/8, outer_radius/2 + outer_radius/8])
            ax.set_ylim([0 - outer_radius/8, outer_radius/2 + outer_radius/8])
        if plot_global_mesh:
            ax.triplot(area_points_coords[:,0], area_points_coords[:,1], area_elements.copy())

        fig.set_figwidth(8)
        fig.set_figheight(8)
        fig.set_facecolor('mintcream')

        plt.show()


    def plot_displacements(self, plot_global_mesh = True):
        self.internal_plot_displacements(self.u, self.area_points_coords, self.area_elements, plot_global_mesh)


    def plot_init_mesh(self, contour = True):
        fig, ax = plt.subplots()

        if contour:
            if self.data['area'] == 'rectangle':
                ax.plot(self.contour_points[:, 0], self.contour_points[:, 1], color = "brown")
            elif self.data['area'] == 'thick_walled_cylinder':
                inner_radius = self.contour_points[0, 0] * 2
                outer_radius = self.contour_points[1, 0] * 2

                inner_circle = mpatches.Arc([0, 0], inner_radius, inner_radius, angle = 0, theta1 = 0, theta2 = 90)
                outer_circle = mpatches.Arc([0, 0], outer_radius, outer_radius, angle = 0, theta1 = 0, theta2 = 90)

                ax.add_patch(outer_circle)
                ax.add_patch(inner_circle)

                ax.plot(self.contour_points[:2, 0], self.contour_points[:2, 1], color = "k")
                ax.plot(self.contour_points[2:-1, 0], self.contour_points[2:-1, 1], color = "k")

        ax.triplot(self.area_points_coords[:, 0], self.area_points_coords[:, 1], self.area_elements.copy())

        fig.set_figwidth(8)
        fig.set_figheight(8)
        fig.set_facecolor('mintcream')

        plt.show()


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


    def get_special_sigma(self):
        return f"min = {abs(abs(min(self.sigma[1])) - 2e+7) / 2e+7:.2e}, max = {abs(abs(max(self.sigma[1])) - 2e+7) / 2e+7:.2e}"


if __name__ == "__main__":
    pass