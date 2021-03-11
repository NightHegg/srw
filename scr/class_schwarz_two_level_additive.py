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

import scr.functions as base_func
from scr.class_schwarz_additive import schwarz_additive

class schwarz_two_level_additive(schwarz_additive):
    # TODO: Можно ли area_limits тоже сделать атрибутом класса?
    def __init__(self, data):
        super().__init__(data)

        self.name_method = "schwarz additive two level method"

        coarse_mesh = meshio.read(f'data/{data["area"]}/meshes/{data["coarse_mesh"]}.dat')
        
        self.area_coarse_points_coords = coarse_mesh.points
        self.area_coarse_points = [num for num, _ in enumerate(self.area_coarse_points_coords)]
        self.area_coarse_elements = coarse_mesh.cells_dict["triangle"]
 
        self.dirichlet_coarse_points = {}
        self.neumann_coarse_points = {}

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
                   
        for row in self.neumann_conditions:
            a, b = np.array(self.contour_points[row[0]]), np.array(self.contour_points[row[1]])
            for point in self.area_coarse_points:
                point_coords = np.array(self.area_coarse_points_coords[point])
                if abs(np.cross(b-a, point_coords - a)) < 1e-15 and np.dot(b-a, point_coords - a) >= 0 and np.dot(b-a, point_coords - a) < np.linalg.norm(a-b):
                    if point in self.neumann_coarse_points:
                        if math.isnan(self.neumann_points[point][0]) and not math.isnan(row[2]):
                            self.neumann_coarse_points[point] = [row[2], self.neumann_coarse_points[point][1]]
                        else:
                            self.neumann_coarse_points[point] = [self.neumann_coarse_points[point][0], row[3]]
                    else:
                        self.neumann_coarse_points[point] = [row[2], row[3]]


    def interal_calculate_u(self):
        u_special = np.ravel(np.zeros_like(self.u))

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

        F_test = np.dot(K.toarray(), np.ravel(self.u_previous))     
        residual = F - F_test

        K_coarse = base_func.calculate_sparse_matrix_stiffness(self.area_coarse_elements, self.area_coarse_points_coords, self.D, self.dim_task)
        F_coarse = np.zeros(self.area_coarse_points_coords.size)

        chosen_element = []
        for num_point, point_coords in enumerate(self.area_points_coords):
            chosen_element = []
            for element in self.area_coarse_elements:
                S_element = base_func.calculate_element_area(self.area_coarse_points_coords[element[0]], self.area_coarse_points_coords[element[1]], self.area_coarse_points_coords[element[2]])
                S_1 = base_func.calculate_element_area(point_coords, self.area_coarse_points_coords[element[1]], self.area_coarse_points_coords[element[2]])
                S_2 = base_func.calculate_element_area(self.area_coarse_points_coords[element[0]], point_coords, self.area_coarse_points_coords[element[2]])
                S_3 = base_func.calculate_element_area(self.area_coarse_points_coords[element[0]], self.area_coarse_points_coords[element[1]], point_coords)
                if S_1 + S_2 + S_3 - S_element < 1e-9:
                        chosen_element = element
                        break

            local_coords = base_func.calculate_local_functions(chosen_element, self.area_coarse_points_coords)

            for i in range(3):
                for dim in range(self.dim_task):
                    F_coarse[chosen_element[i] * self.dim_task + dim] += local_coords(point_coords, i) * residual[num_point * self.dim_task + dim]
        
        for point, condition in self.dirichlet_coarse_points.items():
            if math.isnan(condition[0]):
                K_coarse, F_coarse = base_func.bound_condition_dirichlet(K_coarse, F_coarse, self.dim_task, point, condition[1], 1)
            elif math.isnan(condition[1]):
                K_coarse, F_coarse = base_func.bound_condition_dirichlet(K_coarse, F_coarse, self.dim_task, point, condition[0], 0)
            else:
                K_coarse, F_coarse = base_func.bound_condition_dirichlet(K_coarse, F_coarse, self.dim_task, point, condition[0], 0)
                K_coarse, F_coarse = base_func.bound_condition_dirichlet(K_coarse, F_coarse, self.dim_task, point, condition[1], 1)

        [*arg,] = self.solve_function(K_coarse.tocsr(), F_coarse)
        u_coarse = np.ravel(np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2)))

        for num_point, point_coords in enumerate(self.area_points_coords):
            chosen_element = []
            for element in self.area_coarse_elements:
                S_element = base_func.calculate_element_area(self.area_coarse_points_coords[element[0]], self.area_coarse_points_coords[element[1]], self.area_coarse_points_coords[element[2]])
                S_1 = base_func.calculate_element_area(point_coords, self.area_coarse_points_coords[element[1]], self.area_coarse_points_coords[element[2]])
                S_2 = base_func.calculate_element_area(self.area_coarse_points_coords[element[0]], point_coords, self.area_coarse_points_coords[element[2]])
                S_3 = base_func.calculate_element_area(self.area_coarse_points_coords[element[0]], self.area_coarse_points_coords[element[1]], point_coords)
                if S_1 + S_2 + S_3 - S_element < 1e-9:
                        chosen_element = element
                        break

            function = base_func.calculate_local_functions(chosen_element, self.area_coarse_points_coords)

            for i in range(3):
                for dim in range(self.dim_task):
                    u_special[num_point * self.dim_task + dim] += function(point_coords, i) * u_coarse[chosen_element[i] * self.dim_task + dim]

        self.u = self.u_previous + (self.coef_alpha * self.u_sum) + (self.coef_alpha * u_special.reshape(-1, 2))


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
