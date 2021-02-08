import os
import sys
import math

from scipy.sparse.linalg import dsolve
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from scipy.sparse import linalg

from scr.operations_input_files.read_input_files import read_mesh
import scr.functions as base_func
from scr.class_schwarz_additive import schwarz_additive

class schwarz_two_level_additive(schwarz_additive):
    # TODO: Можно ли area_limits тоже сделать атрибутом класса?
    def __init__(self, cur_task, cur_mesh, cur_amnt_subds = [2, 1], coef_convergence = 1e-3, coef_alpha = 0.5, cur_coarse_mesh = 5, solve_function = linalg.spsolve):
        super().__init__(cur_task, cur_mesh, cur_amnt_subds, coef_convergence, coef_alpha, solve_function)

        self.name_method = "schwarz additive two level method"

        _, self.area_points_coords_coarse, self.area_elements_coarse = read_mesh(cur_coarse_mesh)
        self.area_points_coarse = [i for i in (range(len(self.area_points_coords_coarse)))]

        area_limits = lambda cond, type: [self.area_bounds[cond[0], type], self.area_bounds[cond[0] + 1, type]]

        self.dirichlet_points_coarse = {tuple(idx for idx, val in enumerate(self.area_points_coords_coarse) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))) : cond[1:] for cond in self.dirichlet_conditions}

        self.neumann_points_coarse = {tuple(idx for idx, val in enumerate(self.area_points_coords_coarse) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))) : cond[1:] for cond in self.neumann_conditions}


    def interal_calculate_u(self):
        u_special = np.ravel(np.zeros_like(self.u))

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

        residual = F - np.dot(K.toarray(), np.ravel(self.u))

        K_coarse = base_func.calculate_sparse_matrix_stiffness(self.area_elements_coarse, self.area_points_coords_coarse, self.D, self.dimTask)
        F_coarse = np.zeros(self.area_points_coords_coarse.size)

        for list_points, condition in self.dirichlet_points_coarse.items():
            for point in list_points:
                if condition[0] == 2:
                    K_coarse, F_coarse = base_func.bound_condition_dirichlet(K_coarse, F_coarse, self.dimTask, point, condition[1], 0)
                    K_coarse, F_coarse = base_func.bound_condition_dirichlet(K_coarse, F_coarse, self.dimTask, point, condition[2], 1)
                else:
                    K_coarse, F_coarse = base_func.bound_condition_dirichlet(K_coarse, F_coarse, self.dimTask, point, condition[1], condition[0])

        chosen_element = []
        for num_point, point_coords in enumerate(self.area_points_coords_coarse):
            chosen_element = []
            for element in self.area_elements:
                S_element = base_func.calculate_element_area(self.area_points_coords[element[0]], self.area_points_coords[element[1]], self.area_points_coords[element[2]])
                S_1 = base_func.calculate_element_area(point_coords, self.area_points_coords[element[1]], self.area_points_coords[element[2]])
                S_2 = base_func.calculate_element_area(self.area_points_coords[element[0]], point_coords, self.area_points_coords[element[2]])
                S_3 = base_func.calculate_element_area(self.area_points_coords[element[0]], self.area_points_coords[element[1]], point_coords)
                if S_1 + S_2 + S_3 - S_element < 1e-9:
                        chosen_element = element
                        break
                    
            function = base_func.calculate_local_functions(chosen_element, self.area_points_coords)

            for i in range(3):
                for dim in range(self.dimTask):
                    F_coarse[num_point * self.dimTask + dim] += function(point_coords, i) * residual[chosen_element[i] * self.dimTask + dim]
        
        [*arg,] = self.solve_function(K_coarse.tocsr(), F_coarse)
        u_coarse = np.ravel(np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2)))
        
        for num_point, point_coords in enumerate(self.area_points_coords):
            chosen_element = []
            for element in self.area_elements_coarse:
                S_element = base_func.calculate_element_area(self.area_points_coords_coarse[element[0]], self.area_points_coords_coarse[element[1]], self.area_points_coords_coarse[element[2]])
                S_1 = base_func.calculate_element_area(point_coords, self.area_points_coords_coarse[element[1]], self.area_points_coords_coarse[element[2]])
                S_2 = base_func.calculate_element_area(self.area_points_coords_coarse[element[0]], point_coords, self.area_points_coords_coarse[element[2]])
                S_3 = base_func.calculate_element_area(self.area_points_coords_coarse[element[0]], self.area_points_coords_coarse[element[1]], point_coords)
                if S_1 + S_2 + S_3 - S_element < 1e-9:
                        chosen_element = element
                        break

            function = base_func.calculate_local_functions(chosen_element, self.area_points_coords_coarse)

            for i in range(3):
                for dim in range(self.dimTask):
                    u_special[num_point * self.dimTask + dim] += function(point_coords, i) * u_coarse[chosen_element[i] * self.dimTask + dim]     

        self.u = np.copy(self.u_previous) + (self.coef_alpha * np.copy(self.u_sum)) + (self.coef_alpha * u_special.reshape(-1, 2))
        

if __name__ == "__main__":
    pass
