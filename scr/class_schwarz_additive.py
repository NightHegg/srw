import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from itertools import combinations
import numpy as np
import time
from multiprocessing import Pool, cpu_count
import math

import scr.functions as base_func
from scr.class_schwarz_multiplicative import schwarz_multiplicative


class schwarz_additive(schwarz_multiplicative):
    def __init__(self, data):
        init_time = time.time()
        super().__init__(data)
        
        self.name_method = "schwarz additive method"
        self.coef_alpha = data['coef_alpha']
        self.time_init = time.time() - init_time


    def internal_initialize_displacements(self):
        self.u_previous = np.copy(self.u)
        self.u_current = np.copy(self.u)
        self.u_sum = np.zeros_like(self.u)


    def set_condition_schwarz(self, function_condition_schwarz):
        function_condition_schwarz(self.u_previous)


    def internal_additional_calculations(self):
        self.u_sum += (self.u_current - self.u_previous)
        self.u_current = np.copy(self.u_previous)


    def interal_final_calculate_u(self):
        self.u = np.copy(self.u_previous) + (self.coef_alpha * np.copy(self.u_sum))

    # def test_calculate_u(self, idv, u_current, u_previous, list_subd_points):
    #     dict_points_local_to_global = dict(zip(range(len(list_subd_points)), list_subd_points))
    #     self.dict_points_global_to_local = {v: k for k, v in dict_points_local_to_global.items()}

    #     self.K = self.K_array[idv].copy()
    #     self.F = np.zeros(self.list_subd_points_coords[idv].size)

    #     for point in self.list_dirichlet_points[idv]:
    #         for idx, cur_condition in enumerate(self.dirichlet_points[point]):
    #             modified_point = self.dict_points_global_to_local[point]
    #             if not math.isnan(cur_condition):
    #                 indices = np.array(self.K.rows[modified_point * self.dim_task + idx])

    #                 self.F[indices] -= np.array(self.K.data[modified_point * self.dim_task + idx]) * cur_condition
                    
    #                 for index in indices:
    #                     self.K[modified_point * self.dim_task + idx, index] = 0
                    
    #                 for index in indices:
    #                     self.K[index, modified_point * self.dim_task + idx] = 0

    #                 self.K[modified_point * self.dim_task + idx, modified_point * self.dim_task + idx] = 1
    #                 self.F[modified_point * self.dim_task + idx] = cur_condition
        
    #     for element in self.list_neumann_elements_subd[idv]:
    #         points = list(set(element) & set(self.neumann_points.keys()))
    #         points_coords = [self.area_points_coords[point] for point in points]
    #         len = np.linalg.norm(np.array(points_coords[1]) - np.array(points_coords[0]))
    #         for point in points:
    #             for idx, cur_condition in enumerate(self.neumann_points[point]):
    #                 modified_point = self.dict_points_global_to_local[point]
    #                 if not math.isnan(cur_condition):
    #                     self.F[modified_point * self.dim_task + idx] += cur_condition * len / 2
                        
    #     for point in self.list_schwarz_points[idv]:
    #         modified_point = self.dict_points_global_to_local[point]
    #         for cur_dim in range(self.dim_task):
    #             value = u_current[point, cur_dim]

    #             indices = np.array(self.K.rows[modified_point * self.dim_task + cur_dim])

    #             self.F[indices] -= np.array(self.K.data[modified_point * self.dim_task + cur_dim]) * value

    #             for index in indices:
    #                 self.K[modified_point * self.dim_task + cur_dim, index] = 0

    #             for index in indices:
    #                 self.K[index, modified_point * self.dim_task + cur_dim] = 0

    #             self.K[modified_point * self.dim_task + cur_dim, modified_point * self.dim_task + cur_dim] = 1
    #             self.F[modified_point * self.dim_task + cur_dim] = value

    #     [*arg,] = self.solve_function(self.K.tocsr(), self.F)
    #     u_subd = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

    #     for x in list(dict_points_local_to_global.keys()):
    #         u_current[dict_points_local_to_global[x], :] = np.copy(u_subd[x, :])
    #     return u_current - u_previous


    # def calculate_u(self):
    #     self.amnt_iterations = 0
    #     self.u = np.zeros((self.area_points_coords.shape[0], 2))

    #     self.init_subd_params()

    #     self.list_neumann_elements_subd = []
    #     self.list_schwarz_points = []
    #     self.list_dirichlet_points = []

    #     self.list_sum_elements = [base_func.calculate_local_matrix_stiffness(i, self.area_points_coords, self.dim_task)[1] for i in self.area_elements]
        
    #     for idv in range(len(self.list_full_subd_elements)):          
    #         temp = []
    #         for element in self.list_subd_elements[idv]:
    #             if len(set(self.area_elements[element]) & set(self.neumann_points.keys())) == 2:
    #                 temp.append(self.area_elements[element])
    #         self.list_neumann_elements_subd.append(temp)

    #         self.list_schwarz_points.append(sum([list(set(self.subd_boundary_overlap_points[idv]) & set(subd)) for idx, subd in enumerate(self.list_subd_points) if idx != idv], []))
    #         self.list_dirichlet_points.append(list(set(self.dirichlet_points.keys()) & set(self.list_subd_points[idv])))
    #     idv = range(len(self.list_full_subd_elements))

    #     while True:
    #         self.internal_initialize_displacements()
    #         list_temp = [[val, self.u_current, self.u_previous, self.list_subd_points[val]] for val in idv]
    #         with Pool(processes = len(idv)) as pool:
    #             result = pool.starmap_async(self.test_calculate_u, list_temp).get()
    #             self.u_sum = np.sum(np.array(result), axis = 0).reshape(-1, 2)
    #         self.amnt_iterations += 1
    #         self.interal_final_calculate_u()

    #         crit_convergence = self.calculate_crit_convergence()
    #         print(f"{crit_convergence:.3e}", end = "\r")
    #         if crit_convergence < self.coef_convergence:
    #             break

if __name__ == "__main__":
    pass
