import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from itertools import combinations
import numpy as np
from scipy.sparse import linalg

import scr.functions as base_func
from scr.class_schwarz_multiplicative import schwarz_multiplicative


class schwarz_additive(schwarz_multiplicative):
    def __init__(self, data):
        super().__init__(data)
        
        self.name_method = "schwarz additive method"
        self.coef_alpha = data['coef_alpha']


    def internal_initialize_displacements(self):
        self.u_previous = np.copy(self.u)
        self.u_current = np.copy(self.u)
        self.u_sum = np.zeros_like(self.u)


    def set_condition_schwarz(self, idv, array_condition):
        return super().set_condition_schwarz(idv, self.u_previous)


    def internal_additional_calculations(self):
        self.u_sum += (self.u_current - self.u_previous)
        self.u_current = np.copy(self.u_previous)


    def interal_final_calculate_u(self):
        self.u = np.copy(self.u_previous) + (self.coef_alpha * np.copy(self.u_sum))

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

    #     while True:
    #         self.internal_initialize_displacements()
    #         for idv in range(len(self.list_full_subd_elements)):
    #             self.dict_points_local_to_global = dict(zip(range(len(self.list_subd_points[idv])), self.list_subd_points[idv]))
    #             self.dict_points_global_to_local = {v: k for k, v in self.dict_points_local_to_global.items()}
                
    #             self.K = self.K_array[idv].copy()
    #             self.F = np.zeros(self.list_subd_points_coords[idv].size)

    #             self.set_condition_dirichlet_sub(idv)
    #             self.set_condition_neumann_sub(idv)
    #             self.set_condition_schwarz_sub(idv, self.u_current)

    #             [*arg,] = self.solve_function(self.K.tocsr(), self.F)
    #             u_subd = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

    #             for x in list(self.dict_points_local_to_global.keys()):
    #                 self.u_current[self.dict_points_local_to_global[x], :] = np.copy(u_subd[x, :])

    #             self.internal_additional_calculations()

    #         self.amnt_iterations += 1
    #         self.interal_final_calculate_u()

    #         crit_convergence = self.calculate_crit_convergence()

    #         if crit_convergence < self.coef_convergence:
    #             break


if __name__ == "__main__":
    pass
