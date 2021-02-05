import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from itertools import combinations
import numpy as np
from scipy.sparse import linalg

import modules.basic_functions as base_func
from modules.class_schwarz_multiplicative import schwarz_multiplicative


class schwarz_additive(schwarz_multiplicative):
    def __init__(self, cur_task, cur_mesh, cur_amnt_subds = [2, 1], coef_convergence = 1e-4, solve_function = linalg.spsolve, coef_alpha = 0.5):
        super().__init__(cur_task, cur_mesh, cur_amnt_subds, coef_convergence, solve_function)
        self.coef_alpha = coef_alpha

    def calculate_u(self):
        self.init_subd_params()

        amnt_iterations = 0
        self.u = np.zeros((self.area_points_coords.shape[0], 2))
        while True:
            u_previous = np.copy(self.u)
            self.u = np.zeros_like(u_previous)
            u_current_temp = np.copy(u_previous)
            u_sum = np.zeros_like(self.u)
            for idv, subd in enumerate(self.subd_elements):
                ratioPoints_LocalGlobal = dict(zip(range(len(self.subd_points[idv])), self.subd_points[idv]))
                ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

                K = self.K_array[idv].copy()
                F = np.zeros(self.subd_points_coords[idv].size)
                
                for list_points, condition in self.dirichlet_points.items():
                    listPoints = list(set(list_points) & set(self.subd_points[idv]))
                    for node in listPoints:
                        if condition[0] == 2:
                            K, F = base_func.bound_condition_dirichlet(K, F, self.dimTask, ratioPoints_GlobalLocal[node], condition[1], 0)
                            K, F = base_func.bound_condition_dirichlet(K, F, self.dimTask, ratioPoints_GlobalLocal[node], condition[2], 1)
                        else:
                            K, F = base_func.bound_condition_dirichlet(K, F, self.dimTask, ratioPoints_GlobalLocal[node], condition[1], condition[0])

                rpGL_Change = lambda L: [ratioPoints_GlobalLocal[x] for x in L]

                for list_points, stress in self.neumann_points.items():
                    listPoints = list(set(list_points) & set(self.subd_points[idv]))
                    segmentPoints = list(combinations(listPoints, 2))
                    list_elements = [element for element in subd for x in segmentPoints if x[0] in element and x[1] in element]
                    for element in list_elements:
                        list_neumann_points = list(set(rpGL_Change(element)) & set(rpGL_Change(listPoints))) 
                        for dim in range(self.dimTask):
                            F = base_func.bound_condition_neumann(F, list_neumann_points, self.dimTask, stress[dim], self.subd_points_coords[idv], dim)

                listPoints_Schwarz = sum([list(set(self.subd_boundary_overlap_points[idv]) & set(subd)) for idx, subd in enumerate(self.subd_points) if idx != idv], [])

                for node in listPoints_Schwarz:
                    for dim in range(self.dimTask):
                        K, F = base_func.bound_condition_dirichlet(K, F, self.dimTask, ratioPoints_GlobalLocal[node], u_previous[node, dim], dim)

                [*arg,] = self.solve_function(K.tocsr(), F)
                u_subd = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

                for x in list(ratioPoints_LocalGlobal.keys()):
                    u_current_temp[ratioPoints_LocalGlobal[x], :] = np.copy(u_subd[x, :])

                u_sum += (u_current_temp - u_previous)

            amnt_iterations += 1

            self.u = np.copy(u_previous) + (self.coef_alpha * u_sum)

            crit_convergence = base_func.calculate_crit_convergence(self.u, u_previous, self.area_points_coords, self.dimTask, self.relation_points_elements, self.coef_u)
            if crit_convergence < self.coef_convergence:
                break

if __name__ == "__main__":
    pass
