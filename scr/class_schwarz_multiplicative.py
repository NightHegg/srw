import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import math

from itertools import combinations
import numpy as np
from scipy.sparse import linalg
import matplotlib.pyplot as plt

import scr.functions as base_func
from scr.class_basic_method import basic_method


class schwarz_multiplicative(basic_method):
    def __init__(self, cur_task, cur_mesh, cur_amnt_subds = [2, 1], coef_convergence = 1e-3, solve_function = linalg.spsolve):
        super().__init__(cur_task, cur_mesh, solve_function)

        self.name_method = "schwarz multiplicative method"
        self.cur_amnt_subds = cur_amnt_subds
        self.coef_convergence = coef_convergence

        self.init_subd_params()


    def init_subd_params(self):
        *arg, = base_func.calculate_subd_parameters(self.area_bounds, self.area_points_coords, self.area_elements, self.coef_overlap, self.cur_amnt_subds)

        self.subd_elements                = arg[0]
        self.subd_points                  = arg[1]
        self.subd_points_coords           = arg[2]
        self.subd_boundary_overlap_points = arg[3]
        self.relation_points_elements     = arg[4]

        self.K_array = []
        for idv, subd in enumerate(self.subd_elements):
            ratioPoints_LocalGlobal = dict(zip(range(len(self.subd_points[idv])), self.subd_points[idv]))
            ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

            subd_elements_local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)
            self.K_array.append(base_func.calculate_sparse_matrix_stiffness(subd_elements_local, self.subd_points_coords[idv], self.D, self.dimTask))


    def internal_condition_schwarz(self, K, F, idv, ratioPoints_GlobalLocal):
        listPoints_Schwarz = sum([list(set(self.subd_boundary_overlap_points[idv]) & set(subd)) for idx, subd in enumerate(self.subd_points) if idx != idv], [])

        for node in listPoints_Schwarz:
            for dim in range(self.dimTask):
                K, F = base_func.bound_condition_dirichlet(K, F, self.dimTask, ratioPoints_GlobalLocal[node], self.u_current[node, dim], dim)
        
        return K, F


    def interal_calculate_u(self):
        self.u = np.copy(self.u_current)


    def internal_additional(self):
        pass

    
    def internal_init(self):
        self.u_previous = np.copy(self.u)
        self.u_current = np.copy(self.u)


    def calculate_u(self):
        self.amnt_iterations = 0
        self.u = np.zeros((self.area_points_coords.shape[0], 2))
        while True:
            self.internal_init()
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

                K, F = self.internal_condition_schwarz(K, F, idv, ratioPoints_GlobalLocal)

                [*arg,] = self.solve_function(K.tocsr(), F)
                u_subd = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

                for x in list(ratioPoints_LocalGlobal.keys()):
                    self.u_current[ratioPoints_LocalGlobal[x], :] = np.copy(u_subd[x, :])

                self.internal_additional()

            self.amnt_iterations += 1
            
            self.interal_calculate_u()

            crit_convergence = base_func.calculate_crit_convergence(self.u, self.u_previous, self.area_points_coords, self.dimTask, self.relation_points_elements, self.coef_u)
            print(f"{crit_convergence:.3e}", end = "\r")
            if crit_convergence < self.coef_convergence:
                break


    def plot_subds(self):
        self.init_subd_params()

        nrows = math.ceil(len(self.subd_elements) / 2)
        fig, ax = plt.subplots(nrows, ncols = 2)
        for num, subd in enumerate(self.subd_elements):
            ax_spec = ax[num] if nrows == 1 else ax[num//2, num % 2]

            listPoints_Schwarz = sum([list(set(self.subd_boundary_overlap_points[num]) & set(subd)) for idx, subd in enumerate(self.subd_points) if idx != num], [])
            ax_spec.plot(self.area_points_coords[listPoints_Schwarz, 0], self.area_points_coords[listPoints_Schwarz, 1], marker = "X", markersize = 15, linewidth = 0)
            ax_spec.plot(self.area_bounds[:, 0], self.area_bounds[:, 1], color = "brown")
            ax_spec.triplot(self.area_points_coords[:,0], self.area_points_coords[:,1], subd.copy())
            ax_spec.set(title = f"Подобласть №{num}")

        fig.set_figwidth(15)
        fig.set_figheight(nrows * 4)
        fig.set_facecolor('mintcream')

        plt.show()


    def get_info(self):
        message = (f"Method: {self.name_method}\n"
                   f"Time of execution: {self.time_execution}\n"
                   f"Minimal difference for stress: {abs(abs(min(self.Sigma[1])) - 2e+7):.2e}\n"
                   f"Maximal difference for stress: {abs(abs(max(self.Sigma[1])) - 2e+7):.2e}\n"
                   f"Amount of iterations: {self.amnt_iterations}")
        return message


if __name__ == "__main__":
    pass
