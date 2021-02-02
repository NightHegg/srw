from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix, linalg

import basic_functions as base_func
from class_basic_method import basic_method


class schwarz_multiplicative(basic_method):
    def __init__(self, cur_task, cur_mesh, cur_amnt_subds, coef_convergence, solve_function = linalg.spsolve):
        super().__init__(cur_task, cur_mesh, solve_function)
        self.__cur_amnt_subds = cur_amnt_subds
        self.__coef_convergence = coef_convergence

    def init_subd_params(self):
        *arg, = base_func.calculate_subd_parameters(self.area_bounds, self.area_points_coords, self.area_elements, self.coef_overlap, self.__cur_amnt_subds)

        self.subd_elements      = arg[0]
        self.subd_points        = arg[1]
        self.subd_points_coords = arg[2]

        self.K_array = []
        for idv, subd in enumerate(self.subd_elements):
            ratioPoints_LocalGlobal = dict(zip(range(len(self.subd_points[idv])), self.subd_points[idv]))
            ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

            subd_elements_local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)
            self.K_array.append(base_func.calculate_sparse_matrix_stiffness(self.subd_elements_local, self.subd_points_coords[idv], self.D, self.dimTask))


    def calculate_u(self):
        self.init_subd_params()

        amnt_iterations = 0
        self.u = np.zeros((self.area_points_coords.shape[0], 2))
        while True:
            u_previous = np.copy(self.u)
            for idv, subd in enumerate(self.subd_elements):
                ratioPoints_LocalGlobal = dict(zip(range(len(self.subd_points[idv])), self.subd_points[idv]))
                ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

                K = K_array[idv].copy()
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

                listPoints_Schwarz = sum([list(set(subd_boundary_overlap_points[idv]) & set(subd)) for idx, subd in enumerate(self.subd_points) if idx != idv], [])

                for node in listPoints_Schwarz:
                    for dim in range(self.dimTask):
                        K, F = base_func.bound_condition_dirichlet(K, F, self.dimTask, ratioPoints_GlobalLocal[node], self.u[node, dim], dim)

                [*arg,] = self.solve_function(K.tocsr(), F)
                u_subd = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

                for x in list(ratioPoints_LocalGlobal.keys()):
                    self.u[ratioPoints_LocalGlobal[x], :] = np.copy(u_subd[x, :])

            amnt_iterations += 1

            crit_convergence = base_func.calculate_crit_convergence(self.u, u_previous, self.area_points_coords, self.dimTask, relation_points_elements, self.coef_u)
            if crit_convergence < self.__coef_convergence:
                break
