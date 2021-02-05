import os
import sys
import math

from scipy.sparse.linalg import dsolve
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from scipy.sparse import linalg

from modules.operations_input_files.read_input_files import read_mesh
import modules.basic_functions as base_func
from modules.class_schwarz_additive import schwarz_additive

class schwarz_two_level_additive(schwarz_additive):
    # TODO: Можно ли area_limits тоже сделать атрибутом класса?
    def __init__(self, cur_task, cur_mesh, cur_coarse_mesh = 10, cur_amnt_subds = [2, 1], coef_convergence = 1e-3, solve_function = linalg.spsolve, coef_alpha = 0.5):
        super().__init__(cur_task, cur_mesh, cur_amnt_subds, coef_convergence, solve_function, coef_alpha)

        _, self.area_points_coords_coarse, self.area_elements_coarse = read_mesh(cur_coarse_mesh)

        area_limits = lambda cond, type: [self.area_bounds[cond[0], type], self.area_bounds[cond[0] + 1, type]]

        self.dirichlet_points_coarse = {tuple(idx for idx, val in enumerate(self.area_points_coords_coarse) 
                    if min(area_limits(cond, 0)) <= val[0] <= max(area_limits(cond, 0)) and
                       min(area_limits(cond, 1)) <= val[1] <= max(area_limits(cond, 1))) : cond[1:] for cond in self.dirichlet_conditions}

    def init_subd_params(self):
        super().init_subd_params()

        *arg, = base_func.calculate_subd_parameters(self.area_bounds, self.area_points_coords_coarse, self.area_elements_coarse, self.coef_overlap, self.cur_amnt_subds)

        self.subd_elements_coarse                = arg[0]
        self.subd_points_coarse                  = arg[1]
        self.subd_points_coords_coarse           = arg[2]
        self.subd_boundary_overlap_points_coarse = arg[3]
        self.relation_points_elements_coarse     = arg[4]

        self.K_array_coarse = []
        for idv, subd in enumerate(self.subd_elements_coarse):
            ratioPoints_LocalGlobal = dict(zip(range(len(self.subd_points_coarse[idv])), self.subd_points_coarse[idv]))
            ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

            subd_elements_local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)
            self.K_array_coarse.append(base_func.calculate_sparse_matrix_stiffness(subd_elements_local, self.subd_points_coords_coarse[idv], self.D, self.dimTask))

    def calculate_u(self):
        self.init_subd_params()

        amnt_iterations = 0
        self.u = np.zeros((self.area_points_coords.shape[0], 2))
        while True:
            u_previous = np.copy(self.u)
            self.u = np.zeros_like(u_previous)

            u_current_temp = np.copy(u_previous)
            u_sum = np.zeros_like(self.u)
            u_special = np.ravel(np.zeros_like(self.u))
            for idv, subd in enumerate(self.subd_elements):
                ratioPoints_LocalGlobal = dict(zip(range(len(self.subd_points[idv])), self.subd_points[idv]))
                ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

                ratioPoints_LocalGlobal_coarse = dict(zip(range(len(self.subd_points_coarse[idv])), self.subd_points_coarse[idv]))
                ratioPoints_GlobalLocal_coarse = {v: k for k, v in ratioPoints_LocalGlobal_coarse.items()}

                K = self.K_array[idv].copy()
                F = np.zeros(self.subd_points_coords[idv].size)

                K_coarse = self.K_array_coarse[idv].copy()
                F_coarse = np.zeros(self.subd_points_coords_coarse[idv].size)
                
                for list_points, condition in self.dirichlet_points.items():
                    listPoints = list(set(list_points) & set(self.subd_points[idv]))
                    for node in listPoints:
                        if condition[0] == 2:
                            K, F = base_func.bound_condition_dirichlet(K, F, self.dimTask, ratioPoints_GlobalLocal[node], condition[1], 0)
                            K, F = base_func.bound_condition_dirichlet(K, F, self.dimTask, ratioPoints_GlobalLocal[node], condition[2], 1)
                        else:
                            K, F = base_func.bound_condition_dirichlet(K, F, self.dimTask, ratioPoints_GlobalLocal[node], condition[1], condition[0])

                for list_points, condition in self.dirichlet_points_coarse.items():
                    listPoints = list(set(list_points) & set(self.subd_points_coarse[idv]))
                    for node in listPoints:
                        if condition[0] == 2:
                            K_coarse, F_coarse = base_func.bound_condition_dirichlet(K_coarse, F_coarse, self.dimTask, ratioPoints_GlobalLocal_coarse[node], condition[1], 0)
                            K_coarse, F_coarse = base_func.bound_condition_dirichlet(K_coarse, F_coarse, self.dimTask, ratioPoints_GlobalLocal_coarse[node], condition[2], 1)
                        else:
                            K_coarse, F_coarse = base_func.bound_condition_dirichlet(K_coarse, F_coarse, self.dimTask, ratioPoints_GlobalLocal_coarse[node], condition[1], condition[0])

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

                listPoints_Schwarz_coarse = sum([list(set(self.subd_boundary_overlap_points_coarse[idv]) & set(subd)) for idx, subd in enumerate(self.subd_points_coarse) if idx != idv], [])

                for node in listPoints_Schwarz_coarse:
                    for dim in range(self.dimTask):
                        K_coarse, F_coarse = base_func.bound_condition_dirichlet(K_coarse, F_coarse, self.dimTask, ratioPoints_GlobalLocal_coarse[node], u_previous[node, dim], dim)


                residual = F - np.dot(K.toarray(), np.ravel(u_previous[self.subd_points[idv]]))

                chosen_element = []
                for point in self.subd_points_coarse[idv]:
                    chosen_element = []
                    for element in self.subd_elements[idv]:
                        S_element = base_func.calculate_element_area(self.area_points_coords[element[0]], self.area_points_coords[element[1]], self.area_points_coords[element[2]])
                        S_1 = base_func.calculate_element_area(self.area_points_coords_coarse[point], self.area_points_coords[element[1]], self.area_points_coords[element[2]])
                        S_2 = base_func.calculate_element_area(self.area_points_coords[element[0]], self.area_points_coords_coarse[point], self.area_points_coords[element[2]])
                        S_3 = base_func.calculate_element_area(self.area_points_coords[element[0]], self.area_points_coords[element[1]], self.area_points_coords_coarse[point])
                        if S_1 + S_2 + S_3 - S_element < 1e-9:
                            chosen_element = element
                            break
                    
                    function = base_func.calculate_local_functions(chosen_element, self.area_points_coords)

                    for i in range(3):
                        temp = lambda point: ratioPoints_GlobalLocal_coarse[point] * self.dimTask
                        temp_2 = lambda point: ratioPoints_GlobalLocal[point] * self.dimTask
                        F_coarse[temp(point)] += function(self.area_points_coords_coarse[point], i) * residual[temp_2(chosen_element[i])]
                        F_coarse[temp(point) + 1] += function(self.area_points_coords_coarse[point], i) * residual[temp_2(chosen_element[i]) + 1]
                
                u_coarse = np.array(self.solve_function(K_coarse.tocsr(), F_coarse))

                chosen_element = []
                for point in self.subd_points[idv]:
                    chosen_element = []
                    for element in self.subd_elements_coarse[idv]:
                        S_element = base_func.calculate_element_area(self.area_points_coords_coarse[element[0]], self.area_points_coords_coarse[element[1]], self.area_points_coords_coarse[element[2]])
                        S_1 = base_func.calculate_element_area(self.area_points_coords[point], self.area_points_coords_coarse[element[1]], self.area_points_coords_coarse[element[2]])
                        S_2 = base_func.calculate_element_area(self.area_points_coords_coarse[element[0]], self.area_points_coords[point], self.area_points_coords_coarse[element[2]])
                        S_3 = base_func.calculate_element_area(self.area_points_coords_coarse[element[0]], self.area_points_coords_coarse[element[1]], self.area_points_coords[point])
                        if S_1 + S_2 + S_3 - S_element < 1e-9:
                            chosen_element = element
                            break

                    function = base_func.calculate_local_functions(chosen_element, self.area_points_coords_coarse)

                    u_special[point * self.dimTask] = 0
                    u_special[point * self.dimTask + 1] = 0
                    for i in range(3):
                        temp = lambda point: ratioPoints_GlobalLocal_coarse[point] * self.dimTask
                        u_special[point * self.dimTask] += function(self.area_points_coords[point], i) * u_coarse[temp(chosen_element[i])]
                        u_special[point * self.dimTask + 1] += function(self.area_points_coords[point], i) * u_coarse[temp(chosen_element[i]) + 1]

                [*arg,] = self.solve_function(K.tocsr(), F)
                u_subd = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

                for x in list(ratioPoints_LocalGlobal.keys()):
                    u_current_temp[ratioPoints_LocalGlobal[x], :] = np.copy(u_subd[x, :])

                u_sum += (u_current_temp - u_previous)

            amnt_iterations += 1

            self.u = np.copy(u_previous) + (self.coef_alpha * u_sum) + (self.coef_alpha * u_special.reshape(-1, 2))

            crit_convergence = base_func.calculate_crit_convergence(self.u, u_previous, self.area_points_coords, self.dimTask, self.relation_points_elements, self.coef_u)
            if crit_convergence < self.coef_convergence or amnt_iterations > 0:
                break

    def plot_subds(self):
            self.init_subd_params()

            nrows = math.ceil(len(self.subd_elements) / 2)
            fig, ax = plt.subplots(nrows, ncols = 2)
            for num, subd in enumerate(self.subd_elements):
                ax_spec = ax[num] if nrows == 1 else ax[num//2, num % 2]

                listPoints_Schwarz = sum([list(set(self.subd_boundary_overlap_points[num]) & set(subd)) for idx, subd in enumerate(self.subd_points) if idx != num], [])
                listPoints_Schwarz_coarse = sum([list(set(self.subd_boundary_overlap_points_coarse[num]) & set(subd)) for idx, subd in enumerate(self.subd_points_coarse) if idx != num], [])

                ax_spec.plot(self.area_points_coords[listPoints_Schwarz, 0], self.area_points_coords[listPoints_Schwarz, 1], marker = "X", markersize = 15, linewidth = 0, label = "Мелкая сетка")
                ax_spec.plot(self.area_points_coords_coarse[listPoints_Schwarz_coarse, 0], self.area_points_coords_coarse[listPoints_Schwarz_coarse, 1], marker = "X", markersize = 15, linewidth = 0, label = "Грубая сетка")

                ax_spec.plot(self.area_bounds[:, 0], self.area_bounds[:, 1], color = "brown")

                ax_spec.triplot(self.area_points_coords_coarse[:,0], self.area_points_coords_coarse[:,1], self.subd_elements_coarse[num].copy())
                ax_spec.triplot(self.area_points_coords[:,0], self.area_points_coords[:,1], subd.copy(), alpha = 0.5)

                ax_spec.set(title = f"Подобласть №{num}")
                ax_spec.legend(fontsize = 9)

            fig.set_figwidth(15)
            fig.set_figheight(nrows * 4)
            fig.set_facecolor('mintcream')

            plt.show()


if __name__ == "__main__":
    pass
