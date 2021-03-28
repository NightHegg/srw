import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import math

from itertools import combinations
import numpy as np
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import time
import scipy

import scr.functions as base_func
from scr.class_basic_method import basic_method


class schwarz_multiplicative(basic_method):
    def __init__(self, data):
        init_time = time.time()
        super().__init__(data)

        self.name_method = "schwarz multiplicative method"
        self.cur_amnt_subds = data['amnt_subds']
        self.coef_convergence = data['coef_convergence']
        self.coef_overlap = data['coef_overlap']
        self.time_init = time.time() - init_time


    def init_subd_params(self):
        *arg, = base_func.calculate_subd_parameters(self.contour_points, self.area_points_coords, self.area_elements, self.coef_overlap, self.cur_amnt_subds)

        self.list_full_subd_elements           = arg[0]
        self.list_subd_elements                = arg[1]
        self.list_subd_points                  = arg[2]
        self.list_subd_points_coords           = arg[3]
        self.subd_boundary_overlap_points      = arg[4]
        self.dict_elements_contain_point       = arg[5]

        self.K_array = []
        for idv, subd in enumerate(self.list_full_subd_elements):
            dict_points_local_to_global = dict(zip(range(len(self.list_subd_points[idv])), self.list_subd_points[idv]))
            dict_points_global_to_local = {v: k for k, v in dict_points_local_to_global.items()}

            subd_elements_local = np.array([dict_points_global_to_local[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)
            self.K_array.append(base_func.calculate_sparse_matrix_stiffness(subd_elements_local, self.list_subd_points_coords[idv], self.D, self.dim_task))


    def internal_initialize_displacements(self):
        self.u_previous = np.copy(self.u)
        self.u_current = np.copy(self.u)


    def get_condition_schwarz(self, K, F, list_schwarz_points, modifier):
        def temp(array_condition):
            for point in list_schwarz_points:
                modified_point = modifier[point]
                for cur_dim in range(self.dim_task):
                    value = array_condition[point, cur_dim]
                    indices = np.array(K.rows[modified_point * self.dim_task + cur_dim])

                    F[indices] -= np.array(K.data[modified_point * self.dim_task + cur_dim]) * value
                    for index in indices:
                        K[modified_point * self.dim_task + cur_dim, index] = 0
                    for index in indices:
                        K[index, modified_point * self.dim_task + cur_dim] = 0
                    K[modified_point * self.dim_task + cur_dim, modified_point * self.dim_task + cur_dim] = 1
                    F[modified_point * self.dim_task + cur_dim] = value
        return temp


    def set_condition_schwarz(self, function_condition_schwarz):
        function_condition_schwarz(self.u_current)


    def internal_additional_calculations(self):
        pass


    def interal_final_calculate_u(self):
        self.u = np.copy(self.u_current)


    def calculate_crit_convergence(self):
        dict_u = {idx : value for idx, value in enumerate(self.u) if not np.all((np.isclose(value, np.zeros_like(value))))}

        divisible, divisor, relative_error = 0, 0, 0
        for idx, value in dict_u.items():

            relative_error = np.linalg.norm(value - self.u_previous[idx])**2 / np.linalg.norm(value)**2

            sum_elements = [self.list_sum_elements[cur_element] for cur_element in self.dict_elements_contain_point[idx]]
            divisible += (sum(sum_elements) / 3) * relative_error
            divisor += (sum(sum_elements) / 3)
        return math.sqrt(divisible / divisor)


    def calculate_u(self):
        self.time_final_u_calculations = 0
        self.time_final_1 = 0
        self.time_final_2 = 0
        self.time_final_3 = 0
        self.time_final_4 = 0

        self.amnt_iterations = 0
        self.u = np.zeros((self.area_points_coords.shape[0], 2))

        self.init_subd_params()
        list_subd_neumann_points = []
        list_subd_schwarz_points = []
        list_subd_dirichlet_points = []
          
        for idv in range(len(self.list_full_subd_elements)):          
            temp = []
            for element in self.list_subd_elements[idv]:
                if len(set(self.area_elements[element]) & set(self.dict_area_neumann_points.keys())) == 2:
                    temp.append(self.area_elements[element])
            list_subd_neumann_points.append(temp)

            list_subd_schwarz_points.append(sum([list(set(self.subd_boundary_overlap_points[idv]) & set(subd)) for idx, subd in enumerate(self.list_subd_points) if idx != idv], []))
            list_subd_dirichlet_points.append(list(set(self.dict_area_dirichlet_points.keys()) & set(self.list_subd_points[idv])))

        while True:
            self.internal_initialize_displacements()
            for idv in range(len(self.list_full_subd_elements)):
                self.dict_points_local_to_global = dict(zip(range(len(self.list_subd_points[idv])), self.list_subd_points[idv]))
                self.dict_points_global_to_local = {v: k for k, v in self.dict_points_local_to_global.items()}
                
                K = self.K_array[idv].copy()
                F = np.zeros(self.list_subd_points_coords[idv].size)
                
                self.set_condition_dirichlet(K, F, list_subd_dirichlet_points[idv], self.dict_points_global_to_local)
                self.set_condition_neumann(F, list_subd_neumann_points[idv], self.dict_points_global_to_local)
                
                function_condition_schwarz = self.get_condition_schwarz(K, F, list_subd_schwarz_points[idv], self.dict_points_global_to_local)
                self.set_condition_schwarz(function_condition_schwarz)

                [*arg,] = self.solve_function(K.tocsr(), F)
                u_subd = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

                for x in list(self.dict_points_local_to_global.keys()):
                    self.u_current[self.dict_points_local_to_global[x], :] = np.copy(u_subd[x, :])
                self.internal_additional_calculations()

            self.amnt_iterations += 1
            self.interal_final_calculate_u()

            crit_convergence = self.calculate_crit_convergence()
            # print(f"{crit_convergence:.3e}", end = "\r")
            if crit_convergence < self.coef_convergence:
                break


    def plot_subds(self):
        self.init_subd_params()

        nrows = math.ceil(len(self.list_subd_elements) / 2)
        fig, ax = plt.subplots(nrows, ncols = 2)
        for num, subd in enumerate(self.list_full_subd_elements):
            ax_spec = ax[num] if nrows == 1 else ax[num//2, num % 2]

            # listPoints_Schwarz = sum([list(set(self.subd_boundary_overlap_points[num]) & set(subd)) for idx, subd in enumerate(self.subd_points) if idx != num], [])
            ax_spec.plot(self.area_points_coords[self.subd_boundary_overlap_points[num], 0], self.area_points_coords[self.subd_boundary_overlap_points[num], 1], marker = "o", markersize = 15, linewidth = 0)
            # ax_spec.plot(self.area_points_coords[listPoints_Schwarz, 0], self.area_points_coords[listPoints_Schwarz, 1], marker = "X", markersize = 15, linewidth = 0)
            ax_spec.plot(self.contour_points[:, 0], self.contour_points[:, 1], color = "brown")
            ax_spec.triplot(self.area_points_coords[:,0], self.area_points_coords[:,1], subd.copy())
            ax_spec.set(title = f"Подобласть №{num}")

        fig.set_figwidth(15)
        fig.set_figheight(nrows * 4)
        fig.set_facecolor('mintcream')

        plt.show()


    def construct_info(self):
        super().construct_info()
        self.message['iterations'] = [
            f'Amount of iterations: {self.amnt_iterations}\n',
            f'{"-" * 5}\n'
        ]


if __name__ == "__main__":
    pass
