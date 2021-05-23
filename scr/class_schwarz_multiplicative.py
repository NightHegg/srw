import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import math

import numpy as np
import time

import scr.functions as base_func
from scr._template_class import class_template


class schwarz_multiplicative(class_template):
    def __init__(self, data):
        init_time = time.time()
        super().__init__(data)

        self.name_method = "schwarz_multiplicative"
        self.cur_amnt_subds = data['amnt_subds']
        self.coef_convergence = data['coef_convergence']
        self.coef_overlap = data['coef_overlap']

        self.time_init = time.time() - init_time
    

    def init_subd_params(self):
        self.dict_subd_boundary_points = {}

        list_angles = np.array([0])
        for value in range(1, self.cur_amnt_subds):
            result = np.array([value - self.coef_overlap, value + self.coef_overlap]) * np.pi / (2 * self.cur_amnt_subds)
            list_angles = np.append(list_angles, result)
        list_angles = np.append(list_angles, np.pi / 2 )
        self.dict_subd_elements = {}
        for i in range(self.cur_amnt_subds):
            if i == 0:
                self.dict_subd_elements[i] = np.where((list_angles[i] < self.element_centroid_points_angles) & (self.element_centroid_points_angles < list_angles[i + 2]))[0]
            elif i == self.cur_amnt_subds - 1:
                self.dict_subd_elements[i] = np.where((list_angles[-3] < self.element_centroid_points_angles) & (self.element_centroid_points_angles < list_angles[-1]))[0]
            else:
                self.dict_subd_elements[i] = np.where((list_angles[i * 2 - 1] < self.element_centroid_points_angles) & (self.element_centroid_points_angles < list_angles[i * 2 + 2]))[0]
                
        
        # for value in self.dict_subd_elements.values():
        #     self.internal_plot_displacements(self.area_points_coords, self.area_elements[value])

        for key, value in self.dict_subd_elements.items():
            dict_subd_elements_contain_point = {}
            for element in np.ravel(value):
                for point in self.area_elements[element]:
                    if point in dict_subd_elements_contain_point.keys():
                        dict_subd_elements_contain_point[point].append(element)
                    else:
                        dict_subd_elements_contain_point[point] = [element]
            t = []
            for point, elements in dict_subd_elements_contain_point.items():
                if len(elements) != self.dict_elements_contain_point[point].size:
                    t.append(point)
            self.dict_subd_boundary_points[key] = t

        self.dict_subd_points = {i: np.unique(self.area_points[self.area_elements[self.dict_subd_elements[i]]]) for i in range(self.cur_amnt_subds)}
        
        self.K_array = []
        self.dict_subd_points_local_to_global = {}
        self.dict_subd_diriclet_points = {}
        self.dict_subd_neumann_elements = {}
        for idv in range(self.cur_amnt_subds):
            self.dict_subd_points_local_to_global[idv] = dict(zip(range(len(self.dict_subd_points[idv])), self.dict_subd_points[idv]))
            dict_points_global_to_local = {v: k for k, v in self.dict_subd_points_local_to_global[idv].items()}

            self.K_array.append(base_func.calculate_sparse_matrix_stiffness(self.area_elements[self.dict_subd_elements[idv]], self.area_points_coords, self.dict_subd_points[idv].size, self.D, self.dim_task, dict_points_global_to_local))
            
            self.dict_subd_diriclet_points[idv] = np.intersect1d(np.array(list(self.dict_area_dirichlet_points.keys())), self.dict_subd_points[idv])
            temp = []
            for element in self.dict_subd_elements[idv]:
                if len(set(self.area_elements[element]) & set(self.dict_area_neumann_points.keys())) == 2:
                    temp.append(list(set(self.area_elements[element]) & set(self.dict_area_neumann_points.keys())))
            self.dict_subd_neumann_elements[idv] = temp


    def get_condition_schwarz(self, K, F, list_schwarz_points, modifier):
        def temp(array_condition):
            for point in list_schwarz_points:
                modified_point = modifier[point] if modifier else point
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
        function_condition_schwarz(self.u_current.copy())


    def set_initialization(self):
        self.u_previous = self.u.copy()
        self.u_current = self.u.copy()


    def set_additional_calculations(self):
        pass


    def get_displacements(self):
        self.u = self.u_current.copy()


    def calculate_u(self):
        self.init_subd_params()

        self.amnt_iterations = 0
        self.u = np.zeros((self.area_points_coords.shape[0], 2))
        while True:
            self.set_initialization()
            for idv in range(self.cur_amnt_subds):
                dict_points_global_to_local = {v: k for k, v in self.dict_subd_points_local_to_global[idv].items()}
                
                K = self.K_array[idv].copy()
                F = np.zeros(self.dict_subd_points[idv].size * self.dim_task)

                self.set_condition_neumann(F, self.dict_subd_neumann_elements[idv], self.area_points_coords, self.dict_area_neumann_points, dict_points_global_to_local)
                self.set_condition_dirichlet(K, F, self.dict_area_dirichlet_points, self.dict_subd_diriclet_points[idv], dict_points_global_to_local)

                function_condition_schwarz = self.get_condition_schwarz(K, F, self.dict_subd_boundary_points[idv], dict_points_global_to_local)
                self.set_condition_schwarz(function_condition_schwarz)

                # [*arg,] = self.solve_function(K.tocsr(), F)
                # u_subd = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))
                u_subd = self.conjugate_method(K, F).reshape(-1, 2)

                self.u_current[np.array(list(self.dict_subd_points_local_to_global[idv].values()))] = u_subd.copy()
                self.set_additional_calculations()

            self.amnt_iterations += 1
            self.get_displacements()

            crit_convergence = self.calculate_error(self.u, self.u_previous, 'point')
            print(f"{crit_convergence:.3e}", end = "\r")
            if crit_convergence < self.coef_convergence:
                break


    def construct_info(self):
        super().construct_info()
        self.message['iterations'] = [
            f'Amount of iterations: {self.amnt_iterations}\n',
            f'{"-" * 5}\n'
        ]


if __name__ == "__main__":
    pass
