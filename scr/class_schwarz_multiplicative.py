import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import time

import numpy as np
import copy

import scr.functions as base_func
from scr._class_template import class_template


class schwarz_multiplicative(class_template):
    def __init__(self, data):
        init_time = time.time()
        super().__init__(data)

        self.name_method = "schwarz_multiplicative"
        self.table_name = '$\\text{Мультипликативный МДО}$'
        self.cur_amnt_subds = data['amnt_subds']
        self.coef_convergence = data['coef_convergence']
        self.coef_overlap = data['coef_overlap']

        self.time_init = time.time() - init_time
    

    def init_subd_params(self):
        self.dict_subd_boundary_points = {}
        self.dict_subd_elements = {}

        list_values = np.array([0])
        
        if self.data["fine_area"] == 'rectangle':
            for value in range(1, self.cur_amnt_subds):
                result = np.array([value - self.coef_overlap, value + self.coef_overlap]) * (self.contour_points[1, 0] - self.contour_points[0, 0]) / (self.cur_amnt_subds)
                list_values = np.append(list_values, result)
            list_values = np.append(list_values, 2)
            list_center_points = self.element_centroid_points_coords[:, 0]
        else:
            for value in range(1, self.cur_amnt_subds):
                result = np.array([value - self.coef_overlap, value + self.coef_overlap]) * np.pi / (2 * self.cur_amnt_subds)
                list_values = np.append(list_values, result)
            list_values = np.append(list_values, np.pi / 2 )
            list_center_points = self.element_centroid_points_angles

        for i in range(self.cur_amnt_subds):
            if i == 0:
                self.dict_subd_elements[i] = np.where((list_values[i] < list_center_points) & (list_center_points < list_values[i + 2]))[0]
            elif i == self.cur_amnt_subds - 1:
                self.dict_subd_elements[i] = np.where((list_values[-3] < list_center_points) & (list_center_points < list_values[-1]))[0]
            else:
                self.dict_subd_elements[i] = np.where((list_values[i * 2 - 1] < list_center_points) & (list_center_points < list_values[i * 2 + 2]))[0]

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
        self.F_array = []
        self.dict_subd_points_local_to_global = {}
        self.dict_subd_diriclet_points = {}
        self.dict_subd_neumann_elements = {}
        
        for idv in range(self.cur_amnt_subds):
            self.dict_subd_points_local_to_global[idv] = dict(zip(range(len(self.dict_subd_points[idv])), self.dict_subd_points[idv]))
            dict_points_global_to_local = {v: k for k, v in self.dict_subd_points_local_to_global[idv].items()}

            self.dict_subd_diriclet_points[idv] = np.intersect1d(np.array(list(self.dict_area_dirichlet_points.keys())), self.dict_subd_points[idv])
            temp = []
            for element in self.dict_subd_elements[idv]:
                if len(set(self.area_elements[element]) & set(self.dict_area_neumann_points.keys())) == 2:
                    temp.append(list(set(self.area_elements[element]) & set(self.dict_area_neumann_points.keys())))
            self.dict_subd_neumann_elements[idv] = temp

            init_time = time.time()
            temp_K = base_func.calculate_sparse_matrix_stiffness(self.area_elements[self.dict_subd_elements[idv]], self.area_points_coords, self.dict_subd_points[idv].size, self.D, self.dim_task, dict_points_global_to_local)
            self.time9 += time.time() - init_time
            temp_F = np.zeros(self.dict_subd_points[idv].size * self.dim_task)

            self.set_condition_neumann(temp_F, self.dict_subd_neumann_elements[idv], self.area_points_coords, self.dict_area_neumann_points, dict_points_global_to_local)
            self.set_condition_dirichlet(temp_K, temp_F, self.dict_area_dirichlet_points, self.dict_subd_diriclet_points[idv], dict_points_global_to_local)

            self.K_array.append(temp_K)
            self.F_array.append(temp_F)


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
        init_time = time.time()
        self.init_subd_params()

        lst_iters_cg = 0
        lst_time_cg = 0

        self.amnt_iterations = 0
        self.cur_crit_convergence = 1
        self.u = np.zeros((self.area_points_coords.shape[0], 2))
        self.time1 = time.time() - init_time
        while True:
            self.set_initialization()
            for idv in range(self.cur_amnt_subds):
                init_time = time.time()
                dict_points_global_to_local = {v: k for k, v in self.dict_subd_points_local_to_global[idv].items()}
                
                K = self.K_array[idv].copy()
                F = self.F_array[idv].copy()
                self.time2 += time.time() - init_time

                init_time = time.time()
                function_condition_schwarz = self.get_condition_schwarz(K, F, self.dict_subd_boundary_points[idv], dict_points_global_to_local)
                self.set_condition_schwarz(function_condition_schwarz)
                self.time3 += time.time() - init_time

                init_time = time.time()
                init_u = self.u_current[np.array(list(self.dict_subd_points_local_to_global[idv].values()))].reshape(-1)
                u_subd, amnt_iters_cg, amnt_time_cg = self.conjugate_method(K.tocsr(), F, self.cur_crit_convergence, init_u)
                lst_iters_cg += amnt_iters_cg
                lst_time_cg += amnt_time_cg
                

                self.u_current[np.array(list(self.dict_subd_points_local_to_global[idv].values()))] = u_subd.reshape(-1, 2)

                self.set_additional_calculations()
                self.time4 += time.time() - init_time

            init_time = time.time()

            self.amnt_iterations += 1
            self.get_displacements()

            self.time5 += time.time() - init_time

            init_time = time.time()
            crit_convergence = self.calculate_error(self.u, self.u_previous, 'point')
            self.time6 += time.time() - init_time
            print(f"{crit_convergence:.3e}", end = "\r")
            if crit_convergence < self.coef_convergence:
                print()
                break
            self.cur_crit_convergence = crit_convergence



if __name__ == "__main__":
    pass
