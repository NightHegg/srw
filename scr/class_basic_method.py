import os
import sys
import time

from numpy.core.fromnumeric import sort
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np

import scr.functions as base_func
from scr._class_template import class_template


class basic_method(class_template):
    def __init__(self, data):
        init_time = time.time()
        super().__init__(data)
        self.name_method = "basic_method"
        self.time_init = time.time() - init_time


    def calculate_u(self):
        init_time = time.time()
        
        K = base_func.calculate_sparse_matrix_stiffness(self.area_elements, self.area_points_coords, self.area_points.size, self.D, self.dim_task)
        F = np.zeros(self.area_points_coords.size)

        self.set_condition_neumann(F, self.list_area_neumann_elements, self.area_points_coords, self.dict_area_neumann_points)
        self.set_condition_dirichlet(K, F, self.dict_area_dirichlet_points, self.dict_area_dirichlet_points.keys())

        result, amnt_iters_cg, self.time_cg = self.conjugate_method(K.tocsr(), F)
        self.u = result.reshape(-1, 2)

        self.time_test = time.time() - init_time
        
        self.amnt_iters_cg = amnt_iters_cg


if __name__ == "__main__":
    pass