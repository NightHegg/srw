import os
import sys
from numpy.core.defchararray import index

from numpy.core.numeric import outer
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import time

import numpy as np
from scipy.sparse import linalg
import matplotlib.pyplot as plt

import scr.functions as base_func
from scr._template_class import class_template


class basic_method(class_template):
    def __init__(self, data):
        init_time = time.time()
        super().__init__(data)
        self.name_method = "basic_method"
        self.time_init = time.time() - init_time

    def calculate_u(self):
        K = base_func.calculate_sparse_matrix_stiffness(self.area_elements, self.area_points_coords, self.area_points.size, self.D, self.dim_task)
        F = np.zeros(self.area_points_coords.size)

        self.set_condition_neumann(F, self.list_area_neumann_elements, self.area_points_coords, self.dict_area_neumann_points)
        self.set_condition_dirichlet(K, F, self.dict_area_dirichlet_points, self.dict_area_dirichlet_points.keys())
        
        *arg, = self.solve_function(K.tocsr(), F)
        self.u = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))


    def plot_disp_strain_graphs(self):
        element_centroid_points_coords = np.array(list(map(lambda x: np.mean(self.area_points_coords_modified[x], axis = 0), self.area_elements)))
        points = self.area_points[np.isclose(self.area_points_coords[:, 1], 0)]
        # points = self.area_points[self.inner_radius_points]
        # points = self.area_points[self.outer_radius_points]
        sorted_points = points[np.argsort(self.area_points_coords[points, 0])]

        elements = np.array(list(filter(lambda x: np.intersect1d(self.area_elements[x], points).size == 2, range(self.area_elements.shape[0]))))
        sorted_elements = elements[np.argsort(element_centroid_points_coords[elements, 0])]

        fig, ax = plt.subplots(1, 2, figsize = (12, 7))

        ax[0].plot(self.area_points_coords[sorted_points, 0], self.u_modified_exact[sorted_points], label = 'exact_u_r')
        ax[0].plot(self.area_points_coords[sorted_points, 0], self.u_polar[sorted_points, 0], label = 'num_u_r')

        ax[1].plot(self.area_points_coords[sorted_points, 0], self.sigma_exact[sorted_points, 0] * self.coef_sigma, label = 'exact_sigma_r')
        ax[1].plot(self.area_points_coords[sorted_points, 0], self.sigma_exact[sorted_points, 1] * self.coef_sigma, label = 'exact_sigma_phi')

        ax[1].plot(self.area_points_coords[sorted_points, 0], self.sigma_polar[sorted_points, 0] * self.coef_sigma, label = 'num_sigma_r')
        ax[1].plot(self.area_points_coords[sorted_points, 0], self.sigma_polar[sorted_points, 1] * self.coef_sigma, label = 'num_sigma_phi')

        ax[1].plot(self.area_points_coords[sorted_points, 0], np.ones_like(sorted_points) * self.inner_pressure * self.coef_sigma, '--')
        ax[1].plot(self.area_points_coords[sorted_points, 0], np.ones_like(sorted_points) * self.outer_pressure * self.coef_sigma, '--')

        ax[0].set_title('Перемещения')
        ax[1].set_title('Напряжения')
        
        ax[0].legend(fontsize = 15, facecolor = 'oldlace')
        ax[1].legend(fontsize = 15, facecolor = 'oldlace')

        plt.show()

if __name__ == "__main__":
    pass