import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import meshio

import scr.functions as base_func
from scr.class_schwarz_additive import schwarz_additive

def calculate_element_area(p1, p2, p3):
    return abs((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])) * (0.5)


def check_point_in_elements(point_coords):
    def temp(area, element, area_points_coords):
        S_element = area
        S_1 = calculate_element_area(point_coords, area_points_coords[element[1]], area_points_coords[element[2]])
        S_2 = calculate_element_area(area_points_coords[element[0]], point_coords, area_points_coords[element[2]])
        S_3 = calculate_element_area(area_points_coords[element[0]], area_points_coords[element[1]], point_coords)
        return S_1 + S_2 + S_3 - S_element
    return temp


def caution_check_point_in_elements(point_coords):
    def temp(area, element, area_points_coords):
        S_element = abs(area)
        S_1 = abs(base_func.calculate_element_area(point_coords, area_points_coords[element[1]], area_points_coords[element[2]]))
        S_2 = abs(base_func.calculate_element_area(area_points_coords[element[0]], point_coords, area_points_coords[element[2]]))
        S_3 = abs(base_func.calculate_element_area(area_points_coords[element[0]], area_points_coords[element[1]], point_coords))
        return S_1 + S_2 + S_3 - S_element
    return temp


class schwarz_two_level_additive(schwarz_additive):
    def __init__(self, data):
        self.time_1 = 0
        self.time_2 = 0
        self.time_3 = 0
        self.time_4 = 0
        self.time_5 = 0
        self.time_6 = 0
        self.time_7 = 0
        self.time_8 = 0
        self.time_9 = 0
        init_time = time.time()

        super().__init__(data)

        self.name_method = "schwarz_two_level_additive"
        self.table_name = '$\\text{Двухуровневый аддитивный МДО}$'
        
        coarse_mesh = meshio.read(f'data/{data["coarse_area"]}/meshes_coarse/{self.data["coarse_mesh"]:.3e}.msh')

        self.area_coarse_points_coords = coarse_mesh.points
        self.area_coarse_points_coords = np.delete(self.area_coarse_points_coords, -1, axis = 1)
        self.area_coarse_points = np.array([num for num, _ in enumerate(self.area_coarse_points_coords)])
        self.area_coarse_elements = coarse_mesh.cells_dict["triangle"]

        self.dict_area_coarse_dirichlet_points = {}
        self.dict_area_coarse_neumann_points = {}

        dct_points = {}
        dct_elems = {}
        dct_points_coarse = {}
        dct_elems_coarse = {}
        if self.data["fine_area"] == 'rectangle':
            dct_points = {
                'rectangle': self.area_points
            }
            dct_elems = {
                'rectangle': np.array([num for num, _ in enumerate(self.area_elements)])
            }
        elif self.data["fine_area"] == 'thick_walled_cylinder':
            dct_points = {
                'cylinder': self.area_points
            }
            dct_elems = {
                'cylinder': np.array([num for num, _ in enumerate(self.area_elements)])
            }
        elif self.data["fine_area"] == 'bearing':
            dct_cat = {
                "inner_ring": {
                    'degree': [0, np.pi/2],
                    'radius': [1.5 + 0.2926354830241924, 2.0]
                },
                "outer_ring": {
                    'degree': [0, np.pi/2],
                    'radius': [1.0, 1.5 - 0.2926354830241924]
                },
                "ball_1": {
                    'degree': [0, np.pi/(2 * 8)],
                    'radius': [1.5 - 0.2926354830241924, 1.5 + 0.2926354830241924]
                },
                "ball_2": {
                    'degree': [np.pi/(2 * 8), 3 * np.pi/(2 * 8)],
                    'radius': [1.5 - 0.2926354830241924, 1.5 + 0.2926354830241924]
                },
                "ball_3": {
                    'degree': [3 * np.pi/(2 * 8), 5 * np.pi/(2 * 8)],
                    'radius': [1.5 - 0.2926354830241924, 1.5 + 0.2926354830241924]
                },
                "ball_4": {
                    'degree': [5 * np.pi/(2 * 8), 7 * np.pi/(2 * 8)],
                    'radius': [1.5 - 0.2926354830241924, 1.5 + 0.2926354830241924]
                },
                "ball_5": {
                    'degree': [7 * np.pi/(2 * 8), np.pi/2],
                    'radius': [1.5 - 0.2926354830241924, 1.5 + 0.2926354830241924]
                },
            }
            dct_points = {
                'inner_ring': [],
                'outer_ring': [],
                'ball_1': [],
                'ball_2': [],
                'ball_3': [],
                'ball_4': [],
                'ball_5': []
            }
            for idx, point_coords in enumerate(self.area_points_coords):
                radius = np.linalg.norm(point_coords)
                degree = np.arcsin(point_coords[1] / radius)
                for key, value in dct_cat.items():
                    if (np.isclose(value['degree'][0], degree) or np.isclose(value['degree'][1], degree) or (value['degree'][0] <= degree <= value['degree'][1])) and (np.isclose(value['radius'][0], radius) or np.isclose(value['radius'][1], radius) or (value['radius'][0] <= radius <= value['radius'][1])) and (idx not in dct_points['inner_ring']) and (idx not in dct_points['outer_ring']):
                        if key not in dct_points.keys():
                            dct_points[key] = []
                        dct_points[key].append(idx)

            for idx, elem in enumerate(self.area_elements):
                for key, value in dct_points.items():
                    value = np.array(value)
                    if np.intersect1d(value, elem).size in [2, 3]:
                        if key not in dct_elems.keys():
                            dct_elems[key] = []
                        dct_elems[key].append(idx)

        if self.data["coarse_area"] == 'rectangle':
            dct_points_coarse = {
                'rectangle': self.area_coarse_points
            }
            dct_elems_coarse = {
                'rectangle': np.array([num for num, _ in enumerate(self.area_coarse_elements)])
            }
        elif self.data["coarse_area"] == 'thick_walled_cylinder' or self.data["coarse_area"] == 'simplified_cylinder':
            dct_points_coarse = {
                'cylinder': self.area_coarse_points
            }
            dct_elems_coarse = {
                'cylinder': np.array([num for num, _ in enumerate(self.area_coarse_elements)])
            }
        elif self.data["coarse_area"] == 'bearing':
            dct_points_coarse = {
                'inner_ring': [],
                'outer_ring': [],
                'ball_1': [],
                'ball_2': [],
                'ball_3': [],
                'ball_4': [],
                'ball_5': []
            }
            for idx, point_coords in enumerate(self.area_coarse_points_coords):
                radius = np.linalg.norm(point_coords)
                degree = np.arcsin(point_coords[1] / radius)
                for key, value in dct_cat.items():
                    if (np.isclose(value['degree'][0], degree) or np.isclose(value['degree'][1], degree) or (value['degree'][0] <= degree <= value['degree'][1])) and (np.isclose(value['radius'][0], radius) or np.isclose(value['radius'][1], radius) or (value['radius'][0] <= radius <= value['radius'][1])) and (idx not in dct_points_coarse['inner_ring']) and (idx not in dct_points_coarse['outer_ring']):
                        if key not in dct_points_coarse.keys():
                            dct_points_coarse[key] = []
                        dct_points_coarse[key].append(idx)

            for idx, elem in enumerate(self.area_coarse_elements):
                for key, value in dct_points_coarse.items():
                    value = np.array(value)
                    if np.intersect1d(value, elem).size in [2, 3]:
                        if key not in dct_elems_coarse.keys():
                            dct_elems_coarse[key] = []
                        dct_elems_coarse[key].append(idx)

        if self.data["coarse_area"] == 'rectangle':
            for point_num, point_coords in enumerate(self.area_coarse_points_coords):
                for row in self.dirichlet_conditions:
                    contour_points = self.contour_points[row[:2].astype(int)]
                    if np.isclose(abs(np.cross(np.diff(contour_points, axis = 0), point_coords - contour_points[0])), 0):
                        if point_num in self.dict_area_coarse_dirichlet_points:
                            template_nan = np.isnan(self.dict_area_coarse_dirichlet_points[point_num])
                            self.dict_area_coarse_dirichlet_points[point_num][template_nan] = np.copy(row)[2:][template_nan]
                        else:
                            self.dict_area_coarse_dirichlet_points[point_num] = np.copy(row)[2:]
                
                for row in self.neumann_conditions:
                    contour_points = self.contour_points[row[:2].astype(int)]
                    if np.isclose(abs(np.cross(np.diff(contour_points, axis = 0), point_coords - contour_points[0])), 0):
                        self.dict_area_coarse_neumann_points[point_num] = row[2:]
        else:
            for point_num, point_coords in enumerate(self.area_coarse_points_coords):
                for index, row in enumerate(self.dirichlet_conditions):
                    if index in [0, 1]:
                        radius, displacement = row[0], row[1]
                        if np.isclose(np.linalg.norm(point_coords), radius):
                            self.dict_area_coarse_dirichlet_points[point_num] = point_coords * (displacement / radius)
                    else:
                        contour_points = self.contour_points[row[:2].astype(int)]
                        if np.isclose(abs(np.cross(np.diff(contour_points, axis = 0), point_coords - contour_points[0])), 0):
                            if point_num in self.dict_area_coarse_dirichlet_points:
                                template_nan = np.isnan(self.dict_area_coarse_dirichlet_points[point_num])
                                self.dict_area_coarse_dirichlet_points[point_num][template_nan] = np.copy(row)[2:][template_nan]
                            else:
                                self.dict_area_coarse_dirichlet_points[point_num] = np.copy(row)[2:]
                for row in self.neumann_conditions:
                    radius, pressure = row[0], row[1]
                    if np.isclose(np.linalg.norm(point_coords), radius):
                        self.dict_area_coarse_neumann_points[point_num] = pressure

        self.list_area_coarse_neumann_elements = []
        for index_element, element in enumerate(self.area_coarse_elements):
            if len(set(element) & set(self.dict_area_coarse_neumann_points.keys())) == 2:
                self.list_area_coarse_neumann_elements.append(index_element)

        self.K_special = base_func.calculate_sparse_matrix_stiffness(self.area_elements, self.area_points_coords, self.area_points.size, self.D, self.dim_task)
        self.F_special = np.zeros(self.area_points_coords.size)

        self.set_condition_neumann(self.F_special, self.list_area_neumann_elements, self.area_points_coords, self.dict_area_neumann_points)
        self.set_condition_dirichlet(self.K_special, self.F_special, self.dict_area_dirichlet_points, self.dict_area_dirichlet_points.keys())

        self.K_coarse_special = base_func.calculate_sparse_matrix_stiffness(self.area_coarse_elements, self.area_coarse_points_coords, self.area_coarse_points.size, self.D, self.dim_task)
        self.F_coarse_special = np.zeros(self.area_coarse_points_coords.size)

        self.barycentric_coords_for_coarse_elements = {}
        for num_element, coarse_element in enumerate(self.area_coarse_elements):
            self.barycentric_coords_for_coarse_elements[num_element] = base_func.create_barycentric_coords(coarse_element, self.area_coarse_points_coords)

        self.element_centroid_points_coords_coarse = np.array(list(map(lambda x: np.mean(self.area_coarse_points_coords[x], axis = 0), self.area_coarse_elements)))
        self.list_area_of_coarse_elements = np.array([base_func.calculate_local_matrix_stiffness(i, self.area_coarse_points_coords, self.dim_task)[1] for i in self.area_coarse_elements])
        self.dict_point_in_coarse_elements = {}
        for num_point, point_coords in enumerate(self.area_points_coords):
            bool_check = check_point_in_elements(point_coords)
            for key, value in dct_points.items():
                if num_point in value:
                    area_name = key
                    break
            if self.data["fine_area"] == "bearing" and (self.data["coarse_area"] == "thick_walled_cylinder" or self.data["coarse_area"] == "simplified_cylinder"):
                area_name = "cylinder"

            dct_caution = {}
            for num in dct_elems_coarse[area_name]:
                dct_caution[num] = bool_check(self.list_area_of_coarse_elements[num], self.area_coarse_elements[num], self.area_coarse_points_coords)

            min_triangle = min(list(dct_caution.values()))
            for key, value in dct_caution.items():
                if value == min_triangle:
                    self.dict_point_in_coarse_elements[num_point] = key
        # test_dict = {}
        # for key, value in self.dict_point_in_coarse_elements.items():
        #     if value in test_dict.keys():
        #         test_dict[value].append(key)
        #     else:
        #         test_dict[value] = [key]
        # for key, value in test_dict.items():
        #     self.internal_plot_displacement_coarse(self.area_points_coords[value], self.area_coarse_points_coords, self.area_coarse_elements)
        self.time_init = time.time() - init_time


    def get_displacements(self):
        u_special = np.ravel(np.zeros_like(self.u))

        F_previous = self.K_special.dot(np.ravel(self.u_previous))
        F_residual = self.F_special.copy() - F_previous

        K_coarse = self.K_coarse_special.copy()
        F_coarse = self.F_coarse_special.copy()

        for point, element in self.dict_point_in_coarse_elements.items():
            point_coords = self.area_points_coords[point]
            [a, b, c], A = self.barycentric_coords_for_coarse_elements[element]
            value = (a + b * point_coords[0] + c * point_coords[1]) / 2 / A

            F_coarse[self.area_coarse_elements[element] * self.dim_task] += np.dot(value, F_residual[point * self.dim_task])
            F_coarse[self.area_coarse_elements[element] * self.dim_task + 1] += np.dot(value, F_residual[point * self.dim_task + 1])

        self.set_condition_dirichlet(K_coarse, F_coarse, self.dict_area_coarse_dirichlet_points, self.dict_area_coarse_dirichlet_points.keys())
        [*arg,] = self.solve_function(K_coarse.tocsr(), F_coarse)
        u_coarse = np.ravel(np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2)))

        for point, element in self.dict_point_in_coarse_elements.items():
            point_coords = self.area_points_coords[point]
            [a, b, c], A = self.barycentric_coords_for_coarse_elements[element]
            value = (a + b * point_coords[0] + c * point_coords[1]) / 2 / A

            u_special[point * self.dim_task] += np.dot(value, u_coarse[self.area_coarse_elements[element] * self.dim_task])
            u_special[point * self.dim_task + 1] += np.dot(value, u_coarse[self.area_coarse_elements[element] * self.dim_task + 1])

        self.u = self.u_previous.copy() + (self.coef_alpha * self.u_sum.copy()) + (self.coef_alpha * u_special.reshape(-1, 2))


    def plot_area_init_coarse_mesh(self):
        self.internal_plot_displacement(self.area_coarse_points_coords, self.area_coarse_elements)

if __name__ == "__main__":
    pass
