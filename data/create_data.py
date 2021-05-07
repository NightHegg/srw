import os
import sys
import math
from itertools import chain, product
import matplotlib.pyplot as plt

import numpy as np
import dmsh, meshio, optimesh

class Task:
    def __init__(self, area, contour, dim_task, E, nyu, coef_u, coef_sigma):
        self.cur_area = area
        self.contour = contour

        if not os.path.exists(f'data/meshes'):
            os.makedirs(f'data/meshes/fine')
            os.makedirs(f'data/meshes/coarse')

        if not os.path.exists(f'data/meshes/coarse/{self.cur_area}'):
            os.makedirs(f'data/meshes/coarse/{self.cur_area}')

        if self.cur_area != 'simplified_cylinder':
            if not os.path.exists(f'data/meshes/fine/{self.cur_area}'):
                os.makedirs(f'data/meshes/fine/{self.cur_area}')
        
        if not os.path.exists(f'data/tasks'):
            os.makedirs(f'data/tasks')
            
        with open(f'data/area_info.dat', "w") as f:
            f.write(f'{len(self.contour):g}\n')
            for point in self.contour:
                f.write("{:g} {:g}\n".format(point[0], point[1]))
            f.write(f'{dim_task}\n')
            f.write(f'{E:g} {nyu:g}\n')
            f.write(f'{coef_u:g} {coef_sigma:g}')


    def create_task(self, task_name, task_params):
        value_dir = task_params["dirichlet_conditions"]
        value_neu = task_params["neumann_conditions"]
        with open(f'data/tasks/{task_name}.dat', "w") as f:
            f.write(f'{value_dir["inner_side"]:g}\n')
            f.write(f'{value_dir["outer_side"]:g}\n')
            f.write(f'{len(value_dir["other"]):g}\n')
            for cond in value_dir["other"]:
                f.write(f'{cond[0]:g} {cond[1]:g} {cond[2]:g} {cond[3]:g}\n')
            
            f.write(f'{value_neu["inner_side"]:g}\n')
            f.write(f'{value_neu["outer_side"]:g}')


    def create_mesh(self, edge_size):
        if self.cur_area == 'thick_walled_cylinder':
            inner_border = self.contour[0][0]
            outer_border = self.contour[1][0]

            low_r = dmsh.Rectangle(-outer_border, 0, 0, outer_border)
            left_r = dmsh.Rectangle(-outer_border, outer_border, -outer_border, 0.0)
            full_polygon = dmsh.Union([low_r, left_r])

            small_c = dmsh.Circle([0.0, 0.0], inner_border)
            big_c = dmsh.Circle([0.0, 0.0], outer_border)      
            quarter = dmsh.Difference(big_c, small_c)
      
            self.geo = dmsh.Difference(quarter, full_polygon)

        elif self.cur_area == 'simplified_cylinder':
            outer_border = self.contour[1][0]

            low_r = dmsh.Rectangle(-outer_border, 0, 0, outer_border)
            left_r = dmsh.Rectangle(-outer_border, outer_border, -outer_border, 0.0)
            full_polygon = dmsh.Union([low_r, left_r])

            big_c = dmsh.Circle([0.0, 0.0], outer_border)       
      
            self.geo = dmsh.Difference(big_c, full_polygon)
        
        elif self.cur_area == 'bearing':
            inner_border = self.contour[0][0]
            outer_border = self.contour[1][0]

            full_polygon = dmsh.Polygon(
                [
                    [-2.0, -2.0],
                    [-2.0, 2.0],
                    [0.0, 2.0],
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [2.0, -2.0]
                ]
            )

            inner_small_c = dmsh.Circle([0.0, 0.0], inner_border)
            inner_big_c = dmsh.Circle([0.0, 0.0], inner_border + (outer_border - inner_border) / 4)
            inner_c = dmsh.Difference(inner_big_c, inner_small_c)

            outer_small_c = dmsh.Circle([0.0, 0.0], inner_border + 3 * (outer_border - inner_border) / 4)
            outer_big_c = dmsh.Circle([0.0, 0.0], outer_border)
            outer_c = dmsh.Difference(outer_big_c, outer_small_c)


            r = (outer_border - inner_border) / 4
            center = inner_border + (outer_border - inner_border) / 2
            M = 1
            list_circles = []
            for i in range(4):
                r_main = i * np.pi / 2
                for j in range(M):
                    r_add = (i * np.pi / 2) + ((j + 1) * (np.pi / 2 / (M + 1)))
                    list_circles.append(dmsh.Circle([center * math.cos(r_add), center * math.sin(r_add)], r))
                list_circles.append(dmsh.Circle([center * math.cos(r_main), center * math.sin(r_main)], r))

            inner_circles = dmsh.Union(list_circles)
            full_circle = dmsh.Union([inner_circles, outer_c])
            full_circle = dmsh.Union([full_circle, inner_c])

            self.geo = dmsh.Difference(full_circle, full_polygon)
        
        self.X, self.cells = dmsh.generate(self.geo, edge_size)
        self.X, self.cells = optimesh.optimize_points_cells(self.X, self.cells, "CVT (block-diagonal)", 1.0e-10, 50)

    def show_mesh(self):
        # self.geo.show(False)
        dmsh.helpers.show(self.X, self.cells, self.geo)

    def write_mesh(self, mesh_type, edge_size):
        name_file = f'data/meshes/{mesh_type}/{self.cur_area}/{edge_size:.3e}.dat'
        meshio.write_points_cells(name_file, self.X, {"triangle": self.cells})


if __name__ == "__main__":
    area_names = ['thick_walled_cylinder', 'simplified_cylinder', 'bearing']

    area_parameters = {
            'area':        'bearing',
            'contour':     [[1, 0], [2, 0], [0, 2], [0, 1]],
            'dim_task':    2,
            'E':           70e+9,
            'nyu':         0.34,
            'coef_u':      1000,
            'coef_sigma':  1e-6
    }
    tasks = {
        'pressure_only': {
            'dirichlet_conditions': {
                'inner_side': math.nan,
                'outer_side': math.nan,
                'other': [[0, 1, math.nan, 0], [2, 3, 0, math.nan]]
                },
            'neumann_conditions': {
                'inner_side': 0,
                'outer_side': -1e+6
                }
            },
        'displacements_only': {
            'dirichlet_conditions': {
                'inner_side': math.nan,
                'outer_side': -2e-4,
                'other': [[0, 1, math.nan, 0], [2, 3, 0, math.nan]]
                },
            'neumann_conditions': {
                'inner_side': 0,
                'outer_side': 0
                }
            }
    }
    meshes = {
        "coarse": [],
        "fine": [0.05]
    }

    obj = Task(**area_parameters)

    for task_name, params in tasks.items():
        obj.create_task(task_name, params)

    # for mesh_type, params in meshes.items():
    #     if not (mesh_type == 'fine' and area_parameters['area'] == 'simplified_cylinder'):
    #         for edge_size in params:
    #             obj.create_mesh(edge_size)
    #             obj.show_mesh()
    #             obj.write_mesh(mesh_type, edge_size)