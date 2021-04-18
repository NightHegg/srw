import os
import sys
import math
from itertools import chain, product
import matplotlib.pyplot as plt

import numpy as np
import dmsh, meshio, optimesh

class Task:
    def __init__(self, cur_area, area_params):
        self.cur_area = cur_area
        self.area_params = area_params
        self.contour = self.area_params['points']

        if not os.path.exists(f'data/{self.cur_area}'):
            os.makedirs(f'data/{self.cur_area}')
            os.makedirs(f'data/{self.cur_area}/meshes')
            os.makedirs(f'data/{self.cur_area}/meshes/fine_meshes')
            os.makedirs(f'data/{self.cur_area}/meshes/coarse_meshes')
            os.makedirs(f'data/{self.cur_area}/tasks')
            
        with open(f'data/{self.cur_area}/area_info.dat', "w") as f:
            f.write(f'{len(self.contour):g}\n')
            for point in self.contour:
                f.write("{:g} {:g}\n".format(point[0], point[1]))

            f.write(f'{self.area_params["dim_task"]}\n')
            f.write(f'{self.area_params["E"]:g} {self.area_params["nyu"]:g}\n')
            f.write(f'{self.area_params["coef_u"]:g} {self.area_params["coef_sigma"]:g}')


    def create_task(self, cur_task, task_params):
        value_dir = task_params["dirichlet_conditions"]
        value_neu = task_params["neumann_conditions"]
        with open(f'data/{self.cur_area}/tasks/{cur_task}.dat', "w") as f:
            if self.cur_area == 'rectangle':
                f.write(f'{len(value_dir):g}\n')
                for cond in value_dir:
                    f.write(f'{cond[0]:g} {cond[1]:g} {cond[2]:g} {cond[3]:g}\n')

                f.write(f'{len(value_neu):g}\n')
                for cond in value_neu:
                    f.write(f'{cond[0]:g} {cond[1]:g} {cond[2]:g} {cond[3]:g}\n')
            elif self.cur_area == 'thick_walled_cylinder':
                f.write(f'{value_dir["inner_side"]:g}\n')
                f.write(f'{value_dir["outer_side"]:g}\n')
                f.write(f'{len(value_dir["other"]):g}\n')
                for cond in value_dir["other"]:
                    f.write(f'{cond[0]:g} {cond[1]:g} {cond[2]:g} {cond[3]:g}\n')
                
                f.write(f'{value_neu["inner_side"]:g}\n')
                f.write(f'{value_neu["outer_side"]:g}')

    def create_mesh(self, edge_size):
        if self.cur_area == 'rectangle':
            self.geo = dmsh.Polygon(self.contour)
            self.X, self.cells = dmsh.generate(self.geo, edge_size)
            # self.X, self.cells = optimesh.optimize_points_cells(self.X, self.cells, "CVT (full)", 1.0e-10, 200)

        elif self.cur_area == 'thick_walled_cylinder':
            inner_border = self.contour[0][0]
            outer_border = self.contour[1][0]

            low_r = dmsh.Rectangle(-outer_border, 0, 0, outer_border)
            left_r = dmsh.Rectangle(-outer_border, outer_border, -outer_border, 0.0)
            full_polygon = dmsh.Union([low_r, left_r])

            big_c = dmsh.Circle([0.0, 0.0], outer_border)
            small_c = dmsh.Circle([0.0, 0.0], inner_border)        
            quarter = dmsh.Difference(big_c, small_c)
      
            self.geo = dmsh.Difference(quarter, full_polygon)
            self.X, self.cells = dmsh.generate(self.geo, edge_size)
            self.X, self.cells = optimesh.optimize_points_cells(self.X, self.cells, "CVT (block-diagonal)", 1.0e-10, 50)

    def show_mesh(self):
        dmsh.helpers.show(self.X, self.cells, self.geo)

    def write_mesh(self, edge_size, fine_mesh = True):
        folder = 'fine_meshes' if fine_mesh else 'coarse_meshes'
        name_file = f'data/{self.cur_area}/meshes/{folder}/{edge_size:.2e}.dat'
        meshio.write_points_cells(name_file, self.X, {"triangle": self.cells})


if __name__ == "__main__":
    area_parameters = {
        'rectangle':
        {
            'points':      [[0.01, 0], [0.02, 0], [0.02, 0.01], [0.01, 0.01]],
            'dim_task':    2,
            'E':           70e+9,
            'nyu':         0.34,
            'coef_u':      1000,
            'coef_sigma':  1e-3
        },
        'thick_walled_cylinder':
        {
            'points':      [[1, 0], [2, 0], [0, 2], [0, 1]],
            'dim_task':    2,
            'E':           70e+9,
            'nyu':         0.34,
            'coef_u':      1000,
            'coef_sigma':  1e-6
        }
    }
    tasks = {
        'rectangle':
        {
            '3_bindings': {
                'dirichlet_conditions': [[0, 1, math.nan, 0], [1, 2, 0, math.nan], [0, 3, 0, math.nan]],
                'neumann_conditions':   [[2, 3, math.nan, -2e+7]]
                },
            '2_bindings': {
                'dirichlet_conditions': [[0, 1, math.nan, 0], [1, 2, 0, math.nan], [0, 3, 0, math.nan]],
                'neumann_conditions':   [[2, 3, math.nan, -2e+7]]
                }
        },
        'thick_walled_cylinder': 
        {
            'outer_pressure_only': {
                'dirichlet_conditions': {
                    'inner_side': math.nan,
                    'outer_side': math.nan,
                    'other': [[0, 1, math.nan, 0], [2, 3, 0, math.nan]]
                    },
                'neumann_conditions': {
                    'inner_side': 5e+6,
                    'outer_side': -1e+7
                    }
                },
            'inner_pressure_only': {
                'dirichlet_conditions': {
                    'inner_side': math.nan,
                    'outer_side': math.nan,
                    'other': [[0, 1, math.nan, 0], [2, 3, 0, math.nan]]
                    },
                'neumann_conditions': {
                    'inner_side': 5e+6,
                    'outer_side': 0
                    }
                },
            'outer_displacements_only': {
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
    }
    area_names = list(area_parameters.keys())

    coarse_edge_size = [1]
    fine_edge_size = [1]

    cur_area = area_names[1]
    obj = Task(cur_area, area_parameters[cur_area])

    for cur_task in tasks[cur_area]:
        obj.create_task(cur_task, tasks[cur_area][cur_task])

    # for cur_edge_size in coarse_edge_size:
    #     obj.create_mesh(cur_edge_size)
    #     # obj.show_mesh()
    #     obj.write_mesh(cur_edge_size, False)

    # for cur_edge_size in fine_edge_size:
    #     obj.create_mesh(cur_edge_size)
    #     # obj.show_mesh()
    #     obj.write_mesh(cur_edge_size, True)