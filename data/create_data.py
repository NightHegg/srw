import os
import sys
import math
from itertools import chain, product
import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial import Delaunay

import dmsh, meshio, optimesh

class Task:
    def __init__(self, area_params):
        self.area_params = area_params

        if not os.path.exists(f'data/{self.area_params["folder_name"]}'):
            os.makedirs(f'data/{self.area_params["folder_name"]}')
            os.makedirs(f'data/{self.area_params["folder_name"]}/meshes')
            os.makedirs(f'data/{self.area_params["folder_name"]}/tasks')
            
        self.contour = self.area_params['points']
        with open(f'data/{self.area_params["folder_name"]}/contour.dat', "w") as f:
            f.write(f'{len(self.contour):g}\n')
            for point in self.contour:
                f.write("{:g} {:g}\n".format(point[0], point[1]))

            f.write(f'{self.area_params["dim_task"]}\n')
            f.write(f'{self.area_params["E"]:g} {self.area_params["nyu"]:g}\n')
            f.write(f'{self.area_params["coef_u"]:g} {self.area_params["coef_sigma"]:g}')

    def create_task(self, params):
        with open(f'data/{self.area_params["folder_name"]}/tasks/{params["task_name"]}.dat', "w") as f:
            f.write(f'{len(params["dirichlet_conditions"]):g}\n')
            for cond in params['dirichlet_conditions']:
                f.write(f'{cond[0]:g} {cond[1]:g} {cond[2]:g} {cond[3]:g}\n')

            f.write(f'{len(params["neumann_conditions"]):g}\n')
            for cond in params["neumann_conditions"]:
                f.write(f'{cond[0]:g} {cond[1]:g} {cond[2]:g} {cond[3]:g}\n')

    def create_mesh(self, edge_size):
        if self.area_params['folder_name'] == 'area_01':
            self.geo = dmsh.Polygon(self.contour)
            self.X, self.cells = dmsh.generate(self.geo, edge_size)
            self.X, self.cells = optimesh.optimize_points_cells(self.X, self.cells, "CVT (full)", 1.0e-10, 100)
        elif self.area_params['folder_name'] == 'area_02':
            low_r = dmsh.Rectangle(-2.0, 0, 0, 2.0)
            left_r = dmsh.Rectangle(-2.0, 2.0, -2.0, 0.0)
            full_polygon = dmsh.Union([low_r, left_r])

            big_c = dmsh.Circle([0.0, 0.0], 2)
            small_c = dmsh.Circle([0.0, 0.0], 1)        
            quarter = dmsh.Difference(big_c, small_c)
      
            self.geo = dmsh.Difference(quarter, full_polygon)
            self.X, self.cells = dmsh.generate(self.geo, edge_size)
            # self.X, self.cells = optimesh.optimize_points_cells(self.X, self.cells, "CVT (full)", 1.0e-10, 100)

    def show_mesh(self):
        dmsh.helpers.show(self.X, self.cells, self.geo)

    def write_mesh(self, edge_size):
        name_file = f'data/{self.area_params["folder_name"]}/meshes/{edge_size:.2e}.dat'
        meshio.write_points_cells(name_file, self.X, {"triangle": self.cells})


if __name__ == "__main__":
    area_params = [
        # {
        #     'folder_name': 'area_01',
        #     'points':      [[0.01, 0], [0.02, 0], [0.02, 0.01], [0.01, 0.01]],
        #     'dim_task':    2,
        #     'E':           70e+9,
        #     'nyu':         0.34,
        #     'coef_u':      1000,
        #     'coef_sigma':  1e-3
        # },
        {
            'folder_name': 'area_02',
            'points':      [[0.01, 0], [0.02, 0], [0.02, 0.01], [0.01, 0.01]],
            'dim_task':    2,
            'E':           70e+9,
            'nyu':         0.34,
            'coef_u':      1000,
            'coef_sigma':  1e-3
        }
        ]
    tasks = [
        {
            'task_name':           'task_01',
            'dirichlet_conditions': [[0, 1, math.nan, 0], [1, 2, 0, math.nan], [0, 3, 0, math.nan]],
            'neumann_conditions':   [[2, 3, math.nan, -2e+7]]
            },
        {
            'task_name':            'task_02',
            'dirichlet_conditions': [[0, 1, math.nan, 0], [0, 3, 0, math.nan]],
            'neumann_conditions':   [[2, 3, math.nan, -2e+7]]
        }
    ]
    list_edge_size = [0.1]

    for cur_area in area_params:
        obj = Task(cur_area)

        # for cur_task in tasks:
        #     obj.create_task(cur_task)

        for cur_edge_size in list_edge_size:
            obj.create_mesh(cur_edge_size)
            obj.show_mesh()
            # obj.write_mesh(cur_edge_size)