import os
import math

import numpy as np
import pygmsh

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


    def create_and_write_mesh(self, mesh_type, edge_size):
        inner_radius = self.contour[0][0]
        outer_radius = self.contour[1][0]

        with pygmsh.occ.Geometry() as geom:
            geom.characteristic_length_min = edge_size
            geom.characteristic_length_max = edge_size

            polygon_1 = geom.add_rectangle([-outer_radius, -outer_radius, 0], outer_radius, outer_radius * 2)
            polygon_2 = geom.add_rectangle([0, -outer_radius, 0], outer_radius, outer_radius)

            if self.cur_area == 'thick_walled_cylinder':
                outer_ring = geom.add_disk([0.0, 0.0], outer_radius)
                inner_ring = geom.add_disk([0.0, 0.0], inner_radius)

                ring = geom.boolean_difference(outer_ring, inner_ring)
                geom.boolean_difference(ring, geom.boolean_union([polygon_1, polygon_2]))

            elif self.cur_area == 'simplified_cylinder':
                ring = geom.add_disk([0.0, 0.0], outer_radius)
                geom.boolean_difference(ring, geom.boolean_union([polygon_1, polygon_2]))

            elif self.cur_area == 'bearing':
                amnt_balls = 1
                check = False
                ball_center_point = inner_radius + (outer_radius - inner_radius) / 2.0
                if check:
                    ball_radius = np.sin(np.pi / (4 * (amnt_balls + 1))) * ball_center_point
                    dist = ball_radius - (outer_radius - inner_radius) / 4.0
                else:
                    ball_radius = (outer_radius - inner_radius) / 4.0
                    dist = 0
                
                outer_ring_hole = geom.add_disk([0.0, 0.0], inner_radius + 3 * (outer_radius - inner_radius) / 4 + dist)
                outer_ring = geom.add_disk([0.0, 0.0], outer_radius)
                outer_ring_full = geom.boolean_difference(outer_ring, outer_ring_hole)

                inner_ring_hole = geom.add_disk([0.0, 0.0], inner_radius)
                inner_ring = geom.add_disk([0.0, 0.0], inner_radius + (outer_radius - inner_radius) / 4 - dist)
                inner_ring_full = geom.boolean_difference(inner_ring, inner_ring_hole)

                list_balls = []
                for i in range(4):
                    degree = i * np.pi / 2
                    for j in range(amnt_balls):
                        add_ball_degree = (i * np.pi / 2) + ((j + 1) * (np.pi / 2 / (amnt_balls + 1)))
                        list_balls.append(geom.add_disk([ball_center_point * np.cos(add_ball_degree), ball_center_point * np.sin(add_ball_degree)], ball_radius))
                    list_balls.append(geom.add_disk([ball_center_point * np.cos(degree), ball_center_point * np.sin(degree)], ball_radius))

                ring = geom.boolean_union([outer_ring_full, inner_ring_full, list_balls])
                geom.boolean_difference(ring, geom.boolean_union([polygon_1, polygon_2]))

            geom.generate_mesh()
            pygmsh.write(f'data/meshes/{mesh_type}/{self.cur_area}/{edge_size:.3e}.msh')


if __name__ == "__main__":
    area_names = ['thick_walled_cylinder', 'simplified_cylinder', 'bearing']

    area_parameters = {
            'area':        'bearing',
            'contour':     [[1, 0], [2, 0], [0, 2], [0, 1]],
            'dim_task':    2,
            'E':           210e+9,
            'nyu':         0.25,
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
        "fine": [0.05, 0.025, 0.0125]
    }

    obj = Task(**area_parameters)

    for task_name, params in tasks.items():
        obj.create_task(task_name, params)

    for mesh_type, params in meshes.items():
        if not (mesh_type == 'fine' and area_parameters['area'] == 'simplified_cylinder'):
            for edge_size in params:
                obj.create_and_write_mesh(mesh_type, edge_size)