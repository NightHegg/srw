import os
import math

import numpy as np
import pygmsh

class Task:
    def __init__(self, area, contour, dim_task, E, nyu, coef_u, coef_sigma):
        self.cur_area = area
        self.contour = contour

        if not os.path.exists(f'data/{self.cur_area}'):
            os.makedirs(f'data/{self.cur_area}')
            
        with open(f'data/{self.cur_area}/area_info.dat', "w") as f:
            f.write(f'{len(self.contour):g}\n')
            for point in self.contour:
                f.write("{:g} {:g}\n".format(point[0], point[1]))
            f.write(f'{dim_task}\n')
            f.write(f'{E:g} {nyu:g}\n')
            f.write(f'{coef_u:g} {coef_sigma:g}')


    def create_task(self, task_name, task_params):
        if not os.path.exists(f'data/{self.cur_area}/tasks'):
            os.makedirs(f'data/{self.cur_area}/tasks')

        value_dir = task_params["dirichlet_conditions"]
        value_neu = task_params["neumann_conditions"]
        with open(f'data/{self.cur_area}/tasks/{task_name}.dat', "w") as f:
            if self.cur_area == 'rectangle':
                f.write(f'{len(value_dir):g}\n')
                for cond in value_dir:
                    f.write(f'{cond[0]:g} {cond[1]:g} {cond[2]:g} {cond[3]:g}\n')

                f.write(f'{len(value_neu):g}\n')
                for cond in value_neu:
                    f.write(f'{cond[0]:g} {cond[1]:g} {cond[2]:g} {cond[3]:g}\n')
            else:
                f.write(f'{value_dir["inner_side"]:g}\n')
                f.write(f'{value_dir["outer_side"]:g}\n')
                f.write(f'{len(value_dir["other"]):g}\n')
                for cond in value_dir["other"]:
                    f.write(f'{cond[0]:g} {cond[1]:g} {cond[2]:g} {cond[3]:g}\n')
                
                f.write(f'{value_neu["inner_side"]:g}\n')
                f.write(f'{value_neu["outer_side"]:g}')


    def create_and_write_mesh(self, mesh_type, edge_size):
        if not os.path.exists(f'data/{self.cur_area}/{mesh_type}'):
            os.makedirs(f'data/{self.cur_area}/{mesh_type}')

        with pygmsh.occ.Geometry() as geom:
            geom.characteristic_length_min = edge_size
            geom.characteristic_length_max = edge_size
            if self.cur_area == 'rectangle':
                x_size, y_size = self.contour[2]
                geom.add_rectangle([0.0, 0.0, 0.0], x_size, y_size)
            else:
                inner_radius = self.contour[0][0]
                outer_radius = self.contour[1][0]

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
                    amnt_balls = 3
                    check = True
                    
                    ball_center_point = inner_radius + (outer_radius - inner_radius) / 2.0
                    if check:
                        ball_radius = np.sin(np.pi / (4 * (amnt_balls + 1))) * ball_center_point
                        dist = ball_radius - (outer_radius - inner_radius) / 4.0
                    else:
                        ball_radius = (outer_radius - inner_radius) / 4.0
                        dist = 0
                    
                    inner_ring_hole = geom.add_disk([0.0, 0.0], inner_radius)
                    inner_ring = geom.add_disk([0.0, 0.0], inner_radius + (outer_radius - inner_radius) / 4 - dist)
                    inner_ring_full = geom.boolean_difference(inner_ring, inner_ring_hole)

                    outer_ring_hole = geom.add_disk([0.0, 0.0], inner_radius + 3 * (outer_radius - inner_radius) / 4 + dist)
                    outer_ring = geom.add_disk([0.0, 0.0], outer_radius)
                    outer_ring_full = geom.boolean_difference(outer_ring, outer_ring_hole)

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
            pygmsh.write(f'data/{self.cur_area}/{mesh_type}/{edge_size:.3e}.msh')


if __name__ == "__main__":
    area_names = ['rectangle', 'thick_walled_cylinder', 'simplified_cylinder', 'bearing']
    inner_radius, outer_radius = 1.0, 2.0
    x_size, y_size = 2.0, 1.0

    cur_area = area_names[3]
    contour = [[0, 0], [x_size, 0], [x_size, y_size], [0, y_size]] if cur_area == 'rectangle' else [[inner_radius, 0], [outer_radius, 0], [0, outer_radius], [0, inner_radius]]
    cur_E = 210e+9 if cur_area == "bearing" else 70e+9
    cur_nyu = 0.25 if cur_area == "bearing" else 0.34

    area_parameters = {
        'area':        cur_area,
        'contour':     contour,
        'dim_task':    2,
        'E':           cur_E,
        'nyu':         cur_nyu,
        'coef_u':      1000,
        'coef_sigma':  1e-6
    }

    tasks = {
        'rectangle': {
            '3_fixes': {
                'dirichlet_conditions': [[0, 1, math.nan, 0], [1, 2, 0, math.nan], [3, 0, 0, math.nan]],
                'neumann_conditions': [[2, 3, 0, -2e+7]]
            },
            '2_fixes': {
                'dirichlet_conditions': [[0, 1, math.nan, 0], [3, 0, 0, math.nan]],
                'neumann_conditions': [[2, 3, 0, -2e+7]]
            }
        },
        'thick_walled_cylinder': {
            'pressure_only': {
                'dirichlet_conditions': {
                    'inner_side': math.nan,
                    'outer_side': math.nan,
                    'other': [[0, 1, math.nan, 0], [2, 3, 0, math.nan]]
                },
                'neumann_conditions': {
                    'inner_side': 5e+6,
                    'outer_side': -1e+7
                }
            }
        },
        'simplified_cylinder': {

        },
        'bearing': {
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
            }
        }    
    }

    meshes = {
        'rectangle': {
            "meshes_coarse": [1, 0.5, 0.25, 0.125, 0.0625],
            "meshes_fine": [0.5, 0.25, 0.125, 0.05, 0.025, 0.0125, 0.00625]
        },
        'thick_walled_cylinder': {
            "meshes_coarse": [1, 0.5, 0.25, 0.125, 0.0625],
            "meshes_fine": [1, 0.5, 0.25, 0.125, 0.05, 0.025, 0.0125, 0.00625]
        },
        'simplified_cylinder': {
            "meshes_coarse": [1, 0.5, 0.25, 0.125, 0.0625],
            "meshes_fine": []
        },
        'bearing': {
            "meshes_coarse": [],
            "meshes_fine": [0.5, 0.25, 0.125, 0.05, 0.025, 0.0125, 0.00625]
        }
    }

    obj = Task(**area_parameters)

    for task_name, params in tasks[cur_area].items():
        obj.create_task(task_name, params)

    for mesh_type, list_edge_size in meshes[cur_area].items():
        for cur_edge_size in list_edge_size:
            obj.create_and_write_mesh(mesh_type, cur_edge_size)