import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from scr.class_basic_method import basic_method
from scr.class_schwarz_multiplicative import schwarz_multiplicative
from scr.class_schwarz_additive import schwarz_additive
from scr.class_schwarz_two_level_additive import schwarz_two_level_additive


def test(cur_area, cur_task):
    example_data = {
        'area':             cur_area,
        'task':             cur_task,
        'mesh':             '5.0e-01',
        'amnt_subds':       [2, 1],
        'coef_convergence': 1e-4,
        'coef_overlap':     0.35,
        'coef_alpha':       0.5,
        'coarse_mesh':      '9.5e-04'
    }
    obj = basic_method(example_data)
    # obj.plot_init_mesh(False)
    obj.get_solution()
    # print(*obj.get_info())
    # obj.plot_displacements(False)
    # obj.plot_polar()


def task_iters_sigma():
    data = {
        'method':          schwarz_two_level_additive,
        'text':            'two_level_additive',
        'area':            'area_01',
        'task':            'task_02',
        'mesh_list':       ["4.00e-04", "2.00e-04", "1.00e-04"],
        'amnt_subds_list': [[2, 1], [4, 1], [8, 1]]
    }

    if not os.path.exists(f'results/{data["area"]}'):
        os.makedirs(f'results/{data["area"]}')

    if not os.path.exists(f'results/{data["area"]}/{data["task"]}'):
        os.makedirs(f'results/{data["area"]}/{data["task"]}')
        os.makedirs(f'results/{data["area"]}/{data["task"]}/iterations')
        os.makedirs(f'results/{data["area"]}/{data["task"]}/sigma')
        os.makedirs(f'results/{data["area"]}/{data["task"]}/time')

    dict_iters = {}
    dict_sigma = {}
    dict_time = {}

    for cur_mesh in tqdm(data["mesh_list"]):
        dict_iters_temp = {}
        dict_sigma_temp = {}
        dict_time_temp = {}
        for cur_amnt_subds in tqdm(data["amnt_subds_list"]):
            basic_data = {
                'area':             data['area'],
                'task':             data['task'],
                'coef_convergence': 1e-5,
                'coef_overlap':     0.35,
                'coef_alpha':       0.5,
                'coarse_mesh':      '1.5e-03'
            }
            add_data = {'mesh': cur_mesh, 'amnt_subds': cur_amnt_subds}
            basic_data.update(add_data)

            obj = data['method'](basic_data)
            obj.get_solution()

            array_amnt_subds = np.prod(np.array(cur_amnt_subds))
            styled_amnt_subds = f"{array_amnt_subds} области" if array_amnt_subds < 5 else f"{array_amnt_subds} областей"

            dict_iters_temp[styled_amnt_subds] = obj.amnt_iterations
            dict_sigma_temp[styled_amnt_subds] = obj.get_special_sigma()
            dict_time_temp[styled_amnt_subds] = f'{obj.time_global:.2f}'
        
        dict_iters[f"h={cur_mesh}"] = dict_iters_temp
        dict_sigma[f"h={cur_mesh}"] = dict_sigma_temp
        dict_time[f"h={cur_mesh}"] = dict_time_temp
    
    df_iters = pd.DataFrame.from_dict(dict_iters)
    df_sigma = pd.DataFrame.from_dict(dict_sigma)
    df_time = pd.DataFrame.from_dict(dict_time)

    print(df_iters)
    print(df_sigma)
    print(df_time)

    df_iters.to_csv(f'results/{data["area"]}/{data["task"]}/iterations/{data["text"]}.csv', index = False)
    df_sigma.to_csv(f'results/{data["area"]}/{data["task"]}/sigma/{data["text"]}.csv', index = False)
    df_time.to_csv(f'results/{data["area"]}/{data["task"]}/time/{data["text"]}.csv', index = False)


if __name__ == "__main__":
    area_names = ['rectangle', 'thick_walled_cylinder']
    tasks = {
        'rectangle': ['3_bindings', '2_bindings'], 
        'thick_walled_cylinder': ['outer_pressure_only', 'inner_pressure_only', 'outer_displacements_only']
    }
    cur_area = area_names[1]
    test(cur_area, tasks[cur_area][2])