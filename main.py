import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from scr.class_basic_method import basic_method
from scr.class_schwarz_multiplicative import schwarz_multiplicative
from scr.class_schwarz_additive import schwarz_additive
from scr.class_schwarz_two_level_additive import schwarz_two_level_additive

def test():
    example_data = {
        'area':             'area_01',
        'task':             'task_01',
        'mesh':             '7.5e-04',
        'amnt_subds':       [2, 1],
        'coef_convergence': 1e-3,
        'coef_overlap':     0.35,
        'coef_alpha':       0.5,
        'coarse_mesh':      '1.5e-03'
    }
    obj = schwarz_two_level_additive(example_data)
    obj.get_solution()
    print(obj.get_info())


def task_iters_sigma():
    data = {
        'area':            'area_01',
        'task':            'task_01',
        'mesh_list':       ["5.0e-04", "2.5e-04", "1.3e-04"],
        'amnt_subds_list': [[2, 1], [4, 1], [8, 1]],
    }

    if not os.path.exists(f'results/{data["area"]}'):
        os.makedirs(f'results/{data["area"]}')

    if not os.path.exists(f'results/{data["area"]}/{data["task"]}'):
        os.makedirs(f'results/{data["area"]}/{data["task"]}')
        os.makedirs(f'results/{data["area"]}/{data["task"]}/iterations')
        os.makedirs(f'results/{data["area"]}/{data["task"]}/sigma')

    dict_iters = {}
    dict_sigma = {}

    for cur_mesh in tqdm(data["mesh_list"]):
        dict_iters_temp = {}
        dict_sigma_temp = {}
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

            obj = schwarz_two_level_additive(basic_data)
            obj.get_solution()

            array_amnt_subds = np.prod(np.array(cur_amnt_subds))
            styled_amnt_subds = f"{array_amnt_subds} области" if array_amnt_subds < 5 else f"{array_amnt_subds} областей"

            dict_iters_temp[styled_amnt_subds] = obj.amnt_iterations
            dict_sigma_temp[styled_amnt_subds] = obj.get_special_sigma()
        
        dict_iters[f"h={cur_mesh}"] = dict_iters_temp
        dict_sigma[f"h={cur_mesh}"] = dict_sigma_temp
    
    df_iters = pd.DataFrame.from_dict(dict_iters)
    df_sigma = pd.DataFrame.from_dict(dict_sigma)

    print(df_iters)
    print(df_sigma)

    df_iters.to_csv(f'results/{data["area"]}/{data["task"]}/iterations/two_level.csv')
    df_sigma.to_csv(f'results/{data["area"]}/{data["task"]}/sigma/two_level.csv')


if __name__ == "__main__":
    task_iters_sigma()