import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from scr.class_basic_method import basic_method
from scr.class_schwarz_multiplicative import schwarz_multiplicative
from scr.class_schwarz_additive import schwarz_additive
from scr.class_schwarz_two_level_additive import schwarz_two_level_additive


def test_task(method, cur_area, cur_task):
    example_data = {
        'area':             cur_area,
        'task':             cur_task,
        'mesh':             0.05,
        'amnt_subds':       4,
        'coef_convergence': 1e-4,
        'coef_overlap':     0.35,
        'coef_alpha':       0.5,
        'coarse_mesh':      1
    }
    obj = method(example_data)
    # obj.plot_init_coarse_mesh()
    obj.plot_init_mesh()
    obj.get_solution()
    obj.plot_displacements()


def special_error_table(method, cur_area, cur_task, bool_simplified):

    if not os.path.exists(f'results/{cur_area}'):
        os.makedirs(f'results/{cur_area}')

    if not os.path.exists(f'results/{cur_area}/{cur_task}'):
        os.makedirs(f'results/{cur_area}/{cur_task}')
    
    if not os.path.exists(f'results/{cur_area}/{cur_task}/errors_special'):
        os.makedirs(f'results/{cur_area}/{cur_task}/errors_special')
        
    df_rel = {}
    list_coef_convergence = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    for cur_coef_convergence in list_coef_convergence:
        example_data = {
        'area':             cur_area,
        'task':             cur_task,
        'mesh':             0.025,
        'amnt_subds':       8,
        'coef_convergence': cur_coef_convergence,
        'coef_overlap':     0.35,
        'coef_alpha':       0.5,
        'coarse_mesh':      1
        }
        obj = method(example_data)
        name = obj.name_method
        obj.get_solution()
        error_sigma = obj.get_numerical_error_sigma()
        amnt_iters = obj.amnt_iterations
        df_rel[f'{cur_coef_convergence:.0e}'] = {
            'Количество итераций': amnt_iters,
            'Ошибка численного решения для sigma_r': f'{error_sigma[0]:.2e}',
            'Ошибка численного решения для sigma_phi': f'{error_sigma[1]:.2e}'
        }
    df_errors_rel = pd.DataFrame.from_dict(df_rel).T
    print(df_errors_rel)

    df_errors_rel.index.names = ['eps_0']
    
    name_file = name + '_simplified' if bool_simplified and name == 'schwarz_additive_two_level' else name
    route = f'results/{cur_area}/{cur_task}/errors_special/{name_file}.csv'
    df_errors_rel.to_csv(route, index = True)


def simple_error_table(method, cur_area, cur_task, bool_simplified):
    if not os.path.exists(f'results/{cur_area}'):
        os.makedirs(f'results/{cur_area}')

    if not os.path.exists(f'results/{cur_area}/{cur_task}'):
        os.makedirs(f'results/{cur_area}/{cur_task}')
    
    if not os.path.exists(f'results/{cur_area}/{cur_task}/errors'):
        os.makedirs(f'results/{cur_area}/{cur_task}/errors')

    if not os.path.exists(f'results/{cur_area}/{cur_task}/errors_rel'):
        os.makedirs(f'results/{cur_area}/{cur_task}/errors_rel')

    df = {}
    df_rel = {}
    mesh = [0.05, 0.025, 0.0125]
    for cur_mesh in mesh:
        example_data = {
        'area':             cur_area,
        'task':             cur_task,
        'mesh':             cur_mesh,
        'amnt_subds':       2,
        'coef_convergence': 1e-5,
        'coef_overlap':     0.35,
        'coef_alpha':       0.5,
        'coarse_mesh':      1
        }
        obj = method(example_data)
        name = obj.name_method
        obj.get_solution()
        error_u = obj.get_numerical_error_displacement()
        error_sigma = obj.get_numerical_error_sigma()
        df[cur_mesh] = {
            'u_r': f'{error_u:.2e}',
            'sigma_r': f'{error_sigma[0]:.2e}',
            'sigma_phi': f'{error_sigma[1]:.2e}'
        }
        df_rel[cur_mesh] = {
            'u_r': error_u,
            'sigma_r': error_sigma[0],
            'sigma_phi': error_sigma[1]
        }
    value_1 = list(df_rel[mesh[0]].values())
    for key in df_rel.keys():
        df_rel[key]['u_r'] = f'{value_1[0] / df_rel[key]["u_r"]:.0f}'
        df_rel[key]['sigma_r'] = f'{value_1[1] / df_rel[key]["sigma_r"]:.0f}'
        df_rel[key]['sigma_phi'] = f'{value_1[2] / df_rel[key]["sigma_phi"]:.0f}'

    df_errors = pd.DataFrame.from_dict(df).T
    df_errors_rel = pd.DataFrame.from_dict(df_rel).T

    df_errors.index.names = ['Шаг сетки h']
    df_errors_rel.index.names = ['Шаг сетки h']

    print(df_errors)
    print(df_errors_rel)

    name_file = name + '_simplified' if bool_simplified and name == 'schwarz_additive_two_level' else name
    route = f'results/{cur_area}/{cur_task}/errors/{name_file}.csv'
    route_rel = f'results/{cur_area}/{cur_task}/errors_rel/{name_file}.csv'

    df_errors.to_csv(route, index = True)
    df_errors_rel.to_csv(route_rel, index = True)


def get_iters_time_tables(method, cur_area, cur_task, bool_simplified):
    if not os.path.exists(f'results/{cur_area}'):
        os.makedirs(f'results/{cur_area}')

    if not os.path.exists(f'results/{cur_area}/{cur_task}'):
        os.makedirs(f'results/{cur_area}/{cur_task}')

    if not os.path.exists(f'results/{cur_area}/{cur_task}/iterations'):
        os.makedirs(f'results/{cur_area}/{cur_task}/iterations')

    if not os.path.exists(f'results/{cur_area}/{cur_task}/time'):
        os.makedirs(f'results/{cur_area}/{cur_task}/time')

    dict_iters = {}
    dict_time = {}
    list_mesh = [0.05, 0.025, 0.0125]
    list_amnt_subds = [2, 4, 8]

    for cur_mesh in list_mesh:
        dict_iters_temp = {}
        dict_time_temp = {}
        for cur_amnt_subds in list_amnt_subds:
            example_data = {
            'area':             cur_area,
            'task':             cur_task,
            'mesh':             cur_mesh,
            'amnt_subds':       cur_amnt_subds,
            'coef_convergence': 1e-5,
            'coef_overlap':     0.35,
            'coef_alpha':       0.5,
            'coarse_mesh':      1
            }
            obj = method(example_data)
            name = obj.name_method
            obj.get_solution()
            amnt_iters = obj.amnt_iterations
            styled_amnt_subds = f"{cur_amnt_subds} области" if cur_amnt_subds < 5 else f"{cur_amnt_subds} областей"

            dict_iters_temp[styled_amnt_subds] = f'{amnt_iters:.0f}'
            dict_time_temp[styled_amnt_subds] = f'{obj.time_getting_solution:.2f}'
        
        dict_iters[f"h={cur_mesh}"] = dict_iters_temp
        dict_time[f"h={cur_mesh}"] = dict_time_temp
    
    df_iters = pd.DataFrame.from_dict(dict_iters)
    df_time = pd.DataFrame.from_dict(dict_time)

    df_iters.index.names = ['Количество подобластей']
    df_time.index.names = ['Количество подобластей']

    print(df_iters)
    print(df_time)

    name_file = name + '_simplified' if bool_simplified and name == 'schwarz_additive_two_level' else name
    route_iters = f'results/{cur_area}/{cur_task}/iterations/{name_file}.csv'
    route_time = f'results/{cur_area}/{cur_task}/time/{name_file}.csv'

    df_iters.to_csv(route_iters, index = True)
    df_time.to_csv(route_time, index = True)


def special_get_iters_tables(method, cur_area, cur_task, bool_simplified):
    if not os.path.exists(f'results/{cur_area}'):
        os.makedirs(f'results/{cur_area}')

    if not os.path.exists(f'results/{cur_area}/{cur_task}'):
        os.makedirs(f'results/{cur_area}/{cur_task}')

    if not os.path.exists(f'results/{cur_area}/{cur_task}/iters_coef_overlap'):
        os.makedirs(f'results/{cur_area}/{cur_task}/iters_coef_overlap')


    dict_iters = {}
    list_coef_overlap = [0.15, 0.2, 0.25, 0.3, 0.35]
    list_mesh = [0.05, 0.025, 0.0125]
    list_amnt_subds = [2, 4, 8]
    for cur_coef_overlap in list_coef_overlap:
        for cur_mesh in list_mesh:
            dict_iters_temp = {}
            dict_time_temp = {}
            for cur_amnt_subds in list_amnt_subds:
                example_data = {
                'area':             cur_area,
                'task':             cur_task,
                'mesh':             cur_mesh,
                'amnt_subds':       cur_amnt_subds,
                'coef_convergence': 1e-5,
                'coef_overlap':     cur_coef_overlap,
                'coef_alpha':       0.5,
                'coarse_mesh':      0.1
                }
                obj = method(example_data)
                name = obj.name_method
                obj.get_solution()
                amnt_iters = obj.amnt_iterations
                styled_amnt_subds = f"{cur_amnt_subds} области" if cur_amnt_subds < 5 else f"{cur_amnt_subds} областей"

                dict_iters_temp[styled_amnt_subds] = f'{amnt_iters:.0f}'
            
            dict_iters[f"h={cur_mesh}"] = dict_iters_temp
        
        df_iters = pd.DataFrame.from_dict(dict_iters)

        df_iters.index.names = ['Количество подобластей']

        print(df_iters)

        name_file = name + '_simplified' if bool_simplified and name == 'schwarz_additive_two_level' else name
        route_iters = f'results/{cur_area}/{cur_task}/iters_coef_overlap/{name_file}_{cur_coef_overlap:.2e}.csv'

        df_iters.to_csv(route_iters, index = True)


def mesh_size_get_iters_tables(method, cur_area, cur_task, bool_simplified):
    if not os.path.exists(f'results/{cur_area}'):
        os.makedirs(f'results/{cur_area}')

    if not os.path.exists(f'results/{cur_area}/{cur_task}'):
        os.makedirs(f'results/{cur_area}/{cur_task}')

    if not os.path.exists(f'results/{cur_area}/{cur_task}/dif_mesh'):
        os.makedirs(f'results/{cur_area}/{cur_task}/dif_mesh')


    dict_iters = {}
    list_coarse_mesh = [1, 0.5, 0.1]
    list_mesh = [0.05, 0.025, 0.0125]
    list_amnt_subds = [2, 4, 8]
    for cur_coarse_mesh in list_coarse_mesh:
        for cur_mesh in list_mesh:
            dict_iters_temp = {}
            for cur_amnt_subds in list_amnt_subds:
                example_data = {
                'area':             cur_area,
                'task':             cur_task,
                'mesh':             cur_mesh,
                'amnt_subds':       cur_amnt_subds,
                'coef_convergence': 1e-5,
                'coef_overlap':     0.3,
                'coef_alpha':       0.5,
                'coarse_mesh':      cur_coarse_mesh
                }
                obj = method(example_data)
                name = obj.name_method
                obj.get_solution()
                amnt_iters = obj.amnt_iterations
                styled_amnt_subds = f"{cur_amnt_subds} области" if cur_amnt_subds < 5 else f"{cur_amnt_subds} областей"

                dict_iters_temp[styled_amnt_subds] = f'{amnt_iters:.0f}'
            
            dict_iters[f"h={cur_mesh}"] = dict_iters_temp
        
        df_iters = pd.DataFrame.from_dict(dict_iters)

        df_iters.index.names = ['Количество подобластей']

        print(df_iters)

        name_file = name + '_simplified' if bool_simplified and name == 'schwarz_additive_two_level' else name
        route_iters = f'results/{cur_area}/{cur_task}/dif_mesh/{name_file}_{cur_coarse_mesh:.2e}.csv'

        df_iters.to_csv(route_iters, index = True)


if __name__ == "__main__":
    methods = [schwarz_multiplicative, schwarz_additive, schwarz_two_level_additive]
    tasks = {
        'rectangle': ['3_bindings', '2_bindings'], 
        'thick_walled_cylinder': ['pressure_only', 'displacements_only']
    }
    cur_area = 'bearing'
    cur_task = 'pressure_only'
    test_task(basic_method, cur_area, cur_task)
    # bool_simplified = False
    # get_iters_time_tables(schwarz_two_level_additive, cur_area, cur_task, bool_simplified)
    # for cur_method in methods:
    #     bool_simplified = False
    #     special_error_table(cur_method, cur_area, cur_task, bool_simplified)
    #     print(f'Method {cur_method} finished!')
    # special_get_iters_tables(schwarz_two_level_additive, cur_area, cur_task, bool_simplified = False)
    # bool_simplified = True
    # simple_error_table(schwarz_two_level_additive, cur_area, cur_task, bool_simplified)