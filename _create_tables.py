import os

import numpy as np
import pandas as pd

from scr.class_basic_method import basic_method
from scr.class_schwarz_multiplicative import schwarz_multiplicative
from scr.class_schwarz_additive import schwarz_additive
from scr.class_schwarz_two_level_additive import schwarz_two_level_additive


def create_table_special_error(method, fine_area, coarse_area, task):
    dct = {}
    list_coef_convergence = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    for coef_convergence in list_coef_convergence:
        example_data = {
            'fine_area':        fine_area,
            'coarse_area':      coarse_area,
            'fine_mesh':        0.0125,
            'coarse_mesh':      0.5,
            'task':             task,
            'amnt_subds':       8,
            'coef_convergence': coef_convergence,
            'coef_overlap':     0.3,
            'coef_alpha':       0.5
        }
        obj = method(example_data)
        name = obj.name_method
        obj.get_solution()
        error_sigma = obj.get_numerical_error_sigma()
        amnt_iters = obj.amnt_iterations
        dct[f'{coef_convergence:.0e}'] = {
            'amnt': amnt_iters,
            'sigmar': f'{error_sigma[0]:.2e}',
            'sigmaphi': f'{error_sigma[1]:.2e}'
        }
    df = pd.DataFrame.from_dict(dct).T
    df.index.names = ['coefconv']
    print(df)
    if method == schwarz_two_level_additive:
        route = f'results/{fine_area}/{task}/{name}/errors_special_{coarse_area}.csv'
    else:
        route = f'results/{fine_area}/{task}/{name}/errors_special.csv'
    df.to_csv(route, index = True)


def create_table_simple_error(fine_area, task):
    dct = {}
    dct_rel = {}
    mesh = [0.05, 0.025, 0.0125]
    for cur_mesh in mesh:
        example_data = {
            'fine_area':        fine_area,
            'coarse_area':      'simplified_cylinder',
            'fine_mesh':        cur_mesh,
            'coarse_mesh':      0.5,
            'task':             task,
            'amnt_subds':       2,
            'coef_convergence': 1e-5,
            'coef_overlap':     0.3,
            'coef_alpha':       0.5
        }
        obj = basic_method(example_data)
        name = obj.name_method
        obj.get_solution()
        error_u = obj.get_numerical_error_displacement()
        error_sigma = obj.get_numerical_error_sigma()
        dct[cur_mesh] = {
            'ur': f'{error_u:.2e}',
            'sigmar': f'{error_sigma[0]:.2e}',
            'sigmaphi': f'{error_sigma[1]:.2e}'
        }
        dct_rel[cur_mesh] = {
            'ur': error_u,
            'sigmar': error_sigma[0],
            'sigmaphi': error_sigma[1]
        }
    value_1 = list(dct_rel[mesh[0]].values())
    for key in dct_rel.keys():
        dct_rel[key]['ur'] = f'{value_1[0] / dct_rel[key]["ur"]:.0f}'
        dct_rel[key]['sigmar'] = f'{value_1[1] / dct_rel[key]["sigmar"]:.0f}'
        dct_rel[key]['sigmaphi'] = f'{value_1[2] / dct_rel[key]["sigmaphi"]:.0f}'

    df = pd.DataFrame.from_dict(dct).T
    df_rel = pd.DataFrame.from_dict(dct_rel).T

    df.index.names = ['step']
    df_rel.index.names = ['step']
    print(df)
    print(df_rel)

    if not os.path.exists(f'results/{fine_area}/{task}/{name}'):
        os.makedirs(f'results/{fine_area}/{task}/{name}')

    route = f'results/{fine_area}/{task}/{name}/errors.csv'
    route_rel = f'results/{fine_area}/{task}/{name}/errors_rel.csv'
    df.to_csv(route, index = True)
    df_rel.to_csv(route_rel, index = True)


def create_table_iters_coarse(fine_area, coarse_area, task, bool_save = False):
    dct = {}
    lst_coarse_mesh = [1, 0.5, 0.25, 0.125]
    list_amnt_subds = [2, 4, 8]

    for idx, coarse_mesh in enumerate(lst_coarse_mesh):
        dct_temp = {}
        for cur_amnt_subds in list_amnt_subds:
            example_data = {
                'fine_area':        fine_area,
                'coarse_area':      coarse_area,
                'fine_mesh':        0.0125,
                'coarse_mesh':      coarse_mesh,
                'task':             task,
                'amnt_subds':       cur_amnt_subds,
                'coef_convergence': 1e-5,
                'coef_overlap':     0.3,
                'coef_alpha':       0.5
            }
            obj = schwarz_two_level_additive(example_data)
            obj.get_solution()
            amnt_iters = obj.amnt_iterations
            styled_amnt_subds = f"{cur_amnt_subds} области" if cur_amnt_subds < 5 else f"{cur_amnt_subds} областей"

            dct_temp[styled_amnt_subds] = f'{amnt_iters}'

            print(f'Ended {coarse_mesh} : {cur_amnt_subds}!')

        dct[idx] = dct_temp
    
    df = pd.DataFrame.from_dict(dct)

    df.index.names = ['amnt']

    if bool_save:
        if not os.path.exists(f'results/{fine_area}/{task}/schwarz_two_level_additive'):
            os.makedirs(f'results/{fine_area}/{task}/schwarz_two_level_additive')
        route = f'results/{fine_area}/{task}/schwarz_two_level_additive/iters_{coarse_area}.csv'
        df.to_csv(route, index = True)
    else:
        print(df)


def create_table_iters_time(method, fine_area, coarse_area, task, bool_save = False):
    dict_iters = {}
    dict_time = {}
    list_mesh = [0.05, 0.025, 0.0125, 0.00625]
    list_amnt_subds = [2, 4, 8]

    for idx, cur_mesh in enumerate(list_mesh):
        dict_iters_temp = {}
        dict_time_temp = {}
        for cur_amnt_subds in list_amnt_subds:
            example_data = {
                'fine_area':        fine_area,
                'coarse_area':      coarse_area,
                'fine_mesh':        cur_mesh,
                'coarse_mesh':      0.125,
                'task':             task,
                'amnt_subds':       cur_amnt_subds,
                'coef_convergence': 1e-5,
                'coef_overlap':     0.3,
                'coef_alpha':       0.5
            }
            obj = method(example_data)
            name = obj.name_method
            obj.get_solution()
            amnt_iters = obj.amnt_iterations
            styled_amnt_subds = f"{cur_amnt_subds} области" if cur_amnt_subds < 5 else f"{cur_amnt_subds} областей"

            dict_iters_temp[styled_amnt_subds] = f'{amnt_iters}'
            dict_time_temp[styled_amnt_subds] = f'{obj.time_global:.2f}'

            print(f'Ended {cur_mesh} : {cur_amnt_subds}!')

        dict_iters[idx] = dict_iters_temp
        dict_time[idx] = dict_time_temp
    
    df_iters = pd.DataFrame.from_dict(dict_iters)
    df_time = pd.DataFrame.from_dict(dict_time)

    df_iters.index.names = ['amnt']
    df_time.index.names = ['amnt']

    if bool_save:
        if not os.path.exists(f'results/{fine_area}/{task}/{name}'):
            os.makedirs(f'results/{fine_area}/{task}/{name}')

        route_iters = f'results/{fine_area}/{task}/{name}/iters.csv'
        route_time = f'results/{fine_area}/{task}/{name}/time.csv'

        df_iters.to_csv(route_iters, index = True)
        df_time.to_csv(route_time, index = True)
    else:
        print(df_iters)
        print(df_time)


def create_table_overlap(fine_area, coarse_area, amnt_subds, task):
    dct = {}
    list_methods = [schwarz_multiplicative, schwarz_additive, schwarz_two_level_additive]
    list_overlap = [0.2, 0.3, 0.4]

    for method in list_methods:
        dct_temp = {}
        for idx, coef_overlap in enumerate(list_overlap):
            example_data = {
                'fine_area':        fine_area,
                'coarse_area':      coarse_area,
                'fine_mesh':        0.05,
                'coarse_mesh':      0.5,
                'task':             task,
                'amnt_subds':       amnt_subds,
                'coef_convergence': 1e-5,
                'coef_overlap':     coef_overlap,
                'coef_alpha':       0.5
            }
            obj = method(example_data)
            name = obj.name_method
            table_name = obj.table_name
            obj.get_solution()
            amnt_iters = obj.amnt_iterations

            dct_temp[f'{idx}'] = f'{amnt_iters:.0f}'
            print(f'Ended {method} : {coef_overlap}')

        dct[table_name] = dct_temp
    
    df = pd.DataFrame.from_dict(dct).T
    df.index.names = ['method']
    print(df)

    if not os.path.exists(f'results/{fine_area}/{task}/core'):
        os.makedirs(f'results/{fine_area}/{task}/core')

    route = f'results/{fine_area}/{task}/core/overlap_{coarse_area}.csv'
    df.to_csv(route, index = True)


def create_table_cg(fine_area, task):
    if not os.path.exists(f'results/{fine_area}/{task}/basic_method'):
        os.makedirs(f'results/{fine_area}/{task}/basic_method')

    df = {}
    mesh = [0.05, 0.025, 0.0125]
    for cur_mesh in mesh:
        example_data = {
            'fine_area':        fine_area,
            'coarse_area':      'rectangle',
            'fine_mesh':        cur_mesh,
            'coarse_mesh':      0.5,
            'task':             task,
            'amnt_subds':       2,
            'coef_convergence': 1e-5,
            'coef_overlap':     0.3,
            'coef_alpha':       0.5
        }
        obj = basic_method(example_data)
        obj.get_solution()
        amnt_iters = obj.amnt_iters_cg
        df[cur_mesh] = {
            'amnt_iters': amnt_iters
        }
    df_iters = pd.DataFrame.from_dict(df)
    df_iters.index.names = ['amnt']
    print(df_iters)
    df_iters.to_csv(f'results/{fine_area}/{task}/basic_method/iters_cg.csv', index = True)


if __name__ == "__main__":
    areas = ["rectangle", "thick_walled_cylinder", "simplified_cylinder", "bearing"]
    tasks = ["3_fixes", "2_fixes", "pressure_only"]
    methods = [schwarz_multiplicative, schwarz_additive, schwarz_two_level_additive]

    areas = {
        'rectangle' : {
            'tasks': ['3_fixes', '2_fixes'],
            'coarse': ['rectangle']
        },
        'thick_walled_cylinder' : {
            'tasks': ['pressure_only'],
            'coarse': ['thick_walled_cylinder', 'simplified_cylinder']
        },
        'bearing' : {
            'tasks': ['pressure_only'],
            'coarse': ['thick_walled_cylinder', 'simplified_cylinder']
        }
    }

    area_simple = {
        'rectangle' : {
            'tasks': ['3_fixes', '2_fixes'],
            'coarse': 'rectangle'
        },
        'thick_walled_cylinder' : {
            'tasks': ['pressure_only'],
            'coarse': 'thick_walled_cylinder'
        },
        'bearing' : {
            'tasks': ['pressure_only'],
            'coarse': 'thick_walled_cylinder'
        }
    }
    # create_table_iters_coarse(schwarz_two_level_additive, 'thick_walled_cylinder', 'thick_walled_cylinder', 'pressure_only', False)
    # create_table_special_error(schwarz_multiplicative, 'thick_walled_cylinder', 'rectangle', 'pressure_only')
    # create_table_special_error(schwarz_additive, 'thick_walled_cylinder', 'rectangle', 'pressure_only')

    # for fine_area, data in areas.items():
    #     for task in data['tasks']:
    #         for coarse_area in data["coarse"]:
    #             create_table_iters_coarse(fine_area, coarse_area, task, bool_save=True)
    #             print('Finished:', coarse_area, task, fine_area)

    for cur_method in methods:
        for fine_area, data in area_simple.items():
            for task in data["tasks"]:
                create_table_iters_time(cur_method, fine_area, data["coarse"], task, bool_save=True)
        print(f'Method {cur_method} finished!')

    # for fine_area, data in area_coarse.items():
    #     for task in data["tasks"]:
    #         for coarse_area in data["coarse"]:
    #             get_iters_time_tables(schwarz_two_level_additive, fine_area, coarse_area, task)

    # for fine_area, data in area_coarse.items():
    #     for task in data["tasks"]:
    #         for coarse_area in data["coarse"]:
    #             create_table_overlap(fine_area, coarse_area, 4, task)

    # create_table_simple_error('thick_walled_cylinder', 'pressure_only')
    # for fine_area, data in area_solution.items():
    #     for task in data["tasks"]:
    #         for coarse_area in data["coarse"]:
    #             create_table_simple_error(fine_area, coarse_area, 4, task)