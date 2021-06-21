import os

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import copy

from scr.class_basic_method import basic_method
from scr.class_schwarz_multiplicative import schwarz_multiplicative
from scr.class_schwarz_additive import schwarz_additive
from scr.class_schwarz_two_level_additive import schwarz_two_level_additive


def create_table_rectangle_error(task):
    dct = {}
    mesh = [0.05, 0.025, 0.0125, 0.00625]
    for cur_mesh in mesh:
        example_data = {
            'fine_area':        'rectangle',
            'coarse_area':      '',
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
        dct[cur_mesh] = {
            'sigmay': f'{(np.amax(np.abs(obj.sigma[:, 1]), axis = 0) - 2e+7) / (2e+7):.2e}'
        }
        print(f'Ended {cur_mesh}')
    df = pd.DataFrame.from_dict(dct).T

    df.index.names = ['step']
    print(df)

    route = f'results/rectangle/{task}/core/errors.csv'
    df.to_csv(route, index = True)


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
        dct_rel[key]['ur'] = f'{value_1[0] / dct_rel[key]["ur"]:.2f}'
        dct_rel[key]['sigmar'] = f'{value_1[1] / dct_rel[key]["sigmar"]:.2f}'
        dct_rel[key]['sigmaphi'] = f'{value_1[2] / dct_rel[key]["sigmaphi"]:.2f}'

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
    lst_coarse_mesh = [0.5, 0.25, 0.125]
    list_amnt_subds = [2, 4, 8]

    for idx, coarse_mesh in enumerate(lst_coarse_mesh):
        dct_temp = {}
        for cur_amnt_subds in list_amnt_subds:
            example_data = {
                'fine_area':        fine_area,
                'coarse_area':      coarse_area,
                'fine_mesh':        0.025,
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
                'fine_mesh':        0.0125,
                'coarse_mesh':      0.125,
                'task':             task,
                'amnt_subds':       amnt_subds,
                'coef_convergence': 1e-5,
                'coef_overlap':     coef_overlap,
                'coef_alpha':       0.5
            }
            obj = method(example_data)
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

    route = f'results/{fine_area}/{task}/core/iters_overlap.csv'
    df.to_csv(route, index = True)


def create_table_cg(fine_area, coarse_area, task, table_save = False, pic_save = False):
    dct = {}
    dct_time = {}
    mesh = [0.05, 0.025, 0.0125, 0.00625]
    check_N = 0
    for idx, cur_mesh in enumerate(mesh):
        example_data = {
            'fine_area':        fine_area,
            'coarse_area':      coarse_area,
            'fine_mesh':        cur_mesh,
            'coarse_mesh':      0.125,
            'task':             task,
            'amnt_subds':       8,
            'coef_convergence': 1e-5,
            'coef_overlap':     0.3,
            'coef_alpha':       0.5
        }
        example_data_2 = {
            'fine_area':        fine_area,
            'coarse_area':      coarse_area,
            'fine_mesh':        cur_mesh,
            'coarse_mesh':      0.125,
            'task':             task,
            'amnt_subds':       2,
            'coef_convergence': 1e-5,
            'coef_overlap':     0.3,
            'coef_alpha':       0.5
        }

        example_data_3 = {
            'fine_area':        fine_area,
            'coarse_area':      coarse_area,
            'fine_mesh':        cur_mesh,
            'coarse_mesh':      0.125,
            'task':             task,
            'amnt_subds':       4,
            'coef_convergence': 1e-5,
            'coef_overlap':     0.3,
            'coef_alpha':       0.5
        }

        obj = basic_method(example_data)
        obj.get_solution()
        amnt_iters = obj.amnt_iters_cg
        time = obj.time_u + obj.time_init

        obj1 = schwarz_two_level_additive(example_data)
        obj1.get_solution()
        amnt_iters_sc8 = obj1.amnt_iters_cg
        time_sc8 = obj1.time_u

        if idx == 0:
            check_N = obj.N
        if fine_area == 'bearing':
            obj2 = schwarz_two_level_additive(example_data_2)
            obj2.get_solution()
            amnt_iters_sc2 = obj2.amnt_iters_cg
            time_sc2 = obj2.time_u

            obj3 = schwarz_two_level_additive(example_data_3)
            obj3.get_solution()
            amnt_iters_sc4 = obj3.amnt_iters_cg
            time_sc4 = obj3.time_u

            dct[obj.N] = {
                'N': obj.N,
                'theory': obj.N**(1/2),
                'basic': amnt_iters,
                '2': amnt_iters_sc2,
                '4': amnt_iters_sc4,
                '8': amnt_iters_sc8
            }
            dct_time[obj.N] = {
                'N': obj.N,
                'theory': obj.N**(3/2),
                'basic': time,
                '2': time_sc2,
                '4': time_sc4,
                '8': time_sc8
            }
        else:
            dct[obj.N] = {
                'N': obj.N,
                'theory': obj.N**(1/2),
                'basic': amnt_iters,
                '8': amnt_iters_sc8
            }
            dct_time[obj.N] = {
                'N': obj.N,
                'theory': obj.N**(3/2),
                'basic': time,
                '8': time_sc8
            }
    dct_rel = copy.deepcopy(dct)
    dct_time_rel = copy.deepcopy(dct_time)

    value_1 = list(dct_rel[check_N].values())
    for key, value in dct_rel.items():
        for idx, k in enumerate(value.keys()):
            dct_rel[key][k] = dct_rel[key][k] / value_1[idx]

    value_2 = list(dct_time_rel[check_N].values())
    for key, value in dct_time_rel.items():
        for idx, k in enumerate(value.keys()):
            dct_time_rel[key][k] = dct_time_rel[key][k] / value_2[idx]

    df = pd.DataFrame.from_dict(dct).T
    df_time = pd.DataFrame.from_dict(dct_time).T
    df_rel = pd.DataFrame.from_dict(dct_rel).T
    df_time_rel = pd.DataFrame.from_dict(dct_time_rel).T

    df.index.names = ['index']
    df_time.index.names = ['index']
    df_rel.index.names = ['index']
    df_time_rel.index.names = ['index']

    df.drop(columns = ["N", "theory"], inplace=True)
    for name in df.columns.tolist():
        df[name] = df[name].map('{:.0f}'.format)

    df_time.drop(columns = ["N", "theory"], inplace=True)
    for name in df_time.columns.tolist():
        df_time[name] = df_time[name].map('{:.2f}'.format)

    for name in df_rel.columns.tolist():
        df_rel[name] = df_rel[name].map('{:.2f}'.format)

    for name in df_time_rel.columns.tolist():
        df_time_rel[name] = df_time_rel[name].map('{:.2f}'.format)
    
    if table_save:
        route = f'results/{fine_area}/{task}/core/iters_cg.csv'
        route_time = f'results/{fine_area}/{task}/core/time_cg.csv'
        route_rel = f'results/{fine_area}/{task}/core/iters_cg_rel.csv'
        route_time_rel = f'results/{fine_area}/{task}/core/time_cg_rel.csv'

        df.to_csv(route)
        df_time.to_csv(route_time)
        df_rel.to_csv(route_rel)
        df_time_rel.to_csv(route_time_rel)
    print(df)
    print(df_time)
    print(df_rel)
    print(df_time_rel)

    fig, ax = plt.subplots()

    ax.plot(df_rel["N"].astype('float64'), df_rel["theory"].astype('float64'), label = 'Теория')
    ax.plot(df_rel["N"].astype('float64'), df_rel["basic"].astype('float64'), "x", ms = 14, label = "Базовый метод без МДО")
    if fine_area == 'bearing':
        ax.plot(df_rel["N"].astype('float64'), df_rel["2"].astype('float64'), "s", ms = 10, label = "Двухуровневый аддитивный МДО (M = 2)", mfc = 'r')
        ax.plot(df_rel["N"].astype('float64'), df_rel["4"].astype('float64'), "D", ms = 10, label = "Двухуровневый аддитивный МДО (M = 4)", mfc = 'c')
    ax.plot(df_rel["N"].astype('float64'), df_rel["8"].astype('float64'), "o", ms = 10, label = "Двухуровневый аддитивный МДО (M = 8)", mfc = 'g')

    ax.legend(loc = 'best', prop={'size': 15})

    fig.set_figwidth(10)
    fig.set_figheight(8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(labelsize = 14)

    ax.set_xlabel('$\\frac{n}{n_1}$', fontsize = 22)
    ax.set_ylabel('$\\frac{m}{m_1}$', fontsize = 22)

    if pic_save:
        route = f'results/{fine_area}/{task}/core/iters_cg.png'
        plt.savefig(route)
    else:
        plt.show()

    fig1, ax1 = plt.subplots()

    ax1.plot(df_time_rel["N"].astype('float64'), df_time_rel["theory"].astype('float64'), label = 'Теория')
    ax1.plot(df_time_rel['N'].astype('float64'), df_time_rel["basic"].astype('float64'), "x", ms = 14, label = "Базовый метод")
    if fine_area == 'bearing':
        ax1.plot(df_time_rel["N"].astype('float64'), df_time_rel["2"].astype('float64'), "s", ms = 10, label = "Двухуровневый аддитивный МДО (M = 2)", mfc = 'r')
        ax1.plot(df_time_rel["N"].astype('float64'), df_time_rel["4"].astype('float64'), "D", ms = 10, label = "Двухуровневый аддитивный МДО (M = 4)", mfc = 'c')
    ax1.plot(df_time_rel["N"].astype('float64'), df_time_rel["8"].astype('float64'), "o", ms = 10, label = "Двухуровневый аддитивный МДО (M = 8)", mfc = 'g')

    ax1.legend(loc = 'best', prop={'size': 15})

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.set_xlabel('$\\frac{n}{n_1}$', fontsize = 22)
    ax1.set_ylabel('$\\frac{m}{m_1}$', fontsize = 22)

    ax1.tick_params(labelsize = 14)

    fig1.set_figwidth(10)
    fig1.set_figheight(8)

    if pic_save:
        route = f'results/{fine_area}/{task}/core/time_cg.png'
        plt.savefig(route)
    else:
        plt.show()


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
            'coarse': 'bearing'
        }
    }
    create_table_cg('bearing', 'bearing', 'pressure_only', table_save=True, pic_save=True)
    # create_table_iters_coarse('thick_walled_cylinder', 'thick_walled_cylinder', 'pressure_only', bool_save=True)
    # create_table_iters_coarse('bearing', 'bearing', 'pressure_only', bool_save=True)

    # create_table_overlap('thick_walled_cylinder', 'thick_walled_cylinder', 4, 'pressure_only')
    # create_table_overlap('bearing', 'bearing', 4, 'pressure_only')

    # for fine_area, data in area_simple.items():
    #     for task in data["tasks"]:
    #         create_table_cg(fine_area, data["coarse"], task, table_save=True, pic_save=True)
    #     print('Finished:', fine_area)

    # create_table_iters_coarse('bearing', 'bearing', 'pressure_only', True)
    # create_table_special_error(schwarz_multiplicative, 'thick_walled_cylinder', 'rectangle', 'pressure_only')
    # create_table_special_error(schwarz_additive, 'thick_walled_cylinder', 'rectangle', 'pressure_only')

    # create_table_iters_time(schwarz_two_level_additive, 'rectangle', 'rectangle', '3_fixes', bool_save=True)
    # for cur_method in methods:
    #     for fine_area, data in area_simple.items():
    #         for task in data["tasks"]:
    #             create_table_iters_time(cur_method, fine_area, data["coarse"], task, bool_save=True)
    #             create_table_overlap(cur_method, fine_area, data["coarse"], task, bool_save=True)
    #     print(f'Method {cur_method} finished!')
    # for fine_area, data in area_coarse.items():
    #     for task in data["tasks"]:
    #         for coarse_area in data["coarse"]:
    #             get_iters_time_tables(schwarz_two_level_additive, fine_area, coarse_area, task)
    # create_table_overlap('rectangle', 'rectangle', 4, '3_fixes')
    # create_table_iters_time(schwarz_two_level_additive, 'bearing', 'bearing', 'pressure_only', False)
    # create_table_rectangle_error('3_fixes')
    # for fine_area, data in area_coarse.items():
    #     for task in data["tasks"]:
    #         for coarse_area in data["coarse"]:
    #             create_table_overlap(fine_area, coarse_area, 4, task)

    # create_table_simple_error('thick_walled_cylinder', 'pressure_only')
    # for fine_area, data in area_solution.items():
    #     for task in data["tasks"]:
    #         for coarse_area in data["coarse"]:
    #             create_table_simple_error(fine_area, coarse_area, 4, task)