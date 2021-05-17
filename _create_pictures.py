import os
import sys

import numpy as np
import pandas as pd

from scr.class_basic_method import basic_method
from scr.class_schwarz_multiplicative import schwarz_multiplicative
from scr.class_schwarz_additive import schwarz_additive
from scr.class_schwarz_two_level_additive import schwarz_two_level_additive


def test_task(method, cur_area, cur_task):
    example_data = {
        'area':             cur_area,
        'task':             cur_task,
        'mesh':             0.05,
        'amnt_subds':       2,
        'coef_convergence': 1e-4,
        'coef_overlap':     0.35,
        'coef_alpha':       0.5,
        'coarse_mesh':      0.5
    }
    obj = method(example_data)
    save_route = f'srw_text/img/bearing_init'
    obj.plot_init_mesh(save_route)


if __name__ == "__main__":
    tasks = {
        'rectangle': ['3_bindings', '2_bindings'], 
        'thick_walled_cylinder': ['pressure_only', 'displacements_only']
    }
    cur_area = 'bearing'
    cur_task = 'pressure_only'
    test_task(basic_method, cur_area, cur_task)