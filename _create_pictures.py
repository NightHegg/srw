import os
import sys

import numpy as np
import pandas as pd

from scr.class_basic_method import basic_method
from scr.class_schwarz_multiplicative import schwarz_multiplicative
from scr.class_schwarz_additive import schwarz_additive
from scr.class_schwarz_two_level_additive import schwarz_two_level_additive


def test_task(method, fine_area, coarse_area, fine_mesh, course_mesh, cur_task):
    example_data = {
        'fine_area':        fine_area,
        'coarse_area':      coarse_area,
        'fine_mesh':        fine_mesh,
        'coarse_mesh':      course_mesh,
        'task':             cur_task,
        'amnt_subds':       2,
        'coef_convergence': 1e-5,
        'coef_overlap':     0.3,
        'coef_alpha':       0.5
    }
    obj = method(example_data)
    obj.get_solution()
    obj.plot_displacement_distribution(True)


if __name__ == "__main__":
    areas = ["rectangle", "thick_walled_cylinder", "simplified_cylinder", "bearing"]
    tasks = ["3_fixes", "2_fixes", "pressure_only"]

    area = {
        'rectangle' : ['3_fixes', '2_fixes'],
        'thick_walled_cylinder' : ['pressure_only'],
        'bearing' : ['pressure_only']
    }

    fine_area = 'bearing'
    course_area = 'rectangle'

    fine_mesh = 0.0125
    course_mesh = 1

    cur_task = 'pressure_only'
    test_task(basic_method, fine_area, course_area, fine_mesh, course_mesh, cur_task)
    # for fine_area, tasks in area.items():
    #     for task in tasks:
    #         test_task(basic_method, fine_area, course_area, fine_mesh, course_mesh, task)
    #         print(f'Ended {task}: {fine_area}')