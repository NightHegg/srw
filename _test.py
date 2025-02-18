import numpy as np

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
    # obj.plot_area_init_mesh()
    # obj.plot_area_init_coarse_mesh()
    obj.get_solution()
    obj.analysis_time()
    obj.plot_displacement()


if __name__ == "__main__":
    areas = ['rectangle', 'thick_walled_cylinder', 'simplified_cylinder', 'bearing']
    tasks = ['3_fixes', '2_fixes', 'pressure_only']

    cur_task = 'pressure_only'

    fine_area = 'thick_walled_cylinder'
    course_area = 'thick_walled_cylinder'

    fine_mesh = 0.05
    course_mesh = 0.125
    
    test_task(schwarz_multiplicative, fine_area, course_area, fine_mesh, course_mesh, cur_task)