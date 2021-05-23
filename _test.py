from scr.class_basic_method import basic_method
from scr.class_schwarz_multiplicative import schwarz_multiplicative
from scr.class_schwarz_additive import schwarz_additive
from scr.class_schwarz_two_level_additive import schwarz_two_level_additive


def test_task(method, cur_area, cur_task):
    example_data = {
        'area':             cur_area,
        'coarse_area':      'cur_area',
        'task':             cur_task,
        'mesh':             0.05,
        'amnt_subds':       2,
        'coef_convergence': 1e-4,
        'coef_overlap':     0.35,
        'coef_alpha':       0.5,
        'coarse_mesh':      0.5
    }
    obj = method(example_data)
    # obj.plot_init_mesh()
    obj.get_solution()
    obj.pcolormesh()
    obj.plot_displacements()


if __name__ == "__main__":
    areas = ['rectangle', 'thick_walled_cylinder', 'simplified_cylinder', 'bearing']
    tasks = ['3_fixes', '2_fixes', 'pressure_only']
    cur_area = 'thick_walled_cylinder'
    cur_task = 'pressure_only'
    test_task(schwarz_multiplicative, cur_area, cur_task)