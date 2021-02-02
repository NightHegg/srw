import os
import sys

sys.path.append(os.path.abspath("modules"))
from class_basic_method import basic_method
from class_schwarz_multiplicative import schwarz_multiplicative

if __name__ == "__main__":
    obj = schwarz_multiplicative(cur_task = 1, cur_mesh = 10, cur_amnt_subds = [2, 1], coef_convergence = 1e-3)
    obj.get_solution()
    print(obj.get_info())
    obj.plot_displacements()