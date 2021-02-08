import os
import sys

from scr.class_basic_method import basic_method
from scr.class_schwarz_multiplicative import schwarz_multiplicative
from scr.class_schwarz_additive import schwarz_additive
from scr.class_schwarz_two_level_additive import schwarz_two_level_additive

if __name__ == "__main__":
    obj = schwarz_multiplicative(cur_task = 1, cur_mesh = 20, cur_amnt_subds = [2, 1])
    obj.get_solution()
    print(obj.get_info())
    #obj.plot_displacements()