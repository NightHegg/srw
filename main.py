import os
import sys

from modules.class_basic_method import basic_method
from modules.class_schwarz_multiplicative import schwarz_multiplicative
from modules.class_schwarz_additive import schwarz_additive
from modules.class_schwarz_two_level_additive import schwarz_two_level_additive

if __name__ == "__main__":
    obj = schwarz_two_level_additive(cur_task = 1, cur_mesh = 10, cur_coarse_mesh = 8)
    obj.plot_subds()
    #print(obj.get_info())
    #obj.plot_displacements()