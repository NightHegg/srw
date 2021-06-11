import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from itertools import combinations
import numpy as np
import time
from multiprocessing import Pool, cpu_count
import math

import scr.functions as base_func
from scr.class_schwarz_multiplicative import schwarz_multiplicative


class schwarz_additive(schwarz_multiplicative):

    def __init__(self, data):
        init_time = time.time()
        super().__init__(data)
        self.name_method = "schwarz_additive"
        self.table_name = '$\\text{Аддитивный МДО}$'
        self.coef_alpha = self.data['coef_alpha']
        self.time_init = time.time() - init_time


    def set_initialization(self):
        self.u_previous = self.u.copy()
        self.u_current = self.u.copy()
        self.u_sum = np.zeros_like(self.u)


    def set_additional_calculations(self):
        self.u_sum += self.u_current - self.u_previous
        self.u_current = self.u_previous.copy()


    def get_displacements(self):
        self.u = self.u_previous.copy() + (self.coef_alpha * self.u_sum.copy())


    def set_condition_schwarz(self, function_condition_schwarz):
        function_condition_schwarz(self.u_previous)

if __name__ == "__main__":
    pass
