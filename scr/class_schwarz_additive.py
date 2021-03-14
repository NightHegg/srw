import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from itertools import combinations
import numpy as np
from scipy.sparse import linalg

import scr.functions as base_func
from scr.class_schwarz_multiplicative import schwarz_multiplicative


class schwarz_additive(schwarz_multiplicative):
    def __init__(self, data):
        super().__init__(data)
        
        self.name_method = "schwarz additive method"
        self.coef_alpha = data['coef_alpha']


    def internal_initialize_displacements(self):
        self.u_previous = np.copy(self.u)
        self.u_current = np.copy(self.u)
        self.u_sum = np.zeros_like(self.u)


    def set_condition_schwarz(self, idv, array_condition):
        return super().set_condition_schwarz(idv, self.u_previous)


    def internal_additional_calculations(self):
        self.u_sum += (self.u_current - self.u_previous)
        self.u_current = np.copy(self.u_previous)


    def interal_final_calculate_u(self):
        self.u = np.copy(self.u_previous) + (self.coef_alpha * np.copy(self.u_sum))


if __name__ == "__main__":
    pass
