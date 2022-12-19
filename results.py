# -*- coding: utf-8 -*-
"""
@author: JosephRe
"""

import printfunctions as pf

class Results:
    def __init__(self, maccc, macp, pvn, pd):
        self.mean_accuracy_current = maccc
        self.mean_accuracy_past = macp
        self.predictor_variable_names = pvn
        self.process_duration = pd
        
    def set_mean_accuracy_past(self, param):
        self.mean_accuracy_past = param
        
    def set_mean_accuracy_current(self, param):
        self.mean_accuracy_current = param
        
    def set_predictor_variable_names(self, param):
        self.predictor_variable_names = param
        
    def set_process_duration(self, pd):
        self.process_duration = pd
        
    def print_results(self):
        pf.print_results(self)