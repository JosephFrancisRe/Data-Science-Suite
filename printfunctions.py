# -*- coding: utf-8 -*-
"""
@author: JosephRe
"""

import config

def print_results(r):
    print(f"The model predicts with {r.mean_accuracy_past:.2f}% accuracy for the years 2015 through 2021 across {config.simulations} simulation(s).\n")
    print(f"The model predicts with {r.mean_accuracy_current:.2f}% accuracy for current across {config.simulations} simulation(s).\n")
    print(f"The total processing time was {r.process_duration:.2f} seconds to execute.\n")
    print("The model chose to use the following stats to predict homerun totals: " + ", ".join(r.predictor_variable_names) + ".\n")