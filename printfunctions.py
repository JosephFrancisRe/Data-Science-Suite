# -*- coding: utf-8 -*-
"""
@author: JosephRe
"""


def print_results(r):
    print(f"The model predicts with {r.mean_accuracy_past:.2f}% accuracy for the years {r.start_year} through {r.end_year - 1} across {r.simulations} simulation(s).\n")
    print(f"The model predicts with {r.mean_accuracy_current:.2f}% accuracy for {r.end_year} current across {r.simulations} simulation(s).\n")
    print(f"The total processing time was {r.process_duration:.2f} seconds to execute.\n")
    print("The model chose to use the following stats to predict homerun totals: " + ", ".join(r.predictor_variable_names) + ".\n")
    