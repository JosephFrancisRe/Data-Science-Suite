# -*- coding: utf-8 -*-
"""
@author: JosephRe
"""

# Libraries
from pybaseball import batting_stats
import time

# Time
start_year = 2015
end_year = 2022
start_time = None
end_time = time.time()

# Options
required_qualification = 550
uniqueIDColumnName = 'IDfg'
target_variable_name = 'HR'
predicator_variable_num = 5
simulations = 1
active_library = 1

# Dictionaries
library_mappings = {
        1: batting_stats
}

# Randomn
trueRandomness = True
seed = None

# Models
lr = None

# Datasets
dataset_past = None
dataset_current = None

# Containers
predictor_variable_names = []
y_predictions = []
X_trains = []
X_tests = []
y_trains = []
y_tests = []
X_tests_current = []
y_tests_current = []
lr_fits = []
lr_accuracies_past = []
lr_accuracies_current = []

# Temp Files to Fill Containers
X_test_current = None
y_test_current = None