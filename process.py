# -*- coding: utf-8 -*-
"""
@author: JosephRe
"""

import print_functions as pf
import time

import config

class Process:
    def __init__(self):
        # Settings
        self.__required_qualification = config.required_qualification
        self.__observation_unique_ID = config.observation_unique_ID
        self.__target_variable_name = config.target_variable_name
        self.__predictor_variable_num = config.predictor_variable_num
        self.__simulations = config.simulations
        self.__active_library = config.active_library
        self.__start_year = config.start_year
        self.__end_year = config.end_year
        self.__true_randomness = True
        self.__seed = None

        # Time
        self.__start_time = time.perf_counter()
        self.__end_time = None
        self.__completion_percentage = 0
        self.__process_duration = None

        # Datasets
        self.__dataset_past = None
        self.__dataset_current = None
        self.__pre_feature_selection_dataset_current = None
        self.__pre_feature_selection_dataset_current = None

        # Dataset Statistics
        self.__variance_past = None
        self.__variance_current = None
        self.__correlation_past = None
        self.__correlation_current = None

        # Models
        self.__lr = None

        # Results
        self.__mean_accuracy_past = None
        self.__mean_accuracy_current = None
        self.__rmse = None
        self.__r2 = None

        # DataFrames
        self.__X_test_current = None
        self.__y_test_current = None

        # Data Containers
        self.__predictor_variable_names = []
        self.__y_predictions = []
        self.__X_trains = []
        self.__X_tests = []
        self.__y_trains = []
        self.__y_tests = []
        self.__X_tests_current = []
        self.__y_tests_current = []
        self.__lr_fits = []
        self.__lr_accuracies_past = []
        self.__lr_accuracies_current = []

    def add_tests(self, param1, param2):
        self.X_tests_current.append(param1)
        self.y_tests_current.append(param2)

    def print_results(self):
        pf.print_results(self)

    # Required Qualification (getters, setters, deleters)
    @property
    def required_qualification(self):
        return self.__required_qualification

    @required_qualification.setter
    def required_qualification(self, param):
        self.__required_qualification = param

    @required_qualification.deleter
    def required_qualification(self):
        return self.__required_qualification

    # Observation Unique ID (getters, setters, deleters)
    @property
    def observation_unique_ID(self):
        return self.__observation_unique_ID

    @observation_unique_ID.setter
    def observation_unique_ID(self, param):
        self.__observation_unique_ID = param

    @observation_unique_ID.deleter
    def observation_unique_ID(self):
        return self.__observation_unique_ID

    # Target Variable Name (getters, setters, deleters)
    @property
    def target_variable_name(self):
        return self.__target_variable_name

    @target_variable_name.setter
    def target_variable_name(self, param):
        self.__target_variable_name = param

    @target_variable_name.deleter
    def target_variable_name(self):
        return self.__target_variable_name

    # Predicator Variable Num (getters, setters, deleters)
    @property
    def predicator_variable_num(self):
        return self.__predicator_variable_num

    @predicator_variable_num.setter
    def predicator_variable_num(self, param):
        self.__predicator_variable_num = param

    @predicator_variable_num.deleter
    def predicator_variable_num(self):
        return self.__predicator_variable_num

    # Simulations (getters, setters, deleters)
    @property
    def simulations(self):
        return self.__simulations

    @simulations.setter
    def simulations(self, param):
        self.__simulations = param

    @simulations.deleter
    def simulations(self):
        return self.__simulations

    # Active Library (getters, setters, deleters)
    @property
    def active_library(self):
        return self.__active_library

    @active_library.setter
    def active_library(self, param):
        self.__active_library = param

    @active_library.deleter
    def active_library(self):
        return self.__active_library

    # Start Year (getters, setters, deleters)
    @property
    def start_year(self):
        return self.__start_year

    @start_year.setter
    def start_year(self, param):
        self.__start_year = param

    @start_year.deleter
    def start_year(self):
        return self.__start_year

    # End Year (getters, setters, deleters)
    @property
    def end_year(self):
        return self.__end_year

    @end_year.setter
    def end_year(self, param):
        self.__end_year = param

    @end_year.deleter
    def end_year(self,):
        return self.__end_year

    # True Randomness (getters, setters, deleters)
    @property
    def true_randomness(self):
        return self.__true_randomness

    @true_randomness.setter
    def true_randomness(self, param):
        self.__true_randomness = param

    @true_randomness.deleter
    def true_randomness(self,):
        return self.__true_randomness

    # Seed (getters, setters, deleters)
    @property
    def seed(self):
        return self.__seed

    @seed.setter
    def seed(self, param):
        self.__seed = param

    @seed.deleter
    def seed(self,):
        return self.__seed

    # End Time (getters, setters, deleters)

    @property
    def end_time(self):
        return self.__end_time

    @end_time.setter
    def end_time(self, param):
        self.__end_time = param
        self.__process_duration = (self.__end_time - self.__start_time)

    @end_time.deleter
    def end_time(self):
        del self.__end_time

    # Completion Percentage (getters, setters, deleters)

    @property
    def completion_percentage(self):
        return self.__completion_percentage

    @completion_percentage.setter
    def completion_percentage(self, param):
        self.__completion_percentage = param

    @completion_percentage.deleter
    def completion_percentage(self):
        del self.__completion_percentage

    # Process Duration (getters, setters, deleters)

    @property
    def process_duration(self):
        return self.__process_duration

    @process_duration.setter
    def process_duration(self, param):
        self.__process_duration = param

    @process_duration.deleter
    def process_duration(self):
        del self.__process_duration

    # Dataset Past (getters, setters, deleters)

    @property
    def dataset_past(self):
        return self.__dataset_past

    @dataset_past.setter
    def dataset_past(self, param):
        self.__dataset_past = param

    @dataset_past.deleter
    def dataset_past(self):
        del self.__dataset_past

    # Dataset Current (getters, setters, deleters)

    @property
    def dataset_current(self):
        return self.__dataset_current

    @dataset_current.setter
    def dataset_current(self, param):
        self.__dataset_current = param

    @dataset_current.deleter
    def dataset_current(self):
        del self.__dataset_current

    # Pre Feature Selection Dataset Past (getters, setters, deleters)

    @property
    def pre_feature_selection_dataset_past(self):
        return self.__pre_feature_selection_dataset_past

    @pre_feature_selection_dataset_past.setter
    def pre_feature_selection_dataset_past(self, param):
        self.__pre_feature_selection_dataset_past = param

    @pre_feature_selection_dataset_past.deleter
    def pre_feature_selection_dataset_past(self):
        del self.__pre_feature_selection_dataset_past

    # Pre Feature Selection Dataset Current (getters, setters, deleters)

    @property
    def pre_feature_selection_dataset_current(self):
        return self.__pre_feature_selection_dataset_current

    @pre_feature_selection_dataset_current.setter
    def pre_feature_selection_dataset_current(self, param):
        self.__pre_feature_selection_dataset_current = param

    @pre_feature_selection_dataset_current.deleter
    def pre_feature_selection_dataset_current(self):
        del self.__pre_feature_selection_dataset_current

    # Variance Past (getters, setters, deleters)

    @property
    def variance_past(self):
        return self.__variance_past

    @variance_past.setter
    def variance_past(self, param):
        self.__variance_past = param

    @variance_past.deleter
    def variance_past(self):
        del self.__variance_past

    # Variance Current Current (getters, setters, deleters)

    @property
    def variance_current(self):
        return self.__variance_current

    @variance_current.setter
    def variance_current(self, param):
        self.__variance_current = param

    @variance_current.deleter
    def variance_current(self):
        del self.__variance_current

    # Correlation Past (getters, setters, deleters)

    @property
    def correlation_past(self):
        return self.__correlation_past

    @correlation_past.setter
    def correlation_past(self, param):
        self.__correlation_past = param

    @correlation_past.deleter
    def correlation_past(self):
        del self.__correlation_past

    # Correlation Current (getters, setters, deleters)

    @property
    def correlation_current(self):
        return self.__correlation_current

    @correlation_current.setter
    def correlation_current(self, param):
        self.__correlation_current = param

    @correlation_current.deleter
    def correlation_current(self):
        del self.__correlation_current

    # Logistical Regression Model (getters, setters, deleters)

    @property
    def lr(self):
        return self.__lr

    @lr.setter
    def lr(self, param):
        self.__lr = param

    @lr.deleter
    def lr(self):
        del self.__lr

    # Mean Accuracy Past (getters, setters, deleters)

    @property
    def mean_accuracy_past(self):
        return self.__mean_accuracy_past

    @mean_accuracy_past.setter
    def mean_accuracy_past(self, param):
        self.__mean_accuracy_past = param

    @mean_accuracy_past.deleter
    def mean_accuracy_past(self):
        del self.__mean_accuracy_past

    # Mean Accuracy Current (getters, setters, deleters)

    @property
    def mean_accuracy_current(self):
        return self.__mean_accuracy_current

    @mean_accuracy_current.setter
    def mean_accuracy_current(self, param):
        self.__mean_accuracy_current = param

    @mean_accuracy_current.deleter
    def mean_accuracy_current(self):
        del self.__mean_accuracy_current

    # Root Mean Square Equation (getters, setters, deleters)

    @property
    def rmse(self):
        return self.__rmse

    @rmse.setter
    def rmse(self, param):
        self.__rmse = param

    @rmse.deleter
    def rmse(self):
        del self.__rmse

    # rSquared Value (getters, setters, deleters)

    @property
    def r2(self):
        return self.__r2

    @r2.setter
    def r2(self, param):
        self.__r2 = param

    @r2.deleter
    def r2(self):
        del self.__r2

    # X_test_current (getters, setters, deleters)

    @property
    def X_test_current(self):
        return self.__X_test_current

    @X_test_current.setter
    def X_test_current(self, param):
        self.__X_test_current = param

    @X_test_current.deleter
    def X_test_current(self):
        del self.__X_test_current

    # y_test_current (getters, setters, deleters)

    @property
    def y_test_current(self):
        return self.__y_test_current

    @y_test_current.setter
    def y_test_current(self, param):
        self.__y_test_current = param

    @y_test_current.deleter
    def y_test_current(self):
        del self.__y_test_current

    # Predictor Variable Names (getters, setters, deleters)

    @property
    def predictor_variable_names(self):
        return self.__predictor_variable_names

    @predictor_variable_names.setter
    def predictor_variable_names(self, param):
        self.__predictor_variable_names = param

    @predictor_variable_names.deleter
    def predictor_variable_names(self):
        del self.__predictor_variable_names

    # y Predictions (getters, setters, deleters)

    @property
    def y_predictions(self):
        return self.__y_predictions

    @y_predictions.setter
    def y_predictions(self, param):
        self.__y_predictions = param

    @y_predictions.deleter
    def y_predictions(self):
        del self.__y_predictions

    # X_trains (getters, setters, deleters)

    @property
    def X_trains(self):
        return self.__X_trains

    @X_trains.setter
    def X_trains(self, param):
        self.__X_trains = param

    @X_trains.deleter
    def X_trains(self):
        del self.__X_trains

    # X_tests (getters, setters, deleters)

    @property
    def X_tests(self):
        return self.__X_tests

    @X_tests.setter
    def X_tests(self, param):
        self.__X_tests = param

    @X_tests.deleter
    def X_tests(self):
        del self.__X_tests

    # y_trains (getters, setters, deleters)

    @property
    def y_trains(self):
        return self.__y_trains

    @y_trains.setter
    def y_trains(self, param):
        self.__y_trains = param

    @y_trains.deleter
    def y_trains(self):
        del self.__y_trains

    # y_tests (getters, setters, deleters)

    @property
    def y_tests(self):
        return self.__y_tests

    @y_tests.setter
    def y_tests(self, param):
        self.__y_tests = param

    @y_tests.deleter
    def y_tests(self):
        del self.__y_tests

    # y_tests (getters, setters, deleters)

    @property
    def y_tests(self):
        return self.__y_tests

    @y_tests.setter
    def y_tests(self, param):
        self.__y_tests = param

    @y_tests.deleter
    def y_tests(self):
        del self.__y_tests

    # X_tests_current (getters, setters, deleters, add)

    @property
    def X_tests_current(self):
        return self.__X_tests_current

    @X_tests_current.setter
    def X_tests_current(self, param):
        self.__X_tests_current = param

    @X_tests_current.deleter
    def X_tests_current(self):
        del self.__X_tests_current

    # y_tests_current (getters, setters, deleters)

    @property
    def y_tests_current(self):
        return self.__y_tests_current

    @y_tests_current.setter
    def y_tests_current(self, param):
        self.__y_tests_current = param

    @y_tests_current.deleter
    def y_tests_current(self):
        del self.__y_tests_current

    # lr_fits (getters, setters, deleters)

    @property
    def lr_fits(self):
        return self.__lr_fits

    @lr_fits.setter
    def lr_fits(self, param):
        self.__lr_fits = param

    @lr_fits.deleter
    def lr_fits(self):
        del self.__lr_fits

    # lr_accuracies_past (getters, setters, deleters)

    @property
    def lr_accuracies_past(self):
        return self.__lr_accuracies_past

    @lr_accuracies_past.setter
    def lr_accuracies_past(self, param):
        self.__lr_accuracies_past = param

    @lr_accuracies_past.deleter
    def lr_accuracies_past(self):
        del self.__lr_accuracies_past

    # lr_accuracies_current (getters, setters, deleters)

    @property
    def lr_accuracies_current(self):
        return self.__lr_accuracies_current

    @lr_accuracies_current.setter
    def lr_accuracies_current(self, param):
        self.__lr_accuracies_current = param

    @lr_accuracies_current.deleter
    def lr_accuracies_current(self):
        del self.__lr_accuracies_current
