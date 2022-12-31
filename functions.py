# -*- coding: utf-8 -*-
"""
@author: JosephRe
"""

# Data Science Imports
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

# Other External Imports
import time

# Internal Imports
import config

import process as p


def load_data(process):
    set_initial_configurations(process)
    create_datasets(process)
    print(f'Required qual = {process.required_qualification}\nObservation_unique is = {process.observation_unique_ID}\nprocess target variable = {process.target_variable_name}\npredictor variable num = {process.predictor_variable_num}\nsimulations = {process.simulations}\nend year = {process.end_year}')


def set_initial_configurations(process):
    process.required_qualification = config.required_qualification
    process.observation_unique_ID = config.observation_unique_ID
    process.target_variable_name = config.target_variable_name
    process.predictor_variable_num = config.predictor_variable_num
    process.simulations = config.simulations
    process.active_library = config.active_library
    process.start_year = config.start_year
    process.end_year = config.end_year
    select_library(process.active_library)
    if not process.true_randomness:
        process.seed = 0


def create_datasets(process):
    # Instantiate datasets as all batter statlines from 2015-2021 (past) and 2022 (current)
    process.dataset_past = create_dataset(process, 'past')
    process.dataset_current = create_dataset(process, 'current')


def create_dataset(process, timeline) -> pd.DataFrame:
    if process.active_library in config.library_mappings:
        if timeline in ['past']:
            print('Returning past dataset')
            return select_library(process.active_library)(process.start_year, process.end_year, qual = process.required_qualification)
        elif timeline in ['current']:
            print('Returning current dataset')
            return select_library(process.active_library)(process.end_year, process.end_year + 1, qual = process.required_qualification)


def select_library(library):
    while True:
        try:
            if config.operations < 200:
                for x in range(100):
                    increment_operation()
            return config.library_mappings[library]
        except KeyError:
            print('Invalid library. Try again.')


def increment_operation():
    config.operations += 1
    config.completion_percentage = int((config.operations / config.total_operations) * 100)


def merge_datasets(dataset1, dataset2, uniqueID) -> pd.DataFrame:
    '''
    Merges 2 datasets by removing the symmetric difference of the sets.

    Parameters
    ----------
    dataset1: Any pd.DataFrame

    dataset2: Any pd.DataFrame

    uniqueID: String name of a column present in both dataframes that will be
    used for defining what subset of dataset1 only contains observations with
    values in dataset2.

    Returns
    -------
    Returns a pd.DataFrame subset of dataset1 that only contains overlapping
    elements from dataset2 per the uniqID.
    '''
    def same_unique_values_in_column(dataset1, dataset2, columnName) -> bool:
        return len([*set(getattr(dataset1, columnName))]) == len([*set(getattr(dataset2, columnName))])
    
    # Removes all observations in dataset1 that do not share a value in the provided column name with at least one observation in dataset2
    if not same_unique_values_in_column(dataset1, dataset2, uniqueID):
        return dataset1[getattr(dataset1, uniqueID).isin(getattr(dataset2, uniqueID))]


def removeNonNumericalFeatures(dataset) -> pd.DataFrame:
    # Removes all features in a dataset where the column contains any instance of a non-numerical value
    return dataset.select_dtypes(include = [np.number])


def removeFeaturesWithNan(dataset) -> pd.DataFrame:
    # Removes all features in a dataset where at least one observation at a NaN value
    return dataset.dropna(axis = 1)


def sanitize_dataset(dataset) -> pd.DataFrame:
    '''
    Performs basic numerical sanitization of a dataset (i.e., removes non-numerical and Nan values).

    Parameters
    ----------
    dataset: Any pd.DataFrame

    Returns
    -------
    Returns a pd.DataFrame that contains neither Nan or non-numerical values
    '''
    dataset = removeNonNumericalFeatures(dataset)
    return removeFeaturesWithNan(dataset)


def setIndependentAndDependentVariable(dataset, y, X) -> pd.DataFrame:
    if isinstance(X, str):
        intersectionString = y + " " + X
    else:
        intersectionString = y + " " + " ".join(X)
    return dataset[dataset.columns.intersection(intersectionString.split(" "))]


def createIndexListFromDataFrame(dataframe, columnName):
    res = dataframe.index.tolist()
    # del res[res.index(columnName)]
    return res


def createColumnListFromDataFrame(dataframe, indicesList, columnName):
    index_to_remove = indicesList.index(columnName)
    res = dataframe.loc[dataframe.columns[index_to_remove]].tolist()
    del res[index_to_remove]
    return res


def removeIndependentVariableFromList(list1, columnName):
    del list1[list1.index(columnName)]
    return list1


def combineListsIntoDictionary(list1, list2):
    return { list1[i] : list2[i] for i in range(len(list1)) }


def combineListsIntoDataFrame(list1, list2) -> pd.DataFrame:
    pd.DataFrame(data = { list1[i] : list2[i] for i in range(len(list1)) })
    return pd.DataFrame()


def reverseDictionaryKeyValuePair(dictionary):
    return {v: k for k, v in dictionary.items()} 


def select_predictor_variables(process):
    '''
    Selects predictor variables based on the independent variables that have the highest correlation to the dependent variable.

    Parameters
    ----------
    process: Any process object

    Returns
    -------
    No return value. The function simple updates the value of the chosen predictor variables based on the highest correlations.
    '''
    feature_list = createIndexListFromDataFrame(process.correlation_past, process.target_variable_name)
    value_list = createColumnListFromDataFrame(process.correlation_past, feature_list, process.target_variable_name)
    correlations_sorted = zip(value_list, feature_list)
    correlations_sorted = sorted(correlations_sorted, reverse = True)
    
    # Select predictor variables based on highest correlation values
    for i in range(process.predictor_variable_num):
        process.predictor_variable_names.append(correlations_sorted[i][1])


def set_features_and_target(process):
    # Set features and target
    process.dataset_past = setIndependentAndDependentVariable(process.dataset_past, process.target_variable_name, process.predictor_variable_names)
    process.dataset_current = setIndependentAndDependentVariable(process.dataset_current, process.target_variable_name, process.predictor_variable_names)


def create_past_dataframes(process):
    # Past: Split the dataset into training and testing dataframes
    for i in range(process.simulations):
        X_train, X_test, y_train, y_test = train_test_split(process.dataset_past.loc[:, process.dataset_past.columns != process.target_variable_name], process.dataset_past.loc[:, process.dataset_past.columns == process.target_variable_name], random_state = process.seed)
        process.X_trains.append(X_train)
        process.X_tests.append(X_test)
        process.y_trains.append(y_train)
        process.y_tests.append(y_test)


def create_current_testing_dataframes(process):
    # Current: Split the dataset into testing dataframes
    for i in range(process.simulations):
        process.X_test_current = process.dataset_current.loc[:, process.dataset_current.columns != process.target_variable_name]
        process.y_test_current = process.dataset_current.loc[:, process.dataset_current.columns == process.target_variable_name]
        process.X_tests_current.append(process.X_test_current)
        process.y_tests_current.append(process.y_test_current)


def create_linear_regression(process):
    # Create and fit a linear regression model using the training data
    process.lr = LinearRegression()
    for i in range(process.simulations):
        process.lr.fit(process.X_trains[i].values, process.y_trains[i])
        process.lr_fits.append(process.lr)


def make_predictions(process):
    # Create simulations number of predictions for current
    for i in range(process.simulations):
        y_pred = process.lr.predict(process.X_tests_current[i].values)
        process.y_predictions.append(y_pred)


def calculate_r2(process):
    # Calculate accuracy of model using testing data
    for i in range(process.simulations):
        lr_accuracy_past = float(process.lr.score(process.X_tests[i].values, process.y_tests[i]) * 100)
        lr_accuracy_current = float(process.lr.score(process.X_tests_current[i].values, process.y_tests_current[i]) * 100)
        process.lr_accuracies_past.append(lr_accuracy_past)
        process.lr_accuracies_current.append(lr_accuracy_current)


def calculate_mean_accuracy(model_accuracies):
    model_accuracy_acc = 0
    for model_accuracy in model_accuracies:
        model_accuracy_acc += float(model_accuracy)
    return (model_accuracy_acc / float(len(model_accuracies)))


def calculate_advanced_statistics(process):
    # Advanced statistics
    process.rmse = mean_squared_error(process.y_test_current, process.y_predictions[0], squared = False)
    process.r2 = r2_score(process.y_test_current, process.y_predictions[0])


def set_mean_accuracy(process):
    process.mean_accuracy_past = calculate_mean_accuracy(process.lr_accuracies_past)
    process.mean_accuracy_current = calculate_mean_accuracy(process.lr_accuracies_current)


def calculate_variance(process):
    # Calculate feature variance and correlations to determine if any need to be removed for having a 0 value SHOULD BE MOVED TO INSIDE PROCESS.PY
    process.variance_past = process.dataset_past.var()
    process.variance_current = process.dataset_current.var()


def calculate_correlation(process):
    process.correlation_past = process.dataset_past.corr()
    process.correlation_current = process.dataset_current.corr()


def sanitize_datasets(process):
    # Perform a basic sanitization of the dataset
    process.dataset_past = sanitize_dataset(process.dataset_past)
    process.pre_feature_selection_dataset_current = process.dataset_current = sanitize_dataset(process.dataset_current)
    
    # Ensure datasets only contains overlapping players
    process.pre_feature_selection_dataset_past = process.dataset_past = merge_datasets(process.dataset_past, process.dataset_current, process.observation_unique_ID)


def complete_process(process):
    process.end_time = time.perf_counter()
    process.print_results()


def understand_data(process):
    calculate_variance(process)
    calculate_correlation(process)
    select_predictor_variables(process)
    set_features_and_target(process)
    create_past_dataframes(process)
    create_current_testing_dataframes(process)


def begin_process(process):
    sanitize_datasets(process)
    increment_operation()
    understand_data(process)
    increment_operation()
    create_linear_regression(process)
    increment_operation()
    make_predictions(process)
    increment_operation()
    evaluate_results(process)
    increment_operation()
    print(f'Required qual = {process.required_qualification}\nObservation_unique is = {process.observation_unique_ID}\nprocess target variable = {process.target_variable_name}\npredictor variable num = {process.predictor_variable_num}\nsimulations = {process.simulations}\nend year = {process.end_year}')
    complete_process(process)


def evaluate_results(process):
    # Evaluate results
    calculate_r2(process)
    set_mean_accuracy(process)
    calculate_advanced_statistics(process)
