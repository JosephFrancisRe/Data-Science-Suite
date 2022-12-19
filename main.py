# -*- coding: utf-8 -*-
"""
@author: JosephRe
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import time
import config
import results as r

def select_library(library):
    while True:
        try:
            return config.library_mappings[library]
        except KeyError:
            print('Invalid function, try again.')

def sameUniqueValuesInColumn(dataset1, dataset2, columnName) -> bool:
    return len([*set(getattr(dataset1, columnName))]) == len([*set(getattr(dataset2, columnName))])

def removeSymmetricDifference(dataset1, dataset2, columnName) -> pd.DataFrame:
    # Removes all oberservations in dataset1 that do not share a value in the provided column name with at least one observation in dataset2
    if not sameUniqueValuesInColumn(dataset1, dataset2, columnName):
        return dataset1[getattr(dataset1, columnName).isin(getattr(dataset2, columnName))]

def removeNonNumericalFeatures(dataset) -> pd.DataFrame:
    # Removes all features in a dataset where the column contains any instance of a non-numerical value
    return dataset.select_dtypes(include = [np.number])

def removeFeaturesWithNan(dataset) -> pd.DataFrame:
    # Removes all features in a dataset where at least one observation at a NaN value
    return dataset.dropna(axis = 1)

def sanitizeDataset(dataset) -> pd.DataFrame:
    dataset = removeNonNumericalFeatures(dataset)
    return removeFeaturesWithNan(dataset)

def setIndependentAndDependentVariable(dataset, y, X) -> pd.DataFrame:
    if type(X) == type("string"):
        intersectionString = y + " " + X
    else:
        intersectionString = y + " " + " ".join(X)
    return dataset[dataset.columns.intersection(intersectionString.split(" "))]

def createIndexListFromDataFrame(dataframe, columnName) -> []:
    res = dataframe.index.tolist()
    #del res[res.index(columnName)]
    return res

def createColumnListFromDataFrame(dataframe, indicesList, columnName) -> []:
    index_to_remove = indicesList.index(columnName)
    res = dataframe.loc[dataframe.columns[index_to_remove]].tolist()
    del res[index_to_remove]
    return res

def removeIndependentVariableFromList(list1, columnName) -> []:
    del list1[list1.index(columnName)]
    return list1

def combineListsIntoDictionary(list1, list2) -> {}:
    return { list1[i] : list2[i] for i in range(len(list1)) }

def combineListsIntoDataFrame(list1, list2) -> pd.DataFrame:
    pd.DataFrame(data = { list1[i] : list2[i] for i in range(len(list1)) })
    return pd.DataFrame()

def reverseDictionaryKeyValuePair(dictionary) -> []:
    return {v: k for k, v in dictionary.items()} 

def create_dataset(library, timeline) -> pd.DataFrame:
    if library in config.library_mappings:
        if timeline in ['past']:
            return select_library(library)(config.start_year, config.end_year, qual = config.required_qualification)
        elif timeline in ['current']:
            return select_library(library)(config.end_year, config.end_year + 1, qual = config.required_qualification)

def set_initial_configurations():
    select_library(config.active_library)
    if not config.trueRandomness:
        config.seed = 0
    config.start_time = time.time()

set_initial_configurations()

# Instantiate datasets as all batter statlines from 2015-2021 and current in which the batter had at least 400 plate appearance
config.dataset_past = create_dataset(config.active_library, 'past')
config.dataset_current = create_dataset(config.active_library, 'current')

# Perform a basic sanitization of the dataset
config.dataset_past = sanitizeDataset(config.dataset_past)
pre_feature_selection_dataset_current = config.dataset_current = sanitizeDataset(config.dataset_current)

# Ensure both datasets only contains overlapping players
pre_feature_selection_dataset = config.dataset_past = removeSymmetricDifference(config.dataset_past, config.dataset_current, config.uniqueIDColumnName)

# Calculate feature variance and correlations to determine if any need to be removed for having a 0 value
variance = config.dataset_past.var()
variance_current = config.dataset_current.var()
correlation = config.dataset_past.corr()
correlation_current = config.dataset_current.corr()

# Create correlation lists to eventually sort
correlation_index_list = createIndexListFromDataFrame(correlation, config.target_variable_name)
correlation_value_list = createColumnListFromDataFrame(correlation, correlation_index_list, config.target_variable_name)


# Sort correlation lists 
correlations_sorted = zip(correlation_value_list, correlation_index_list)
correlations_sorted = sorted(correlations_sorted, reverse = True)

# Select predicator variables based on highest correlation values
for i in range(config.predicator_variable_num):
    config.predictor_variable_names.append(correlations_sorted[i][1])

def set_features_and_target():
    # Set features and target
    config.dataset_past = setIndependentAndDependentVariable(config.dataset_past, config.target_variable_name, config.predictor_variable_names)
    config.dataset_current = setIndependentAndDependentVariable(config.dataset_current, config.target_variable_name, config.predictor_variable_names)

def create_past_dataframes():
    # Past: Split the dataset into training and testing dataframes
    for i in range(config.simulations):
        X_train, X_test, y_train, y_test = train_test_split(config.dataset_past.loc[:, config.dataset_past.columns != config.target_variable_name], config.dataset_past.loc[:, config.dataset_past.columns == config.target_variable_name], random_state = config.seed)
        config.X_trains.append(X_train)
        config.X_tests.append(X_test)
        config.y_trains.append(y_train)
        config.y_tests.append(y_test)

def create_current_testing_dataframes():
    # Current: Split the dataset into testing dataframes
    for i in range(config.simulations):
        config.X_test_current = config.dataset_current.loc[:, config.dataset_current.columns != config.target_variable_name]
        config.y_test_current = config.dataset_current.loc[:, config.dataset_current.columns == config.target_variable_name]
        config.X_tests_current.append(config.X_test_current)
        config.y_tests_current.append(config.y_test_current)

def create_linear_regression():
    # Create and fit a linear regression model using the training data
    config.lr = LinearRegression()
    for i in range(config.simulations):
        config.lr.fit(config.X_trains[i].values, config.y_trains[i])
        config.lr_fits.append(config.lr)

set_features_and_target()
create_past_dataframes()
create_current_testing_dataframes()
create_linear_regression()

# Create simulations number of predictions for current
for i in range(config.simulations):
    y_pred = config.lr.predict(config.X_tests_current[i].values)
    config.y_predictions.append(y_pred)

# Calculate accuracy of model using testing data
for i in range(config.simulations):
    lr_accuracy_2015_to_2021 = float(config.lr.score(config.X_tests[i].values, config.y_tests[i]) * 100)
    lr_accuracy_current = float(config.lr.score(config.X_tests_current[i].values, config.y_tests_current[i]) * 100)
    config.lr_accuracies_past.append(lr_accuracy_2015_to_2021)
    config.lr_accuracies_current.append(lr_accuracy_current)

accuracy_past_acc = 0
for accuracy_rate in config.lr_accuracies_past:
    accuracy_past_acc += float(accuracy_rate)
mean_accuracy_past = accuracy_past_acc / float(len(config.lr_accuracies_past))

accuracy_current_acc = 0
for accuracy_rate in config.lr_accuracies_current:
    accuracy_current_acc += float(accuracy_rate)
mean_accuracy_current = accuracy_current_acc / float(len(config.lr_accuracies_current))

# Advanced statistics
rmse = mean_squared_error(config.y_test_current, y_pred, squared = False)
r2 = r2_score(config.y_test_current, y_pred)

config.end_time = time.time()
results = r.Results(mean_accuracy_past, mean_accuracy_current, config.predictor_variable_names, (float(config.end_time) - float(config.start_time)))
results.print_results()

# current player comparable -----------------------------------------------------------------------------------------------------
# User inputs a player name and year
#player_comparison = input('What player would you like to find a comparison for?:')
#year_selection = input('Select a year between 2015 and 2021 during which the player was active: ')

# Program fetches that player's predictive_variables
# Model predicts with those variables how many homeruns that would be in current
# Model locates the player/players closest to that homerun total in current
# Print result