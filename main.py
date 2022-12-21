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
import os.path
import threading

# User Interface Imports
import customtkinter
from tkinter import ttk
from pandastable import Table, TableModel

# Internal Imports
import config
import process as p


def increment_operation():
    config.operations += 1
    config.completion_percentage = int((config.operations / config.total_operations) * 100)


def select_library(library):
    while True:
        print('Selecting library')
        try:
            print('Trying to return selected library')
            if config.operations < 200:
                for x in range(100):
                    increment_operation()
            return config.library_mappings[library]
        except KeyError:
            print('Invalid library. Try again.')


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
    dataset = removeNonNumericalFeatures(dataset)
    return removeFeaturesWithNan(dataset)


def setIndependentAndDependentVariable(dataset, y, X) -> pd.DataFrame:
    if isinstance(X, str):
        intersectionString = y + " " + X
    else:
        intersectionString = y + " " + " ".join(X)
    return dataset[dataset.columns.intersection(intersectionString.split(" "))]


def createIndexListFromDataFrame(dataframe, columnName) -> []:
    res = dataframe.index.tolist()
    # del res[res.index(columnName)]
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


def select_predictor_variables(process) -> []:
    feature_list = createIndexListFromDataFrame(process.correlation_past, process.target_variable_name)
    value_list = createColumnListFromDataFrame(process.correlation_past, feature_list, process.target_variable_name)
    correlations_sorted = zip(value_list, feature_list)
    correlations_sorted = sorted(correlations_sorted, reverse = True)
    
    # Select predicator variables based on highest correlation values
    for i in range(process.predicator_variable_num):
        process.predictor_variable_names.append(correlations_sorted[i][1])


def create_dataset(process, timeline) -> pd.DataFrame:
    if process.active_library in config.library_mappings:
        print('Checked library')
        
        if timeline in ['past']:
            print('Returning past')
            return select_library(process.active_library)(process.start_year, process.end_year, qual = process.required_qualification)
        elif timeline in ['current']:
            print('Returning current')
            return select_library(process.active_library)(process.end_year, process.end_year + 1, qual = process.required_qualification)


def set_initial_configurations(process):
    print('Entered set_initial_configurations')
    process.required_qualification = config.required_qualification
    process.observation_unique_ID = config.observation_unique_ID
    process.target_variable_name = config.target_variable_name
    process.predicator_variable_num = config.predicator_variable_num
    process.simulations = config.simulations
    process.active_library = config.active_library
    process.start_year = config.start_year
    process.end_year = config.end_year
    select_library(process.active_library)
    if not process.true_randomness:
        process.seed = 0


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


def calculate_variance(dataset):
    # Calculate feature variance and correlations to determine if any need to be removed for having a 0 value SHOULD BE MOVED TO INSIDE PROCESS.PY
    dataset.variance_past = dataset.dataset_past.var()
    dataset.variance_current = dataset.dataset_current.var()

    
def calculate_correlation(dataset):
    dataset.correlation_past = dataset.dataset_past.corr()
    dataset.correlation_current = dataset.dataset_current.corr()


def create_datasets(process):
    print('Entered create_datasets')
    # Instantiate datasets as all batter statlines from 2015-2021 (past) and 2022 (current)
    process.dataset_past = create_dataset(process, 'past')
    print('Created past')
    process.dataset_current = create_dataset(process, 'current')
    print('Created current')


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
    
    
    
def begin_process():
    print('Entered begin_process')
    p1 = p.Process()
    set_initial_configurations(p1)
    create_datasets(p1)
    process_request(p1)


def evaluate_results(process):
    print('Entered evaluate_results')
    # Evaluate results
    calculate_r2(process)
    set_mean_accuracy(process)
    calculate_advanced_statistics(process)


def process_request(process):
    print('Entered process_request')
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
    
    complete_process(process)
    

customtkinter.set_appearance_mode('dark')
customtkinter.set_default_color_theme('dark-blue')

root = customtkinter.CTk()
root.minsize(640, 480)
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry('1920x1080x0x0')
#root.geometry("%dx%d-8-1" % (w, h))
#root.state('zoomed')
root.iconbitmap(os.path.dirname(os.path.abspath(__file__)) + '\\images\\favicon.ico')
root.title('Data Science Suite')
root.rowconfigure((0,1,2), weight=1)
root.columnconfigure((0,1,2,3,4,5), weight=1)
'''

X
X
X
X

'''


frame = customtkinter.CTkFrame(master=root)
frame.grid(row=0, column=1, columnspan=3, sticky='nsew')
frame.grid_columnconfigure((1,2,3), weight=1)
frame.grid_rowconfigure((0, 1, 2), weight=0)

label1 = customtkinter.CTkLabel(master=frame, text='Data Science Suite', font=('Roboto', 78))
label1.grid(row=3, column=0, columnspan=3, pady=0, padx=10)

label2 = customtkinter.CTkLabel(master=frame, text='A Machine Learning Tool\nby Joseph Re', font=('Roboto', 24))
label2.grid(row=4, column=0, columnspan=3, pady=12, padx=10)



def process_request_thread():
    def set_settings():
        config.required_qualification = int(entry1.get())
        config.observation_unique_ID = entry2.get()
        config.target_variable_name = entry3.get()
        config.predicator_variable_num = int(entry4.get())
        config.simulations = int(entry5.get())
        config.active_library = int(entry6.get())
        config.start_year = int(entry7.get())
        config.end_year = int(entry8.get())
    set_settings()
    global process_request_thread
    process_request_thread = threading.Thread(target=begin_process)
    process_request_thread.daemon = True
    process_request_thread.start()
    root.after(20, check_process_request_thread)


def check_process_request_thread():
    if config.completion_percentage >= 100:
        pb['value'] = config.completion_percentage
    if pb['value'] < config.completion_percentage:
        pb['value'] += .275
    else:
        pb['value'] = config.completion_percentage
    root.update_idletasks()
    #global value_label
    value_label['text'] = update_progress_label()
    if process_request_thread.is_alive():
        root.after(20, check_process_request_thread)
    else:
        if config.completion_percentage >= 100:
            pb['value'] = config.completion_percentage
        

def update_progress_label():
    return f"Current Progress: {int(pb['value'])}%"


entry1 = customtkinter.CTkEntry(master=frame, placeholder_text='Required Qualification')
entry1.grid(row=5, column=0, pady=12, padx=10)
entry1.insert(0, '550')

entry2 = customtkinter.CTkEntry(master=frame, placeholder_text='Observation Unique ID')
entry2.grid(row=5, column=1, pady=12, padx=10)
entry2.insert(0, 'IDfg')

entry3 = customtkinter.CTkEntry(master=frame, placeholder_text='Target Variable Name')
entry3.grid(row=5, column=2, pady=12, padx=10)
entry3.insert(0, 'HR')

entry4 = customtkinter.CTkEntry(master=frame, placeholder_text='Predicator Variable Num')
entry4.grid(row=6, column=0, pady=12, padx=10)
entry4.insert(0, '5')

entry5 = customtkinter.CTkEntry(master=frame, placeholder_text='Simulations')
entry5.grid(row=6, column=1, pady=12, padx=10)
entry5.insert(0, '1')

entry6 = customtkinter.CTkEntry(master=frame, placeholder_text='Active Library')
entry6.grid(row=6, column=2, pady=12, padx=10)
entry6.insert(0, '1')

entry7 = customtkinter.CTkEntry(master=frame, placeholder_text='Start Year')
entry7.grid(row=7, column=0, pady=12, padx=10)
entry7.insert(0, '2015')

entry8 = customtkinter.CTkEntry(master=frame, placeholder_text='End Year')
entry8.grid(row=7, column=1, pady=12, padx=10)
entry8.insert(0, '2022')

button = customtkinter.CTkButton(master=frame, text='Load Data', command=process_request_thread)
button.grid(row=7, column=1, pady=12, padx=10, sticky='nsew')

button = customtkinter.CTkButton(master=frame, text='Start Processing', command=process_request_thread)
button.grid(row=7, column=2, pady=12, padx=10, sticky='nsew')

# progressbar
pb = ttk.Progressbar(master=frame, orient='horizontal', mode='determinate', length=280)

# place the progressbar
pb.grid(row=8, column=1, columnspan=2, padx=10, pady=20)

# label
value_label = ttk.Label(master=frame, text=update_progress_label())
value_label.grid(row=8, column=0, columnspan=2, pady=0, padx=0)

# table model
frame.grid(row=1, column=0, rowspan=8, pady=0, padx=0)
df = config.batting_stats(2022, 2022, qual = 550)
#TableModel.getSampleData()
table = Table(frame, dataframe=df, showtoolbar=False, showstatusbar=False)
table.show()

root.mainloop()
