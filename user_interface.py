# -*- coding: utf-8 -*-
"""
@author: JosephRe
"""

# Other External Imports
import os.path
import threading

# User Interface Imports
import customtkinter
from tkinter import ttk
from pandastable import Table

# Internal Imports
import config
import functions
import process

import pybaseball


class UserInterface:
    def __init__(self):
        # General Options
        self.appearance = customtkinter.set_appearance_mode('dark')
        self.color = customtkinter.set_default_color_theme('dark-blue')

        # Root and process
        self.root = self.create_root_frame()
        self.process = process.Process()

        # Load Module
        self.load_module = customtkinter.CTkFrame(master=self.root)
        self.load_module_title_label = customtkinter.CTkLabel(master=self.load_module, text='Data Science Suite', font=('Roboto', 78))
        self.load_module_subtitle_label = customtkinter.CTkLabel(master=self.load_module, text='A Machine Learning Tool\nby Joseph Re', font=('Roboto', 24))
        self.active_library_entry = customtkinter.CTkEntry(master=self.load_module, placeholder_text='Active Library')
        self.button1 = customtkinter.CTkButton(master=self.load_module, text='Load Data', command=self.create_load_thread)
        self.load_module_pb = ttk.Progressbar(master=self.load_module, orient='horizontal', mode='indeterminate', length=280)
        self.create_load_module()

        # Process Module
        self.process_module = customtkinter.CTkFrame(master=self.root)
        self.process_module_pb = ttk.Progressbar(master=self.process_module, orient='horizontal', mode='determinate', length=280)
        self.required_qualification_label = customtkinter.CTkEntry(master=self.process_module, placeholder_text='Required Qualification')
        self.observation_unique_ID_entry = customtkinter.CTkEntry(master=self.process_module, placeholder_text='Observation Unique ID')
        self.target_variable_name_entry = customtkinter.CTkEntry(master=self.process_module, placeholder_text='Target Variable Name')
        self.predictor_variable_num_entry = customtkinter.CTkEntry(master=self.process_module, placeholder_text='Predictor Variable Num')
        self.simulations_entry = customtkinter.CTkEntry(master=self.process_module, placeholder_text='Simulations')
        self.start_year_entry = customtkinter.CTkEntry(master=self.process_module, placeholder_text='Start Year')
        self.end_year_entry = customtkinter.CTkEntry(master=self.process_module, placeholder_text='End Year')
        self.processing_button = customtkinter.CTkButton(master=self.process_module, text='Start Processing', command=self.create_process_thread)
        self.process_module_pb_label = ttk.Label(master=self.process_module, text=self.update_progress_label)
        self.create_process_module()

        # Dataset Module
        self.table = None
        self.load_thread = None
        self.process_request_thread = None
        self.dataset_module = self.create_dataset_module()

        # Set Config Values
        config.ui = self


    def create_root_frame(self):
        # Root
        root = customtkinter.CTk()
        root.minsize(1580, 720)
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry('1920x1080x0x0')
        root.geometry("%dx%d-8-1" % (w, h))
        root.state('zoomed')
        root.iconbitmap(os.path.dirname(os.path.abspath(__file__)) + '\\images\\favicon.ico')
        root.title('Data Science Suite')
        root.rowconfigure((0,1), weight=1)
        root.columnconfigure(1, weight=1)
        return root


    def check_load_thread(self):
        if self.load_thread.is_alive():
            self.root.after(20, self.check_load_thread())
        else:
            self.load_module_pb.stop()
            print(self.process.dataset_past)
            print(self.process.dataset_current)
            self.update_dataset_module()


    def create_load_thread(self):
        # Create Load Thread
        self.load_thread = threading.Thread(target=functions.load_data(self.process))
        self.load_thread.daemon = True
        self.load_thread.start()
        self.load_module_pb.start()
        self.root.after(20, self.check_load_thread())


    def set_load_settings(self):
        print(f'process active library = {int(self.active_library_entry.get())}')
        self.process.active_library = config.active_library = int(self.active_library_entry.get())


    def create_load_module(self):
        # LoadModule
        self.load_module.grid(row=0, column=0, sticky='nsew')
        self.load_module.grid_columnconfigure(0, weight=0)
        self.load_module.grid_rowconfigure(0, weight=0)
        self.load_module_title_label.grid(row=0, column=0, columnspan=3, pady=12, padx=30)
        self.load_module_subtitle_label.grid(row=1, column=0, columnspan=3, pady=12, padx=10)
        self.active_library_entry.grid(row=6, column=2, pady=12, padx=10)
        self.active_library_entry.insert(0, '1')
        self.button1.grid(row=7, column=1, pady=12, padx=10, sticky='nsew')
        self.load_module_pb.grid(row=8, column=1, columnspan=3, padx=0, pady=0)
        self.set_load_settings()


    def check_process_request_thread(self):
        if config.completion_percentage >= 100:
            self.process_module_pb['value'] = config.completion_percentage
            if self.process_module_pb['value'] < config.completion_percentage:
                self.process_module_pb['value'] += .275
            else:
                self.process_module_pb['value'] = config.completion_percentage
            self.root.update_idletasks()
            self.process_module_pb_label['text'] = self.update_progress_label()
            if self.process_request_thread.is_alive():
                self.root.after(20, self.check_process_request_thread())


    def update_progress_label(self):
        return f"Current Progress: {int(self.process_module_pb['value'])}%"


    def create_process_thread(self):
        self.process_request_thread = threading.Thread(target=functions.begin_process(self.process))
        self.process_request_thread.daemon = True
        self.process_request_thread.start()
        self.root.after(20, self.check_process_request_thread())


    def set_process_settings(self):
        print(f'Required qual = {int(self.required_qualification_label.get())}\nObservation_unique is = {self.observation_unique_ID_entry.get()}\nprocess target variable = {self.target_variable_name_entry.get()}\npredictor variable num = {int(self.predictor_variable_num_entry.get())}\nsimulations = {int(self.simulations_entry.get())}\nend year = {int(self.end_year_entry.get())}')
        self.process.required_qualification = config.required_qualification = int(self.required_qualification_label.get())
        self.process.observation_unique_ID = config.observation_unique_ID = self.observation_unique_ID_entry.get()
        self.process.target_variable_name = config.target_variable_name = self.target_variable_name_entry.get()
        self.process.predictor_variable_num = config.predictor_variable_num = int(self.predictor_variable_num_entry.get())
        self.process.simulations = config.simulations = int(self.simulations_entry.get())
        self.process.start_year = config.start_year = int(self.start_year_entry.get())
        self.process.end_year = config.end_year = int(self.end_year_entry.get())


    def create_process_module(self):
        # Process Module
        self.process_module.grid(row=1, column=0, sticky='nsew')
        self.process_module.grid_columnconfigure(4, weight=1)
        self.process_module.grid_rowconfigure(0, weight=0)
        self.required_qualification_label.grid(row=5, column=0, pady=12, padx=10)
        self.required_qualification_label.insert(0, '550')
        self.observation_unique_ID_entry.grid(row=5, column=1, pady=12, padx=10)
        self.observation_unique_ID_entry.insert(0, 'IDfg')
        self.target_variable_name_entry.grid(row=5, column=2, pady=12, padx=10)
        self.target_variable_name_entry.insert(0, 'HR')
        self.predictor_variable_num_entry.grid(row=6, column=0, pady=12, padx=10)
        self.predictor_variable_num_entry.insert(0, '5')
        self.simulations_entry.grid(row=6, column=1, pady=12, padx=10)
        self.simulations_entry.insert(0, '1')
        self.start_year_entry.grid(row=6, column=2, pady=12, padx=10)
        self.start_year_entry.insert(0, '2015')
        self.end_year_entry.grid(row=6, column=3, pady=12, padx=10)
        self.end_year_entry.insert(0, '2022')
        self.processing_button.grid(row=8, column=0, pady=12, padx=10, sticky='nsew')
        self.process_module_pb.grid(row=8, column=1, columnspan=3, padx=0, pady=0)
        self.process_module_pb_label.grid(row=9, column=1, columnspan=3, pady=0, padx=0)
        self.set_process_settings()
        return self.process_module


    def create_dataset_module(self):
        # table model
        self.dataset_module = customtkinter.CTkFrame(master=self.root)
        self.dataset_module.grid(row=0, column=1, rowspan=1, pady=0, padx=0, sticky='nsew')
        self.table = Table(self.dataset_module, dataframe=pybaseball.batting_stats(2015, 2022, qual=550), width=300, maxcellwidth=1500, showtoolbar=True, showstatusbar=True)
        self.table.adjustColumnWidths()
        self.table.show()
        return self.dataset_module


    def update_dataset_module(self):
        self.root.update_idletasks()
        del self.table
        del self.dataset_module
        self.dataset_module = customtkinter.CTkFrame(master=self.root)
        self.dataset_module.grid(row=0, column=1, rowspan=1, pady=0, padx=0, sticky='nsew')
        self.table = Table(self.dataset_module, dataframe=self.process.dataset_past, width=300, maxcellwidth=1500, showtoolbar=True, showstatusbar=True)
        print('Creating table')
        self.table.adjustColumnWidths()
        print('Adjusting column widths')
        self.table.redraw()
        print('Showing table in module')
        print('Finished')


ui = UserInterface()
process_Z = ui.process
# Run user interface
ui.root.mainloop()
