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
from tkinter import scrolledtext
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

        # Process Module
        self.process_module = customtkinter.CTkFrame(master=self.root)
        self.process_module_pb = ttk.Progressbar(master=self.process_module, orient='horizontal', mode='determinate', length=280)
        self.process_module_required_qualification_label = customtkinter.CTkLabel(master=self.process_module, text='Required Qualification:', font=('Roboto', 16))
        self.required_qualification_entry = customtkinter.CTkEntry(master=self.process_module, placeholder_text='Required Qualification')
        self.process_module_observation_unique_ID_label = customtkinter.CTkLabel(master=self.process_module, text='Observation Unique ID:', font=('Roboto', 16))
        self.observation_unique_ID_entry = customtkinter.CTkEntry(master=self.process_module, placeholder_text='Observation Unique ID')
        self.process_module_target_variable_name_label = customtkinter.CTkLabel(master=self.process_module, text='Target Variable Name:', font=('Roboto', 16))
        self.target_variable_name_entry = customtkinter.CTkEntry(master=self.process_module, placeholder_text='Target Variable Name')
        self.process_module_predictor_variable_num_label = customtkinter.CTkLabel(master=self.process_module, text='Predictor Variable Num:', font=('Roboto', 16))
        self.predictor_variable_num_entry = customtkinter.CTkEntry(master=self.process_module, placeholder_text='Predictor Variable Num')
        self.process_module_simulations_label = customtkinter.CTkLabel(master=self.process_module, text='Simulations:', font=('Roboto', 16))
        self.simulations_entry = customtkinter.CTkEntry(master=self.process_module, placeholder_text='Simulations')
        self.process_module_start_year_label = customtkinter.CTkLabel(master=self.process_module, text='Start Year:', font=('Roboto', 16))
        self.start_year_entry = customtkinter.CTkEntry(master=self.process_module, placeholder_text='Start Year')
        self.process_module_end_year_label = customtkinter.CTkLabel(master=self.process_module, text='End Year:', font=('Roboto', 16))
        self.end_year_entry = customtkinter.CTkEntry(master=self.process_module, placeholder_text='End Year')
        self.processing_button = customtkinter.CTkButton(master=self.process_module, text='Start Processing', command=self.create_process_thread)
        self.process_module_pb_label = ttk.Label(master=self.process_module, text=self.update_progress_label)

        # Output Module
        self.output_module = customtkinter.CTkFrame(master=self.root)
        self.output_module_title_label = customtkinter.CTkLabel(master=self.output_module, text='Output Module:', font=('Roboto', 24))
        self.outputText = scrolledtext.ScrolledText(self.output_module)
        self.outputText.config(state='disabled')
        self.outputText.grid(row=1, column=0, columnspan=4, sticky='nsew')

        # Dataset Module
        self.table = None
        self.load_thread = None
        self.process_request_thread = None
        
        # Create Modules
        self.create_load_module()
        self.create_process_module()
        self.create_output_module()
        self.dataset_module = self.create_dataset_module()

        # Set Config
        config.ui = self


    def create_root_frame(self):
        # Root Frame
        root = customtkinter.CTk()
        root.minsize(1580, 980)
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
        self.outputText.config(state='normal')
        self.outputText.insert(config.current_output_line_num, "Process Active Library Number = " + self.active_library_entry.get() + "\n")
        self.outputText.config(state='disabled')
        self.process.active_library = config.active_library = int(self.active_library_entry.get())


    def create_load_module(self):
        # LoadModule
        self.load_module.grid(row=0, column=0, sticky='nsew')
        self.load_module.grid_columnconfigure(0, weight=0)
        self.load_module.grid_rowconfigure(0, weight=0)
        self.load_module_title_label.grid(row=0, column=0, columnspan=3, pady=12, padx=30)
        self.load_module_subtitle_label.grid(row=1, column=0, columnspan=3, pady=12, padx=10)
        self.active_library_entry.grid(row=2, column=0, pady=12, padx=10)
        self.active_library_entry.insert(0, '1')
        self.button1.grid(row=2, column=1, pady=12, padx=0, sticky='nsew')
        self.load_module_pb.grid(row=3, column=0, columnspan=3, padx=0, pady=0)
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
        # Set Process Values Based on Entries
        self.process.required_qualification = config.required_qualification = int(self.required_qualification_entry.get())
        self.process.observation_unique_ID = config.observation_unique_ID = self.observation_unique_ID_entry.get()
        self.process.target_variable_name = config.target_variable_name = self.target_variable_name_entry.get()
        self.process.predictor_variable_num = config.predictor_variable_num = int(self.predictor_variable_num_entry.get())
        self.process.simulations = config.simulations = int(self.simulations_entry.get())
        self.process.start_year = config.start_year = int(self.start_year_entry.get())
        self.process.end_year = config.end_year = int(self.end_year_entry.get())

        # Output Process Settings
        self.outputText.config(state='normal')
        self.outputText.insert(config.current_output_line_num, "Required Qualifications = " + str(config.required_qualification) + "\n")
        self.outputText.insert(config.current_output_line_num, "Observation Unique ID = " + str(config.observation_unique_ID) + "\n")
        self.outputText.insert(config.current_output_line_num, "Process Target Variable = " + str(config.target_variable_name) + "\n")
        self.outputText.insert(config.current_output_line_num, "Predictor Variable Number = " + str(config.predictor_variable_num) + "\n")
        self.outputText.insert(config.current_output_line_num, "Simulations = " + str(config.simulations) + "\n")
        self.outputText.insert(config.current_output_line_num, "End Year = " + str(config.end_year) + "\n")
        self.outputText.config(state='disabled')


    def create_process_module(self):
        # Process Module Grid Configurations
        self.process_module.grid(row=1, column=0, sticky='nsew')
        self.process_module.grid_columnconfigure(4, weight=1)
        self.process_module.grid_rowconfigure(0, weight=0)

        # Required Qualifications
        self.process_module_required_qualification_label.grid(row=5, column=0, padx=10)
        self.required_qualification_entry.grid(row=5, column=1, pady=12, padx=10)
        self.required_qualification_entry.insert(0, '550')

        # Observation Unique ID
        self.process_module_observation_unique_ID_label.grid(row=5, column=3, padx=10)
        self.observation_unique_ID_entry.grid(row=5, column=4, pady=12, padx=10)
        self.observation_unique_ID_entry.insert(0, 'IDfg')

        # Target Variable Name
        self.process_module_target_variable_name_label.grid(row=6, column=0, padx=10)
        self.target_variable_name_entry.grid(row=6, column=1, pady=12, padx=10)
        self.target_variable_name_entry.insert(0, 'HR')

        # Predictor Variable Number
        self.process_module_predictor_variable_num_label.grid(row=6, column=3, padx=10)
        self.predictor_variable_num_entry.grid(row=6, column=4, pady=12, padx=10)
        self.predictor_variable_num_entry.insert(0, '5')

        # Simulations
        self.process_module_simulations_label.grid(row=7, column=0, padx=10)
        self.simulations_entry.grid(row=7, column=1, pady=12, padx=10)
        self.simulations_entry.insert(0, '1')

        # Start Year
        self.process_module_start_year_label.grid(row=7, column=3, padx=10)
        self.start_year_entry.grid(row=7, column=4, pady=12, padx=10)
        self.start_year_entry.insert(0, '2015')

        # End Year
        self.process_module_end_year_label.grid(row=8, column=0, padx=10)
        self.end_year_entry.grid(row=8, column=1, pady=12, padx=10)
        self.end_year_entry.insert(0, '2022')

        # Button and Progress Bar
        self.processing_button.grid(row=8, column=3, columnspan=2, pady=12, padx=10, sticky='nsew')
        self.process_module_pb.grid(row=9, column=1, columnspan=3, padx=0, pady=0)
        self.process_module_pb_label.grid(row=9, column=1, columnspan=3, pady=0, padx=0)

        self.set_process_settings()
        return self.process_module


    def create_output_module(self):
        # Output Module
        self.output_module.grid(row=2, column=0, sticky='nsew')
        self.output_module_title_label.grid(row=0, column=0, pady=12, padx=30)


    def create_dataset_module(self):
        # Table Model
        self.dataset_module = customtkinter.CTkFrame(master=self.root)
        self.dataset_module.grid(row=0, column=1, rowspan=3, pady=0, padx=0, sticky='nsew')
        self.table = Table(self.dataset_module, dataframe=pybaseball.batting_stats(2015, 2022, qual=550), width=300, maxcellwidth=1500, showtoolbar=True, showstatusbar=True)
        self.table.adjustColumnWidths()
        self.table.show()
        return self.dataset_module


    def update_dataset_module(self):
        self.root.update_idletasks()
        del self.table
        del self.dataset_module
        self.dataset_module = customtkinter.CTkFrame(master=self.root)
        self.dataset_module.grid(row=0, column=1, rowspan=3, pady=0, padx=0, sticky='nsew')
        self.table = Table(self.dataset_module, dataframe=self.process.dataset_past, width=300, maxcellwidth=1500, showtoolbar=True, showstatusbar=True)
        self.outputText.config(state='normal')
        self.outputText.insert(config.current_output_line_num, "Creating table" + "\n")
        self.outputText.config(state='disabled')
        self.table.adjustColumnWidths()
        self.outputText.config(state='normal')
        self.outputText.insert(config.current_output_line_num, "Adjusting column widths" + "\n")
        self.outputText.config(state='disabled')
        self.table.redraw()
        self.outputText.config(state='normal')
        self.outputText.insert(config.current_output_line_num, "Showing table in module" + "\n")
        self.outputText.config(state='disabled')
        self.outputText.config(state='normal')
        self.outputText.insert(config.current_output_line_num, "Finished updating the table operations" + "\n")
        self.outputText.config(state='disabled')


ui = UserInterface()
process_Z = ui.process
# Run user interface
ui.root.mainloop()
