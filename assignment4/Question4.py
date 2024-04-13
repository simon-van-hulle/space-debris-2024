import matplotlib
import scienceplots
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import os
import pandas as pd
import sys
sys.path.append('assignment3')
import EstimationUtilities
import ukf_tuning

settings = ukf_tuning.UkfSettings()
intsettings = settings.get_int_params()
filter_params = settings.get_filter_params()
bodies = ukf_tuning.tudat_initialize_bodies()


# Set the font family and size to use for Matplotlib figures.
plt.style.use('science')
matplotlib.rcParams.update({'font.size': 16, 'font.family': 'serif'})

file_path = 'assignment3\group4_sensor_tasking_file.json'
# Open the JSON file and load its contents into a Python dictionary
with open(file_path, 'r') as file:
    data = json.load(file)


# Replace 'your_directory_path' with the actual directory path that contains your .pkl files
directory_path = 'assignment4\data\group4'

# Data structure to store all the returned values from each file
all_data = []

# Loop through each file in the directory and process it with the provided function
for filename in os.listdir(directory_path):
    if filename.endswith('.pkl'):
        filepath = os.path.join(directory_path, filename)
        state_params, meas_dict, sensor_params = EstimationUtilities.read_measurement_file(filepath)
        all_data.append({
            'filename': filename,
            'state_params': state_params,
            'meas_dict': meas_dict,
            'sensor_params': sensor_params
        })

print('reeeeee')
# Data structure to store all the returned values from each file
ukf_results = []

# Loop through each file in the directory and process it with the provided function
for filename in os.listdir(directory_path):
    if filename.endswith('.pkl') and filename == 'q3_meas_rso_99004.pkl':
        print(filename)
        filepath = os.path.join(directory_path, filename)
        
        # Read the measurement file
        state_params, meas_dict, sensor_params = EstimationUtilities.read_measurement_file(filepath)
        
        # Now run the Unscented Kalman Filter with the parameters
        ukf_result = EstimationUtilities.ukf(state_params, meas_dict, sensor_params, intsettings, filter_params, bodies)
        
        # Append the result of the UKF for this file to the results list
        ukf_results.append({
            'filename': filename,
            'ukf_result': ukf_result
        })
print(ukf_results)

print('UKF processing complete.')