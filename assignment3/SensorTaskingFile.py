import numpy as np
import json
from datetime import datetime

###############################################################################
# This script will generate the sensor tasking file in correct format.
#
# Edit the fields in this section based on your desired sensor tasks.
#
###############################################################################

# File name should include group number
json_file = 'group4_sensor_tasking_file.json'

# Sensor type is a string, either 'radar' or 'optical'
# sensor_type = 'radar'
sensor_type = 'optical'

# Sensor location: data for Diego Garcia GEODSS
latitude_radians = np.deg2rad(7.411754)
longitude_radians = np.deg2rad(72.452285)
height_meters = -73.58 + 1.2192

# Indicate start times and object IDs for each task.
# You must ensure that sensor constraints are met.
# If an object is not visible or violates, no measurements will be generated.

# All sensor tasks will track the object for 5 minutes, producing measurements
# at an interval of 10 seconds. If the visible pass is shorter than 5 minutes
# from the start time, measurements will be computed until the object is out
# of view.

# Start time should be given as a datetime object in UTC
# Round times to nearest second

# Task 1
task1_start_UTC = datetime(2024, 3, 22, 0, 27, 30)
task1_object_id = 91104

# Task 2
task2_start_UTC = datetime(2024, 3, 22, 0, 22, 20)
task2_object_id = 91861

# Task 3
task3_start_UTC = datetime(2024, 3, 22, 0, 17, 10)
task3_object_id = 91787

# Task 4
task4_start_UTC = datetime(2024, 3, 21, 23, 36, 10)
task4_object_id = 91401

# Task 5
task5_start_UTC = datetime(2024, 3, 21, 20, 27, 10)
task5_object_id = 91821

# Task 6
task6_start_UTC = datetime(2024, 3, 21, 19, 14, 10)
task6_object_id = 91518

# Task 7
task7_start_UTC = datetime(2024, 3, 21, 18, 37, 50)
task7_object_id = 91368

# Task 8
task8_start_UTC = datetime(2024, 3, 21, 18, 32, 40)
task8_object_id = 91714

# Task 9
task9_start_UTC = datetime(2024, 3, 21, 18, 26, 30)
task9_object_id = 91447

# Task 10
task10_start_UTC = datetime(2024, 3, 21, 15, 13, 40)
task10_object_id = 91438




###############################################################################
# This code will generate the tasking file in correct JSON format
#
# DO NOT EDIT
#
###############################################################################




tasking_dict = {}
tasking_dict['sensor_type'] = sensor_type
tasking_dict['latitude_radians'] = latitude_radians
tasking_dict['longitude_radians'] = longitude_radians
tasking_dict['height_meters'] = height_meters

tasking_dict['1'] = {}
tasking_dict['1']['start'] = task1_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['1']['obj_id'] = task1_object_id

tasking_dict['2'] = {}
tasking_dict['2']['start'] = task2_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['2']['obj_id'] = task2_object_id

tasking_dict['3'] = {}
tasking_dict['3']['start'] = task3_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['3']['obj_id'] = task3_object_id

tasking_dict['4'] = {}
tasking_dict['4']['start'] = task4_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['4']['obj_id'] = task4_object_id

tasking_dict['5'] = {}
tasking_dict['5']['start'] = task5_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['5']['obj_id'] = task5_object_id

tasking_dict['6'] = {}
tasking_dict['6']['start'] = task6_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['6']['obj_id'] = task6_object_id

tasking_dict['7'] = {}
tasking_dict['7']['start'] = task7_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['7']['obj_id'] = task7_object_id

tasking_dict['8'] = {}
tasking_dict['8']['start'] = task8_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['8']['obj_id'] = task8_object_id

tasking_dict['9'] = {}
tasking_dict['9']['start'] = task9_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['9']['obj_id'] = task9_object_id

tasking_dict['10'] = {}
tasking_dict['10']['start'] = task10_start_UTC.strftime('%Y-%m-%d %H:%M:%S')
tasking_dict['10']['obj_id'] = task10_object_id


with open(json_file, 'w') as f:
    json.dump(tasking_dict, f)





























