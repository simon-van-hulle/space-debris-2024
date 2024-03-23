import pickle as pkl
import pandas as pd
import numpy as np
import os

from tudatpy.astro.element_conversion import cartesian_to_keplerian
from tudatpy import constants
from tudatpy.data import save2txt
from tudatpy import constants
from tudatpy.interface import spice
from scipy.fft import fft, fftfreq
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
import TudatPropagator as prop

# print current directory
print(os.getcwd())



# file path
file_path = "data\group4\estimated_rso_catalog.pkl"

central_bodies = ['Earth']
bodies_to_create = ['Earth', 'Moon', 'Sun']
# bodies = prop.tudat_initialize_bodies(bodies_to_create)
global_frame_origin = 'Earth'
global_frame_orientation = 'ECLIPJ2000'
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)
bodies = environment_setup.create_system_of_bodies(body_settings)

def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data

data = read_pkl(file_path)[0]
keys = data.keys()

# Create dataframe to store the results
columns = ['Key', 'r_a', 'r_p']
df = pd.DataFrame(columns=columns)

# loop through the objects
for i, key in enumerate(keys):
    state = data[key]['state']
    SMA, ECC, INC, AoP, LON, TA = cartesian_to_keplerian(state, bodies.get_body("Earth").gravitational_parameter)

    # Calculate the apogee and perigee for each object
    r_a = SMA * (1 + ECC)
    r_p = SMA * (1 - ECC)

    # Append the results to the dataframe
    df = df.append({'Key': key, 'r_a': r_a, 'r_p': r_p}, ignore_index=True)

# save the row with key 36585 in a separate dataframe and remove it from the original dataframe
df_GPS = df[df['Key'] == 36585]
df = df[df['Key'] != 36585]

# calculate r_p of the object - r_a of GPS
df['p_a_distance'] = df['r_p'] - df_GPS['r_a'].values[0]
df['a_p_distance'] = df_GPS['r_p'].values[0] - df['r_a']

# If this distance is larger than 1000 km we discard the object
df = df[(df['p_a_distance'] < 1000000) & (df['a_p_distance'] < 1000000)]

# save the dataframe to a csv file
file_path_to_save = "filtered_results.csv"
df.to_csv(file_path_to_save, index=False)

print('breakpoint')