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
import matplotlib.pyplot as plt

# print current directory
plt.rcParams['axes.titlesize'] = 20  # Set the font size for plot titles
plt.rcParams['axes.labelsize'] = 18  # Set the font size for axis labels
plt.rcParams['xtick.labelsize'] = 16  # Set the font size for X tick labels
plt.rcParams['ytick.labelsize'] = 16  # Set the font size for Y tick labels
plt.rcParams['legend.fontsize'] = 14  # Set the font size for legends
plt.rcParams['font.size'] = 12  # Default font size for all text (if not specified otherwise)

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
columns = ['Key', 'r_a', 'r_p', 'SMA']
df = pd.DataFrame(columns=columns)

# loop through the objects
for i, key in enumerate(keys):
    state = data[key]['state']
    SMA, ECC, INC, AoP, LON, TA = cartesian_to_keplerian(state, bodies.get_body("Earth").gravitational_parameter)

    # Calculate the apogee and perigee for each object
    r_a = SMA * (1 + ECC)
    r_p = SMA * (1 - ECC)

    # Append the results to the dataframe
    new_row = pd.DataFrame([{'Key': key, 'r_a': r_a, 'r_p': r_p, 'SMA': SMA}])
    df = pd.concat([df, new_row], ignore_index=True)

# save the row with key 36585 in a separate dataframe and remove it from the original dataframe
df_GPS = df[df['Key'] == 36585]
df = df[df['Key'] != 36585]

# # calculate r_p of the object - r_a of GPS
# df['p_a_distance'] = df['r_p'] - df_GPS['r_a'].values[0]
# df['a_p_distance'] = df_GPS['r_p'].values[0] - df['r_a']

# # If this distance is larger than 1000 km we discard the object
# df = df[(df['p_a_distance'] < 1000000) & (df['a_p_distance'] < 1000000)]

# # save the dataframe to a csv file
# file_path_to_save = "filtered_results.csv"
# df.to_csv(file_path_to_save, index=False)


def create_gabbard_plot_sma_apogee_perigee(df, title, df_gps):
    semi_major_axis = df['SMA'] / 1000  # Convert from meters to altitude in kilometers
    apocenter = df['r_a'] / 1000  # Calculate apogee /1000
    pericenter = df['r_p'] / 1000  # Calculate perigee

    # Extract the apogee and perigee values for the GPS satellite (assuming df_gps is a DataFrame with one row)
    ra_gps = df_gps['r_a'].values[0] / 1000 +1000 # Ensure division applies to the numeric value
    rp_gps = df_gps['r_p'].values[0] / 1000 -1000

    plt.figure(figsize=(10, 6))
    plt.scatter(semi_major_axis, apocenter, s=10, c='blue', alpha=0.5, label='Apocenter Debris')
    plt.scatter(semi_major_axis, pericenter, s=10, c='red', alpha=0.5, label='Pericenter Debris')

    # Add horizontal lines for the GPS apogee and perigee
    plt.axhline(y=ra_gps, color='green', linestyle='--', label=f'NAVSTAR-65 Apocenter + 1000 km)')
    plt.axhline(y=rp_gps, color='purple', linestyle='--', label=f'NAVSTAR-65 Pericenter - 1000 km)')

    plt.title(title)
    plt.xlabel('Semi-major Axis (km)')
    plt.ylabel('Distance from Earth Center (km)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
create_gabbard_plot_sma_apogee_perigee(df, 'Gabbard Plot: Debris VS NAVSTAR-65',df_GPS)

print('breakpoint')