from tudatpy.astro.time_conversion import epoch_from_date_time_components
from tudatpy.kernel.astro import element_conversion
from TudatPropagator import propagate_state_and_covar, propagate_orbit
from ConjunctionUtilities import compute_TCA
from tudatpy.numerical_simulation import environment
import TudatPropagator as prop

import os
import json
import time
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm


# constants
delta_t_days = 2
spherical_harmonic_degree = 8
spherical_harmonic_order = 8
central_bodies = ['Earth']
bodies_to_create = ['Earth', 'Moon', 'Sun']
bodies = prop.tudat_initialize_bodies(bodies_to_create)

# for the RK4 integrator
step_rk4 = -10        # initial step size [s]

# for the RKF78 integrator
step_rkf78 = 10       # initial step size
max_step_rkf78 = 50   # maximum step size
min_step_rkf78 = 1    # minimum step size
atol_rkf78 = 1e-8     # absolute tolerance
rtol_rkf78 = 1e-8     # relative tolerance

file_path = "results_tca_updated.csv"
data_TCA = pd.read_csv(file_path)

# rank the pd df by the miss distance from low to high
data_TCA = data_TCA.sort_values(by='Miss Distance [m]')

# only keep the first 10 rows
data_TCA = data_TCA.head(10)
keys_TCA = data_TCA['Key']

def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data

file_path_RSO_catalog = "data\group4\estimated_rso_catalog.pkl"
data_RSO_catalog = read_pkl(file_path_RSO_catalog)[0]
keys_RSO_catalog = data_RSO_catalog.keys()

integrator_type = 'rk4'
int_params={}
int_params['step'] = 100



GPS_TLE = environment.Tle(
    '1 41328U 16007A   24081.19136676  .00000027  00000-0  00000-0 0  9995',
    '2 41328  55.0887 121.9461 0078577 237.3631 121.9508  2.0056633559433'
)
t0 = epoch_from_date_time_components(2024, 3, 23, 5, 30, 0)
GPS_ephemeris = environment.TleEphemeris( "Earth", "J2000", GPS_TLE, False )
sat_cart = GPS_ephemeris.cartesian_state( t0 )
trange = [t0, t0+3600*48]

sat_dict = {}
sat_dict['C_D'] = 2.3
sat_dict['C_R'] = 1.0
sat_dict['area'] = 1.0
sat_dict['mass'] = 20.0
sat_dict['sph_deg'] = 8
sat_dict['sph_ord'] = 8
sat_dict['central_bodies'] = ['Earth']
sat_dict['bodies_to_create'] = ['Earth', 'Moon', 'Sun']



def detailed_propagation(state, trange, state_params, int_params):


    # Get RSO for the GPS satellite
    RSO_GPS = data_RSO_catalog[36585]



    # Integration parameters
    int_params = {}

    if integrator_type == 'rk4':
        int_params['tudat_integrator'] = 'rk4'
        int_params['step'] = step_rk4
    elif integrator_type == 'rkf78':
        int_params['tudat_integrator'] = 'rkf78'
        int_params['step'] = step_rkf78
        int_params['max_step'] = max_step_rkf78
        int_params['min_step'] = min_step_rkf78
        int_params['rtol'] = rtol_rkf78
        int_params['atol'] = atol_rkf78

    # propagate the GPS satellite's state and covariance if the JSON file is not present in the JSON_files directory
    if not os.path.exists(f"dependent_variables\\dep_var_{str(36585)}.csv"):

        # get start time
        time_start = time.time()

        print("Propagating the GPS satellite's state...\n")
        tf_GPS, Xf_GPS, dep_var_GPS = prop.propagate_orbit_and_get_dep_var(X_GPS, trange, GPS_params, int_params)

        # Turn dep_var_GPS dictionary into a numpy array
        dep_var_GPS_array = np.zeros((len(tf_GPS), 6))
        for i in range(len(tf_GPS)):
            dep_var_GPS_array[i] = dep_var_GPS[tf_GPS[i]]

        print(f"Propagation of the GPS satellite took {time.time() - time_start} seconds")

        # Save the results to a JSON file for each object
        file_name = "dependent_variables\\dep_var_" + str(36585) + ".csv"

        # create pandas dataframe with the dep_var_GPS_array and the tf_GPS array as the first column
        df = pd.DataFrame(data=dep_var_GPS_array, columns=['ax SRP [m/s^2]', 'ay SRP [m/s^2]', 'az SRP [m/s^2]', 'rx Sun [m]', 'ry Sun [m]', 'rz Sun [m]'])
        df['x [m]'] = Xf_GPS[:, 0]
        df['y [m]'] = Xf_GPS[:, 1]
        df['z [m]'] = Xf_GPS[:, 2]
        df['vx [m/s]'] = Xf_GPS[:, 3]
        df['vy [m/s]'] = Xf_GPS[:, 4]
        df['vz [m/s]'] = Xf_GPS[:, 5]
        df.insert(0, 'Time', tf_GPS)

        # save the dataframe to a csv file
        df.to_csv(file_name)

    # get start time
    time_start = time.time()

    print("Propagating the GPS satellite's state...\n")
    tf_debris, Xf_debris, dep_var_debris = prop.propagate_orbit_and_get_dep_var(X_debris, trange, debris_params,
                                                                       int_params)

    # Turn dep_var_GPS dictionary into a numpy array
    dep_var_debris_array = np.zeros((len(tf_debris), 6))
    for i in range(len(tf_debris)):
        dep_var_debris_array[i] = dep_var_debris[tf_debris[i]]

    print(f"Propagation of {key} took {time.time() - time_start} seconds")

    # Save the results to a JSON file for each object
    file_name = "dependent_variables\\dep_var_" + str(key) + ".csv"

    # create pandas dataframe with the dep_var_GPS_array and the tf_GPS array as the first column
    df = pd.DataFrame(data=dep_var_debris_array, columns=['ax SRP [m/s^2]', 'ay SRP [m/s^2]', 'az SRP [m/s^2]', 'rx Sun [m]', 'ry Sun [m]', 'rz Sun [m]'])
    df['x [m]'] = Xf_debris[:, 0]
    df['y [m]'] = Xf_debris[:, 1]
    df['z [m]'] = Xf_debris[:, 2]
    df['vx [m/s]'] = Xf_debris[:, 3]
    df['vy [m/s]'] = Xf_debris[:, 4]
    df['vz [m/s]'] = Xf_debris[:, 5]
    df.insert(0, 'Time', tf_debris)

    # save the dataframe to a csv file
    df.to_csv(file_name)
    return None

if __name__ == "__main__":
    status = detailed_propagation(keys_TCA, 'rk4')

    print('breakpoint')
