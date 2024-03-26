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
step_rk4 = 1        # initial step size [s]

# for the RKF78 integrator
step_rkf78 = 10       # initial step size
max_step_rkf78 = 50   # maximum step size
min_step_rkf78 = 1    # minimum step size
atol_rkf78 = 1e-8     # absolute tolerance
rtol_rkf78 = 1e-8     # relative tolerance

file_path = "results_tca_updated.csv"
data_TCA = pd.read_csv(file_path)

# filter out the objects with a miss distance higher than 10km
data_TCA = data_TCA[data_TCA['Miss Distance [m]'] < 10000]
keys_TCA = data_TCA['Key']

def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data

file_path_RSO_catalog = "data\group4\estimated_rso_catalog.pkl"
data_RSO_catalog = read_pkl(file_path_RSO_catalog)[0]
keys_RSO_catalog = data_RSO_catalog.keys()


# filter out the objects with a miss distance higher than 10km



def detailed_propagation(objects_in, integrator_type):

    for i, key in enumerate(objects_in):

        # Ensure the key is an integer
        key = int(key)

        # Get RSO for this specific object
        RSO_object = data_RSO_catalog[key]

        # Get RSO for the GPS satellite
        RSO_GPS = data_RSO_catalog[36585]

        # Get the TCA for this specific object
        TCA_object = data_TCA[data_TCA['Key'] == key]['TCA [s]'].values[0]

        # Initial time conversions to seconds since J2000
        t0 = RSO_object['UTC']
        t0 = epoch_from_date_time_components(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second)

        tf = t0 + 3600 * 48          # 2 hours after the TCA

        # Create time interval
        trange = np.array([t0, tf])

        # Get GPS satellite state and covariance at t0
        X_debris = RSO_object['state']
        P_debris = RSO_object['covar']

        # Object parameters
        debris_params = RSO_object
        debris_params['sph_deg'] = spherical_harmonic_degree
        debris_params['sph_ord'] = spherical_harmonic_order
        debris_params['central_bodies'] = central_bodies
        debris_params['bodies_to_create'] = bodies_to_create

        # Get GPS satellite state and covariance at t0
        X_GPS = RSO_GPS['state']
        P_GPS = RSO_GPS['covar']

        # Object parameters
        GPS_params = RSO_GPS
        GPS_params['sph_deg'] = spherical_harmonic_degree
        GPS_params['sph_ord'] = spherical_harmonic_order
        GPS_params['central_bodies'] = central_bodies
        GPS_params['bodies_to_create'] = bodies_to_create

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
        if not os.path.exists(f"JSON_files\\{str(36585)}.json"):

            # get start time
            time_start = time.time()

            print("Propagating the GPS satellite's covariance...")
            tf_GPS, Xf_GPS, Pf_GPS = propagate_state_and_covar(X_GPS, P_GPS, trange, GPS_params, int_params)

            print("Propagating the GPS satellite's state...\n")
            tf_GPS, Xf_GPS = prop.propagate_orbit(X_GPS, trange, GPS_params, int_params)

            print(f"Propagation of the GPS satellite took {time.time() - time_start} seconds")

            # Save the results to a JSON file for each object
            file_name = "JSON_files\\" + str(36585) + ".json"

            with open(file_name, 'w') as f:
                json.dump({'Times': tf_GPS.tolist(), 'Positions [m]': Xf_GPS.tolist(), 'Covariances': Pf_GPS.tolist()}, f)

        # propagate the object's state and covariance
        print(f"Propagating {key}'s covariance...")
        tf_debris, Xf_debris, Pf_debris = propagate_state_and_covar(X_debris, P_debris, trange, debris_params, int_params)

        print(f"Propagating {key}'s state...\n")
        tf_debris, Xf_debris = prop.propagate_orbit(X_debris, trange, debris_params, int_params)

        # Save the results to a JSON file for each object
        file_name = "JSON_files\\" + str(key) + ".json"

        with open(file_name, 'w') as f:
            json.dump({'Times': tf_debris.tolist(), 'Positions [m]': Xf_debris.tolist(), 'Covariances': Pf_debris.tolist()}, f)

    return None

if __name__ == "__main__":
    status = detailed_propagation(keys_TCA, 'rk4')

    print('breakpoint')
