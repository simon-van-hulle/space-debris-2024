from tudatpy.astro.time_conversion import epoch_from_date_time_components
from tudatpy.kernel.astro import element_conversion
from TudatPropagator import propagate_state_and_covar, propagate_orbit
from ConjunctionUtilities import compute_TCA
from tudatpy.numerical_simulation import environment
import TudatPropagator as prop

import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

# constants
delta_t_days = 2
spherical_harmonic_degree = 8
spherical_harmonic_order = 8
central_bodies = ['Earth']
bodies_to_create = ['Earth', 'Moon', 'Sun']
bodies = prop.tudat_initialize_bodies(bodies_to_create)

# for the RK4 integrator
step_rk4 = 10

# for the RKF78 integrator
step_rkf78 = 10       # initial step size
max_step_rkf78 = 50   # maximum step size
min_step_rkf78 = 1    # minimum step size
atol_rkf78 = 1e-8     # absolute tolerance
rtol_rkf78 = 1e-8     # relative tolerance

# file path
file_path = "data\group4\estimated_rso_catalog.pkl"


def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data


data = read_pkl(file_path)[0]
keys = data.keys()

# Get the keys of the objects to be propagated after the filtering
file_path_filter_results = "filtered_results.csv"
data_filter_results = pd.read_csv(file_path_filter_results)
filtered_keys = data_filter_results['Key']

# set the objects to be propagated and the integrator type
#objects_in = list(keys)[:3]  # first 2 keys
objects_in = {36585, 91861, 91368}
objects_in = filtered_keys
# select all keys except 36585
#objects_in = [key for key in keys if key != 36585]
integrator_type = 'rk4'


# Initialize empty list
TCA_list = []
rho_list = []
def get_TCA(objects_in, integrator_type):
    for i, key in tqdm(enumerate(objects_in)):

        # Get RSO for this specific object
        RSO_object = data[key]

        # Initial time conversions to seconds since J2000
        t0 = RSO_object['UTC']
        t0 = epoch_from_date_time_components(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second)

        # Create time interval
        tf = t0 + 3600 * 48 # replace the last number by the amount of days wanted for the propagation
        trange = np.array([t0, tf])

        # Get GPS satellite state at t0
        X1 = data[36585]['state']

        # Get debris state at t0
        X2 = RSO_object['state']

        # create parameter dictionary for the GPS satellite
        rso1_params = {}
        rso1_params['mass'] = 1630.
        rso1_params['area'] = 13.73259
        rso1_params['Cd'] = 2.2
        rso1_params['Cr'] = 1.3
        rso1_params['sph_deg'] = spherical_harmonic_degree
        rso1_params['sph_ord'] = spherical_harmonic_order
        rso1_params['central_bodies'] = central_bodies
        rso1_params['bodies_to_create'] = bodies_to_create

        rso2_params = RSO_object
        rso2_params['sph_deg'] = spherical_harmonic_degree
        rso2_params['sph_ord'] = spherical_harmonic_order
        rso2_params['central_bodies'] = central_bodies
        rso2_params['bodies_to_create'] = bodies_to_create

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

        # Expected result
        TCA_true = 0  # 764445600.0
        rho_true = 0.

        # get starting time to measure runtime
        start = time.time()

        print("Computing TCA for object", key, "using", integrator_type, "integrator")
        Ts, rhos = compute_TCA(X1, X2, trange, rso1_params, rso2_params,
                                       int_params, bodies)

        # Append results to list
        TCA_list.append(Ts)
        rho_list.append(rhos)

        print('')
        print('TCA unit test runtime [seconds]:', time.time() - start)
        print('TCA [seconds]:', Ts[0] - TCA_true)
        print('Miss distance error [m]:', rhos[0] - rho_true)

    print('TCA_list:', TCA_list)
    print('rho_list:', rho_list)
    return TCA_list, rho_list

if __name__ == "__main__":
    TCA_list, rho_list = get_TCA(objects_in, integrator_type)

    # reshape the lists to a 1D list
    TCA_list = [item for sublist in TCA_list for item in sublist]
    rho_list = [item for sublist in rho_list for item in sublist]

    # save results to a csv file
    results = pd.DataFrame()
    results['Key'] = list(objects_in)
    results['TCA [s]'] = list(TCA_list)
    results['Miss Distance [m]'] = list(rho_list)
    results.to_csv("results_tca_updated.csv", index=False)
    print('breakpoint')