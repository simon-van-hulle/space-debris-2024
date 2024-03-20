import pickle as pkl
import os
import datetime
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.astro.time_conversion import epoch_from_date_time_components
from tudatpy.kernel.astro import element_conversion
from TudatPropagator import propagate_state_and_covar
from ConjunctionUtilities import compute_TCA
from tudatpy.numerical_simulation import environment
import TudatPropagator as prop
import numpy as np
import pandas as pd
import time

# print current directory
print(os.getcwd())

# constants
delta_t_days = 2
spherical_harmonic_degree = 8
spherical_harmonic_order = 8
central_bodies = ['Sun']
bodies_to_create = ['Earth', 'Moon', 'Sun']
tudat_integrator = 'rkf78'
step = 10
max_step = 50
min_step = 1
atol = 1e-8
rtol = 1e-8


# read a pkl file
def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data



file_path = "assignment3\data\group4\estimated_rso_catalog.pkl"
data = read_pkl(file_path)



keys = data[0].keys()
results_tca = []
results_states = []

for key in keys:
 # Initial time and state vectors
    # t0 = (datetime(2024, 3, 23, 5, 30, 0) - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
    dict = data[0][key]
    t0 = dict['UTC']
    t0 = epoch_from_date_time_components(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second)
    TLE = ['1 36585U 10022A   24079.80162258  .00000010  00000+0  00000+0 0  9995',
            '2 36585  54.4831 241.6070 0116206  61.5075 308.7360  2.00572189101153']
    GPS_TLE = environment.Tle(
    '1 36585U 10022A   24079.80162258  .00000010  00000+0  00000+0 0  9995',
    '2 36585  54.4831 241.6070 0116206  61.5075 308.7360  2.00572189101153'
)
    GPS_ephemeris = environment.TleEphemeris( "Earth", "J2000", GPS_TLE, False )
    sat_cart = GPS_ephemeris.cartesian_state( t0 )

    X1 = sat_cart
    
    X2 = dict['state']
    
    # Basic setup parameters
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    bodies = prop.tudat_initialize_bodies(bodies_to_create) 
    
    rso1_params = {}
    rso1_params['mass'] = 1630.
    rso1_params['area'] = 13.73259
    rso1_params['Cd'] = 2.2
    rso1_params['Cr'] = 1.3
    rso1_params['sph_deg'] = 8
    rso1_params['sph_ord'] = 8    
    rso1_params['central_bodies'] = ['Earth']
    rso1_params['bodies_to_create'] = bodies_to_create
    
    rso2_params = dict
    rso2_params['sph_deg'] = 8
    rso2_params['sph_ord'] = 8    
    rso2_params['central_bodies'] = ['Earth']
    rso2_params['bodies_to_create'] = bodies_to_create
    
    int_params = {}
    
    # Expected result
    TCA_true = 764445600.0  
    rho_true = 0.
    
    # Interval times
    tf = t0 + 3600*48
    trange = np.array([t0, tf])
    
    # RK4 test
    start = time.time()
    
    
    # RK78 test
    int_params['tudat_integrator'] = 'rkf78'
    int_params['step'] = 10.
    int_params['max_step'] = 1000.
    int_params['min_step'] = 1e-3
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    
    
    start = time.time()
    T_list, rho_list = compute_TCA(X1, X2, trange, rso1_params, rso2_params, 
                                   int_params, bodies,rho_min_crit=1000)
    

    print('')
    print('RK78 TCA unit test runtime [seconds]:', time.time() - start)
    print('RK78 TCA error [seconds]:', T_list[0]-TCA_true)
    print('RK78 miss distance error [m]:', rho_list[0]-rho_true)
    Po = dict['covar']
    tf, Xf, Pf = propagate_state_and_covar(X2, Po, trange, rso2_params, int_params)
    results_tca.append({
        'Key': key,
        'Runtime': time.time() - start,
        'TCA ': T_list[0] - TCA_true,
        'Miss Distance': rho_list[0] - rho_true
    })
    results_states.append({
        'Key': key,
        'time_array':tf,
        'state_array ': Xf,
        'covariance_array': Pf
    })
df_tca = pd.DataFrame(results_tca)
df_states = pd.DataFrame(results_states)
# Choose a file path for your CSV file
csv_file_path_tca = "results_tca.csv"
csv_file_path_states = "results_states.csv"
# Save the DataFrame to a CSV file
df_tca.to_csv(csv_file_path_tca, index=False)
df_states.to_csv(csv_file_path_states, index=False)

