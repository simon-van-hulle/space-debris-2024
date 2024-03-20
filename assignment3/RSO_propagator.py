import pickle as pkl
import os
import datetime
from tudatpy import constants
from tudatpy.astro.time_conversion import epoch_from_date_time_components
from TudatPropagator import propagate_state_and_covar

# print current directory
print(os.getcwd())

# constants
delta_t_days = 0.1
spherical_harmonic_degree = 8
spherical_harmonic_order = 8
central_bodies = ['Sun']
bodies_to_create = ['Earth', 'Moon', 'Sun']
tudat_integrator = 'rk4'
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



file_path = "data/group4/estimated_rso_catalog.pkl"
data = read_pkl(file_path)



keys = data[0].keys()

for key in keys:
    dict = data[0][key]
    Xo = dict['state']
    Po = dict['covar']
    t0 = dict['UTC']

    # convert datetime to seconds since J2000
    t0 = epoch_from_date_time_components(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second)

    # final epoch is 1 day after the initial epoch
    tf = t0 + delta_t_days * constants.JULIAN_DAY

    t_vec = [t0, tf]

    state_params = dict

    # add spherical degree and order to the state_params
    state_params['sph_deg'] = spherical_harmonic_degree
    state_params['sph_ord'] = spherical_harmonic_order

    # add the central body and bodies to create to the state_params
    state_params['central_bodies'] = central_bodies
    state_params['bodies_to_create'] = bodies_to_create

    int_params = {}
    int_params['tudat_integrator'] = tudat_integrator
    int_params['step'] = step
    int_params['max step'] = max_step
    int_params['min step'] = min_step
    int_params['atol'] = atol
    int_params['rtol'] = rtol

    tf, Xf, Pf = propagate_state_and_covar(Xo, Po, t_vec, state_params, int_params)

    print(tf, Xf)

