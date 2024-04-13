import pickle
import numpy as np
import TudatPropagator
import EstimationUtilities as EstUtil
from tudatpy.astro.time_conversion import DateTime
import time

# import scienceplots

import cProfile
from EstimationUtilities import compute_measurement
import pstats



with open('data/group4/q1_meas_objchar_91861.pkl', 'rb') as f:
    data = pickle.load(f)

state_info = data[0]
# dict_keys(['UTC', 'state', 'covar', 'mass', 'area', 'Cd', 'Cr', 'sph_deg', 'sph_ord', 'central_bodies',
# 'bodies_to_create'])
UTC = state_info['UTC']
state = state_info['state']
covar = state_info['covar']
state_params = {key: state_info[key] for key in ['mass', 'area', 'Cd', 'Cr', 'sph_deg', 'sph_ord', 'central_bodies',
                                                 'bodies_to_create']}

optical_sensor_params = data[1]
# dict_keys(['sensor_itrf', 'el_lim', 'az_lim', 'rg_lim', 'FOV_hlim', 'FOV_vlim', 'sun_elmask', 'meas_types',
# 'sigma_dict', 'eop_alldata', 'XYs_df'])

times_and_ra_dec = data[2]
# keys: 'tk_list', 'Yk_list'
tk_list = times_and_ra_dec['tk_list']  # times in seconds from J2000
Yk_list = times_and_ra_dec['Yk_list']  # right ascension and declination at all times
Yk_arr = np.squeeze(np.array(Yk_list))  # so it's 2d 420x2, before it was 3d 420x2x1

# initialize stuff for tudat propagator

# insert initial time on top of tk_list to make the propagation start at the time in UTC. the other values in tk_list
# will be the ones for which the states are saved to compute the rms of the residuals.
t_vec_prop = tk_list[:]
t_vec_prop.insert(0, DateTime(UTC.year, UTC.month, UTC.day, UTC.hour).epoch())

bodies = TudatPropagator.tudat_initialize_bodies()

# integrator parameters to have a fixed step size of 10 seconds for rk78
int_params_rk78_fixed = {'tudat_integrator': 'rkf78', 'step': 10, 'min_step': 10, 'max_step': 10, 'rtol': np.inf,
                         'atol': np.inf}
int_params_rk4_fixed = {'tudat_integrator': 'rk4', 'step': 10}

# THIS COMMENTED BELOW works for one propagation with the propagate_orbit, so no ukf
# # propagation, save only states and times needed for the residuals
# t_out, X_out = TudatPropagator.propagate_orbit_mod_output(state, t_vec_prop, state_params, int_params_rk4_fixed)
#
# # preallocate
# predicted_observations = np.zeros((len(tk_list), 2))
# # compute predicted observations
# for i in range(len(tk_list)):
#     predicted_observations[i, :] = EstUtil.compute_measurement(tk_list[i], X_out[i, :], optical_sensor_params).flatten()
#
# # true angular difference (arc between predicted and observed position) with spherical trig (cosines law)
# angular_difference = np.cos(np.pi - predicted_observations[:, 1]) * np.cos(np.pi - Yk_arr[:, 1]) + np.sin(np.pi - predicted_observations[:, 1]) * np.sin(np.pi - Yk_arr[:, 1]) * np.cos(predicted_observations[:, 0] - Yk_arr[:, 0])
# angular_difference = np.arccos(angular_difference)
# # rms
# rms_angular_difference = np.sqrt(np.mean(angular_difference**2))


# code for n iterations and see behaviour of rms when varying gamma = Cr * A / m (here mimicked by varying area and
# keeping constant the others)
# keep the same until right before propagation
# initialize outside once, then it gets overwritten every time
# predicted_observations = np.zeros((len(tk_list), 2))
# number_of_iterations = 10
# gamma = state_params['Cr'] * state_params['area'] / state_params['mass']
# area_array = np.linspace(94, 106, number_of_iterations)
# rms_array = np.zeros(number_of_iterations)
# for area in area_array:
#     state_params['area'] = area
#     t_out, X_out = TudatPropagator.propagate_orbit_mod_output(state, t_vec_prop, state_params, int_params_rk4_fixed)
#     for i in range(len(tk_list)):
#         predicted_observations[i, :] = EstUtil.compute_measurement(tk_list[i], X_out[i, :],
#                                                                    optical_sensor_params).flatten()
#     angular_difference = np.cos(np.pi - predicted_observations[:, 1]) * np.cos(np.pi - Yk_arr[:, 1]) + np.sin(
#         np.pi - predicted_observations[:, 1]) * np.sin(np.pi - Yk_arr[:, 1]) * np.cos(
#         predicted_observations[:, 0] - Yk_arr[:, 0])
#     angular_difference = np.arccos(angular_difference)
#     rms_array[np.where(area_array == area)[0][0]] = np.sqrt(np.mean(angular_difference ** 2))
#
# print(rms_array)

# gradient descent method in one variable: basically compute first derivative and move in that direction with a constant
# step size multiplied by the derivative. here the derivative is computed numerically with a central difference scheme
area_initial_guess = 95
print('area_initial_guess: ' + str(area_initial_guess))
predicted_observations = np.zeros((len(tk_list), 2))
iteration = 0
area_current_guess = area_initial_guess
while iteration < 10:
    # compute derivative
    time_pre = time.time()
    state_params['area'] = area_current_guess * 1.001
    t_out, X_out_right = TudatPropagator.propagate_orbit_mod_output(state, t_vec_prop, state_params,
                                                                    int_params_rk4_fixed)
    for i in range(len(tk_list)):
        predicted_observations[i, :] = EstUtil.compute_measurement(tk_list[i], X_out_right[i, :],
                                                                   optical_sensor_params).flatten()
    angular_difference_right = np.arccos(np.cos(np.pi - predicted_observations[:, 1]) * np.cos(np.pi - Yk_arr[:, 1]) +
                                         np.sin(np.pi - predicted_observations[:, 1]) * np.sin(np.pi - Yk_arr[:, 1]) *
                                         np.cos(predicted_observations[:, 0] - Yk_arr[:, 0]))
    rms_right = np.sqrt(np.mean(angular_difference_right ** 2))

    state_params['area'] = area_current_guess * 0.999
    t_out, X_out_left = TudatPropagator.propagate_orbit_mod_output(state, t_vec_prop, state_params,
                                                                   int_params_rk4_fixed)
    time_pre_measurement = time.time()
    for i in range(len(tk_list)):
        predicted_observations[i, :] = EstUtil.compute_measurement(tk_list[i], X_out_left[i, :],
                                                                   optical_sensor_params).flatten()
    time_post_measurement = time.time()
    print('measurement time: ' + str(time_post_measurement - time_pre_measurement))
    angular_difference_left = np.arccos(np.cos(np.pi - predicted_observations[:, 1]) * np.cos(np.pi - Yk_arr[:, 1]) +
                                        np.sin(np.pi - predicted_observations[:, 1]) * np.sin(np.pi - Yk_arr[:, 1]) *
                                        np.cos(predicted_observations[:, 0] - Yk_arr[:, 0]))
    rms_left = np.sqrt(np.mean(angular_difference_left ** 2))

    derivative = (rms_right - rms_left) / (2 * 0.001 * area_current_guess)
    area_new_guess = area_current_guess - 1e6 * derivative
    time_post = time.time()
    print('time of one iteration: ' + str(time_post - time_pre))
    # check if deviation from previous area value is significant
    print('new value: ' + str(area_new_guess))
    deviation = np.abs(area_new_guess - area_current_guess) / area_current_guess
    if deviation < 0.001:
        break
    iteration +=1
    area_current_guess = area_new_guess


# Define the function first
def function_to_profile():
    # Call your function here with the necessary arguments
    t_out, X_out_right = TudatPropagator.propagate_orbit_mod_output(state, t_vec_prop, state_params,
                                                                    int_params_rk4_fixed)
    for i in range(len(tk_list)):
        predicted_observations[i, :] = EstUtil.compute_measurement(tk_list[i], X_out_right[i, :],
                                                                   optical_sensor_params).flatten()

# Then, you can profile it
predicted_observations = np.zeros((len(tk_list), 2))
cProfile.run('function_to_profile()', 'outputfile.prof')
p = pstats.Stats('outputfile.prof')
p.sort_stats('cumulative').print_stats(10)  # This will print the top 10 functions that took the most time