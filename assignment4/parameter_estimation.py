import pickle
import numpy as np
import TudatPropagator
import EstimationUtilities as EstUtil
from tudatpy.astro.time_conversion import DateTime
import time
import os
import sys
sys.path.append(r'..\assignment3')
import ukf_tuning
import matplotlib.pyplot as plt
import matplotlib
import scienceplots


# import scienceplots

import cProfile
from EstimationUtilities import compute_measurement
import pstats


def get_inertial_to_ric_matrix(state_ref):
    pos = state_ref[:3]
    vel = state_ref[3:]

    u_r = (pos / np.linalg.norm(pos))
    angular_momentum = np.cross(pos, vel)
    u_c = angular_momentum / np.linalg.norm(angular_momentum)
    u_i = np.cross(u_c, u_r)
    #
    return np.vstack((u_r, u_i, u_c)).T     # transpose so the columns of the matrix are the unit vectors of the new
                                            # (RIC) frame


# Set the font family and size to use for Matplotlib figures.
# plt.style.use('science')
matplotlib.rcParams.update({'font.size': 14, 'font.family': 'serif'})

#%% data unpacking
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

#%% initialize stuff for tudat propagator

# insert initial time on top of tk_list to make the propagation start at the time in UTC. the other values in tk_list
# will be the ones for which the states are saved to compute the rms of the residuals.
t_vec_prop = tk_list[:]
t_vec_prop.insert(0, DateTime(UTC.year, UTC.month, UTC.day, UTC.hour).epoch())

bodies = TudatPropagator.tudat_initialize_bodies()

# integrator parameters to have a fixed step size of 10 seconds for rk78
int_params_rk78_fixed = {'tudat_integrator': 'rkf78', 'step': 10, 'min_step': 10, 'max_step': 10, 'rtol': np.inf,
                         'atol': np.inf}
int_params_rk4_fixed = {'tudat_integrator': 'rk4', 'step': 10}

#%% one propagation with the propagate_orbit, so no ukf
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


#%% code for n iterations and see behaviour of rms when varying gamma = Cr * A / m (here mimicked by varying area and
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


#%% gradient descent method in one variable: basically compute first derivative and move in that direction with a
# constant step size multiplied by the derivative. here the derivative is computed numerically with a central difference
# scheme

# area_initial_guess = 100
# print('area_initial_guess: ' + str(area_initial_guess))
# predicted_observations = np.zeros((len(tk_list), 2))
# iteration = 0
# area_current_guess = area_initial_guess
# while iteration < 20:
#     # compute derivative
#     time_pre = time.time()
#     state_params['area'] = area_current_guess * 1.001
#     t_out, X_out_right = TudatPropagator.propagate_orbit_mod_output(state, t_vec_prop, state_params,
#                                                                     int_params_rk4_fixed)
#     for i in range(len(tk_list)):
#         predicted_observations[i, :] = EstUtil.compute_measurement(tk_list[i], X_out_right[i, :],
#                                                                    optical_sensor_params).flatten()
#     angular_difference_right = np.arccos(np.cos(np.pi - predicted_observations[:, 1]) * np.cos(np.pi - Yk_arr[:, 1]) +
#                                          np.sin(np.pi - predicted_observations[:, 1]) * np.sin(np.pi - Yk_arr[:, 1]) *
#                                          np.cos(predicted_observations[:, 0] - Yk_arr[:, 0]))
#     rms_right = np.sqrt(np.mean(angular_difference_right ** 2))
#
#     state_params['area'] = area_current_guess * 0.999
#     t_out, X_out_left = TudatPropagator.propagate_orbit_mod_output(state, t_vec_prop, state_params,
#                                                                    int_params_rk4_fixed)
#     time_pre_measurement = time.time()
#     for i in range(len(tk_list)):
#         predicted_observations[i, :] = EstUtil.compute_measurement(tk_list[i], X_out_left[i, :],
#                                                                    optical_sensor_params).flatten()
#     time_post_measurement = time.time()
#     print('measurement time: ' + str(time_post_measurement - time_pre_measurement))
#     angular_difference_left = np.arccos(np.cos(np.pi - predicted_observations[:, 1]) * np.cos(np.pi - Yk_arr[:, 1]) +
#                                         np.sin(np.pi - predicted_observations[:, 1]) * np.sin(np.pi - Yk_arr[:, 1]) *
#                                         np.cos(predicted_observations[:, 0] - Yk_arr[:, 0]))
#     rms_left = np.sqrt(np.mean(angular_difference_left ** 2))
#
#     derivative = (rms_right - rms_left) / (2 * 0.001 * area_current_guess)
#     area_new_guess = area_current_guess - 1e6 * derivative
#     time_post = time.time()
#     print('time of one iteration: ' + str(time_post - time_pre))
#     # check if deviation from previous area value is significant
#     print('new value: ' + str(area_new_guess))
#     deviation = np.abs(area_new_guess - area_current_guess) / area_current_guess
#     if deviation < 0.000001:
#         break
#     iteration +=1
#     area_current_guess = area_new_guess


#%% code for Monte Carlo simulation

# # set up the montecarlo simulation
# std_dev = optical_sensor_params['sigma_dict']['ra']
# area_initial_guess = 100.1
# predicted_observations = np.zeros((len(tk_list), 2))
# how_many_mc = 200
# area_final = np.zeros(how_many_mc)
# time_mc_pre = time.time()
# for montecarlo in range(how_many_mc):
#     Yk_arr_mc = np.random.normal(Yk_arr, std_dev)
#     area_current_guess = area_initial_guess
#     iteration = 0
#     while iteration < 10:
#         # compute derivative
#         state_params['area'] = area_current_guess * 1.001
#         t_out, X_out_right = TudatPropagator.propagate_orbit_mod_output(state, t_vec_prop, state_params,
#                                                                         int_params_rk4_fixed)
#         for i in range(len(tk_list)):
#             predicted_observations[i, :] = EstUtil.compute_measurement(tk_list[i], X_out_right[i, :],
#                                                                        optical_sensor_params).flatten()
#         angular_difference_right = np.arccos(
#             np.cos(np.pi - predicted_observations[:, 1]) * np.cos(np.pi - Yk_arr_mc[:, 1]) +
#             np.sin(np.pi - predicted_observations[:, 1]) * np.sin(np.pi - Yk_arr_mc[:, 1]) *
#             np.cos(predicted_observations[:, 0] - Yk_arr_mc[:, 0]))
#         rms_right = np.sqrt(np.mean(angular_difference_right ** 2))
#
#         state_params['area'] = area_current_guess * 0.999
#         t_out, X_out_left = TudatPropagator.propagate_orbit_mod_output(state, t_vec_prop, state_params,
#                                                                        int_params_rk4_fixed)
#         for i in range(len(tk_list)):
#             predicted_observations[i, :] = EstUtil.compute_measurement(tk_list[i], X_out_left[i, :],
#                                                                        optical_sensor_params).flatten()
#         angular_difference_left = np.arccos(
#             np.cos(np.pi - predicted_observations[:, 1]) * np.cos(np.pi - Yk_arr_mc[:, 1]) +
#             np.sin(np.pi - predicted_observations[:, 1]) * np.sin(np.pi - Yk_arr_mc[:, 1]) *
#             np.cos(predicted_observations[:, 0] - Yk_arr_mc[:, 0]))
#         rms_left = np.sqrt(np.mean(angular_difference_left ** 2))
#
#         derivative = (rms_right - rms_left) / (2 * 0.001 * area_current_guess)
#         area_new_guess = area_current_guess - 1e6 * derivative
#         deviation = np.abs(area_new_guess - area_current_guess) / area_current_guess
#         if deviation < 0.001:
#             break
#         area_current_guess = area_new_guess
#         iteration += 1
#
#     area_final[montecarlo] = area_new_guess
#     print('montecarlo iteration: ' + str(montecarlo))
# time_mc_post = time.time()
# print('time of montecarlo simulation: ' + str(time_mc_post - time_mc_pre))
#
# print(area_final)
# np.savetxt('area_final.dat', area_final)

# #%% run UKF
settings = ukf_tuning.UkfSettings()
intsettings = settings.get_int_params()
filter_params = settings.get_filter_params()
bodies = ukf_tuning.tudat_initialize_bodies()

directory_path = r'..\assignment4\data\group4'
filename = 'q1_meas_objchar_91861.pkl'
filepath = os.path.join(directory_path, filename)
state_params, meas_dict, sensor_params = EstUtil.read_measurement_file(filepath)
area_final_value = 100.18395
state_params['area'] = area_final_value
ukf_result = EstUtil.ukf(state_params, meas_dict, sensor_params, intsettings, filter_params, bodies)
print(ukf_result)

# plot the post fit residuals
tk_arr = np.array(tk_list)
residuals_ra_dec = np.zeros((len(tk_list), 2))
i = 0
for key in ukf_result.keys():
    residuals_ra_dec[i, :] = ukf_result[key]['resids'].flatten()
    i += 1

residual = np.arccos( np.cos(residuals_ra_dec[:, 0] * np.cos(residuals_ra_dec[:, 1])) )

initial_epoch = DateTime(state_params['UTC'].year, state_params['UTC'].month, state_params['UTC'].day, state_params['UTC'].hour).epoch()

plt.plot((tk_arr - initial_epoch)/3600, residual)
plt.xlabel('time after initial state [hours]')
plt.ylabel('residual [rad]')
plt.yscale('log')
plt.title('Post fit angular residuals')

dir_path = r"C:\Users\uliul\OneDrive\Documenti\Ulisse\Delft\TU Delft\Q3\Space debris\simon shared\assignment4\plot"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
plt.savefig(os.path.join(dir_path, "Post_fit_residuals_q1.png"))

plt.show()

rms_residual = np.sqrt(np.mean(residual**2))
print('rms residual: ' + str(rms_residual))

i = 0
# get covariance matrix in RIC from ECI and extract std dev in RIC to plot it
std_dev_ric = np.zeros((len(tk_list), 3))
for key in ukf_result.keys():
    covar_eci = ukf_result[key]['covar']
    eci_to_ric_3 = get_inertial_to_ric_matrix(ukf_result[key]['state'].flatten())
    eci_to_ric_6 = np.block([[eci_to_ric_3, np.zeros((3, 3))], [np.zeros((3, 3)), eci_to_ric_3]])
    covar_ric = np.dot(np.dot(eci_to_ric_6, covar_eci), eci_to_ric_6.T)
    std_dev_ric[i, :] = np.sqrt(np.diag(covar_ric[:3, :3]))
    i += 1

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(tk_arr - initial_epoch, std_dev_ric[:, 0])
ax[0].set_ylabel('std dev R [m]')
ax[1].plot(tk_arr - initial_epoch, std_dev_ric[:, 1])
ax[1].set_ylabel('std dev I [m]')
ax[2].plot(tk_arr - initial_epoch, std_dev_ric[:, 2])
ax[2].set_ylabel('std dev C [m]')
plt.xlabel('time after initial state [s]')
plt.suptitle('Standard deviation in RIC frame')

dir_path = r"C:\Users\uliul\OneDrive\Documenti\Ulisse\Delft\TU Delft\Q3\Space debris\simon shared\assignment4\plot"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
plt.savefig(os.path.join(dir_path, "Std_dev_q1.png"))

plt.show()

# plot 3sigma bounds
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].fill_between((tk_arr - initial_epoch)/3600, -3*std_dev_ric[:, 0], 3*std_dev_ric[:, 0], alpha=0.5)
ax[0].set_ylabel('std dev R [m]')
ax[1].fill_between((tk_arr - initial_epoch)/3600, -3*std_dev_ric[:, 1], 3*std_dev_ric[:, 1], alpha=0.5)
ax[1].set_ylabel('std dev I [m]')
ax[2].fill_between((tk_arr - initial_epoch)/3600, -3*std_dev_ric[:, 2], 3*std_dev_ric[:, 2], alpha=0.5)
ax[2].set_ylabel('std dev C [m]')
plt.xlabel('time after initial state [hours]')
plt.suptitle('3 $\sigma$ covariance bounds in RIC frame')
fig.tight_layout()
dir_path = r"C:\Users\uliul\OneDrive\Documenti\Ulisse\Delft\TU Delft\Q3\Space debris\simon shared\assignment4\plot"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
plt.savefig(os.path.join(dir_path, "3sigma_bounds_ric.png"))

plt.show()





#%% function profiling
# # Define the function first
# def function_to_profile():
#     # Call your function here with the necessary arguments
#     t_out, X_out_right = TudatPropagator.propagate_orbit_mod_output(state, t_vec_prop, state_params,
#                                                                     int_params_rk4_fixed)
#     for i in range(len(tk_list)):
#         predicted_observations[i, :] = EstUtil.compute_measurement(tk_list[i], X_out_right[i, :],
#                                                                    optical_sensor_params).flatten()
#
# # Then, you can profile it
# predicted_observations = np.zeros((len(tk_list), 2))
# cProfile.run('function_to_profile()', 'outputfile.prof')
# p = pstats.Stats('outputfile.prof')
# p.sort_stats('cumulative').print_stats(10)  # This will print the top 10 functions that took the most time