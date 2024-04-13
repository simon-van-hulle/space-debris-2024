import pickle
import numpy as np
# import scienceplots
import TudatPropagator
import EstimationUtilities as EstUtil
from tudatpy.astro.time_conversion import DateTime
import time


with open('data/group4/q1_meas_objchar_91861.pkl', 'rb') as f:
    data = pickle.load(f)

# print(data)
state_info = data[0]
# dict_keys(['UTC', 'state', 'covar', 'mass', 'area', 'Cd', 'Cr', 'sph_deg', 'sph_ord', 'central_bodies',
# 'bodies_to_create'])
UTC = state_info['UTC']
state = state_info['state']
covar = state_info['covar']
mass = state_info['mass']
area = state_info['area']
Cd = state_info['Cd']
Cr = state_info['Cr']
sph_deg = state_info['sph_deg']
sph_ord = state_info['sph_ord']
central_bodies = state_info['central_bodies']
bodies_to_create = state_info['bodies_to_create']

optical_sensor_params = data[1]       # eop is earth orientation parameters, there are all the information from 1962 to
# today
# dict_keys(['sensor_itrf', 'el_lim', 'az_lim', 'rg_lim', 'FOV_hlim', 'FOV_vlim', 'sun_elmask', 'meas_types',
# 'sigma_dict', 'eop_alldata', 'XYs_df'])
sensor_itrf = optical_sensor_params['sensor_itrf']
el_lim = optical_sensor_params['el_lim']
az_lim = optical_sensor_params['az_lim']
rg_lim = optical_sensor_params['rg_lim']
FOV_hlim = optical_sensor_params['FOV_hlim']
FOV_vlim = optical_sensor_params['FOV_vlim']
sun_elmask = optical_sensor_params['sun_elmask']
meas_types = optical_sensor_params['meas_types']
sigma_dict = optical_sensor_params['sigma_dict']
eop_alldata = optical_sensor_params['eop_alldata']
XYs_df = optical_sensor_params['XYs_df']

times_and_ra_dec = data[2]
# keys: 'tk_list', 'Yk_list'
tk_list = times_and_ra_dec['tk_list']       # times in seconds from J2000
Yk_list = times_and_ra_dec['Yk_list']       # right ascension and declination at all times

Yk_arr = np.squeeze(np.array(Yk_list))      # so it's 2d 420x2, before it was 3d 420x2x1


t_vec_prop = tk_list[:]
t_vec_prop.insert(0, DateTime(2024, 3, 21, 12).epoch())
bodies = TudatPropagator.tudat_initialize_bodies()
state_params = {key: state_info[key] for key in ['mass', 'area', 'Cd', 'Cr', 'sph_deg', 'sph_ord', 'central_bodies',
                                                 'bodies_to_create']}
# integrator parameters to have a fixed step size of 10 seconds for rk78
int_params_rk78_fixed = {'tudat_integrator': 'rkf78', 'step': 10, 'min_step': 10, 'max_step': 10, 'rtol': np.inf, 'atol': np.inf}
int_params_rk4_fixed = {'tudat_integrator': 'rk4', 'step': 10}
# time_pre = time.time()
# tout_rk78, Xout_rk78, Pf_rk78 = TudatPropagator.propagate_state_and_covar_original(state, covar, t_vec_prop, state_params,
#                                                                     int_params_rk78_fixed)
#
# # this is for the propagation with the UKF, first let's do it normally without any uncertainty
# time_post = time.time()
# print('time rk78 fixed step size: ' + str(time_post - time_pre))
# time_pre = time.time()
# tout_rk4, Xout_rk4, Pf_rk4 = TudatPropagator.propagate_state_and_covar_original(state, covar, t_vec_prop, state_params,
#                                                                     int_params_rk4_fixed)
# time_post = time.time()
# print('time rk4 fixed step size: ' + str(time_post - time_pre))
# error = np.sum(np.linalg.norm(Xout_rk78 - Xout_rk4, axis=1))
# print(error)


time_pre = time.time()
t_out, X_out = TudatPropagator.propagate_orbit_mod_output(state, t_vec_prop, state_params, int_params_rk4_fixed)
time_post = time.time()
print('time rk4 fixed step size: ' + str(time_post - time_pre))
# now the epochs in t_out coincide with the epochs in tk_list
print(X_out[-1, :])
time_pre = time.time()
t_out_rk78, X_out_rk78 = TudatPropagator.propagate_orbit_mod_output(state, t_vec_prop, state_params,
                                                                    int_params_rk78_fixed)
time_post = time.time()
print('time rk78 fixed step size: ' + str(time_post - time_pre))
print(X_out_rk78[-1, :])
print(X_out_rk78[-1, :] - X_out[-1, :])


# tout_rk4, Xout_rk4 have 7390 rows bc each of them is a timestep. i'm interested only in the last 420 (len(tk_list))
# so i will take the last 420 rows and calculate the rms from those. Xout_rk4 has 6 columns
predicted_observations = np.zeros((len(tk_list), 2))
# t_out = tout_rk4[-len(tk_list):]
# X_out = Xout_rk4[-len(tk_list):, :]
for i in range(len(tk_list)):
    predicted_observations[i, :] = EstUtil.compute_measurement(tk_list[i], X_out[i, :], optical_sensor_params).flatten()

# now i have the predicted observations, i can compare them with the real observations
# Yk_arr for the real observations
# calculate the rms of the residuals between the predicted and the real observations

# true angular difference (arc between predicted and observed position)
angular_difference = np.cos(np.pi - predicted_observations[:, 1]) * np.cos(np.pi - Yk_arr[:, 1]) + np.sin(np.pi - predicted_observations[:, 1]) * np.sin(np.pi - Yk_arr[:, 1]) * np.cos(predicted_observations[:, 0] - Yk_arr[:, 0])
angular_difference = np.arccos(angular_difference)
# rms
rms_angular_difference = np.sqrt(np.mean(angular_difference**2))
print(rms_angular_difference)

tout_ukf, Xout_ukf, Pf_ukf = TudatPropagator.propagate_state_and_covar_original(state, covar, t_vec_prop, state_params,
                                                                                int_params_rk4_fixed)
print(Xout_ukf)
print(Xout_ukf.flatten()-X_out[-1, :])

# # unwrapping
# difference = Yk_arr - predicted_observations
# difference_unwrapped = np.copy(difference)
# difference_unwrapped[312:360, 0] += 2*np.pi
#
# rms_ra = np.sqrt(np.mean(difference_unwrapped[:, 0]**2))
# rms_dec = np.sqrt(np.mean(difference_unwrapped[:, 1]**2))
# rms_ra_and_dec = np.sqrt(np.mean(difference_unwrapped**2))
# print(rms_ra, rms_dec, rms_ra_and_dec)


