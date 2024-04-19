import matplotlib
import scienceplots
import numpy as np
import matplotlib.pyplot as plt
import TudatPropagator
import json
import pickle
import os
import pandas as pd
import sys
import time
from scipy.optimize import minimize
import EstimationUtilities as EstUtil
from tudatpy.astro.time_conversion import DateTime
sys.path.append('assignment3')
import EstimationUtilities
import ukf_tuning


settings = ukf_tuning.UkfSettings()
intsettings = settings.get_int_params()
filter_params = settings.get_filter_params()
bodies = TudatPropagator.tudat_initialize_bodies()

# integrator parameters to have a fixed step size of 10 seconds for rk78
int_params_rk78_fixed = {'tudat_integrator': 'rkf78', 'step': 10, 'min_step': 10, 'max_step': 10, 'rtol': np.inf,
                        'atol': np.inf}
int_params_rk4_fixed = {'tudat_integrator': 'rk4', 'step': 10}

def get_inertial_to_ric_matrix(state_ref):
    pos = state_ref[:3]
    vel = state_ref[3:]

    u_r = (pos / np.linalg.norm(pos))
    angular_momentum = np.cross(pos, vel)
    u_c = angular_momentum / np.linalg.norm(angular_momentum)
    u_i = np.cross(u_c, u_r)
    return np.vstack((u_r, u_i, u_c)).T  


file_path = 'assignment3\group4_sensor_tasking_file.json'
# Open the JSON file and load its contents into a Python dictionary
with open(file_path, 'r') as file:
    data = json.load(file)


# Replace 'your_directory_path' with the actual directory path that contains your .pkl files
directory_path = 'assignment4\data\group4'

# Data structure to store all the returned values from each file
all_data = []


# and filename != 'q1_meas_objchar_91861.pkl'
# Loop through each file in the directory and process it with the provided function
for filename in os.listdir(directory_path):
    if filename.endswith('.pkl') and filename != 'q1_meas_objchar_91861.pkl' and filename != 'q3_meas_iod_99004.pkl' and filename != 'q3_meas_rso_99004.pkl'   and filename != 'q2_meas_maneuver_91104.pkl' and filename!= 'q3_meas_rso_99004':
        filepath = os.path.join(directory_path, filename)
        state_params, meas_dict, sensor_params = EstimationUtilities.read_measurement_file(filepath)
        all_data.append({
            'filename': filename,
            'state_params': state_params,
            'meas_dict': meas_dict,
            'sensor_params': sensor_params
        })

print('reeeeee')



areas = []
for i in range(len(all_data)):
    tk_list = all_data[i]['meas_dict']['tk_list']
    Yk_list = all_data[i]['meas_dict']['Yk_list']
    t_vec_prop = tk_list[:]
    UTC = all_data[i]['state_params']['UTC']
    t_vec_prop.insert(0, DateTime(UTC.year, UTC.month, UTC.day, UTC.hour).epoch())
    optical_sensor_params = all_data[i]['sensor_params']
    Yk_arr = np.squeeze(np.array(Yk_list))  # so it's 2d 420x2, before it was 3d 420x2x1
    state = all_data[i]['state_params']['state']
    state_params = all_data[i]['state_params']
    # state_params = {key: state[key] for key in ['mass', 'area', 'Cd', 'Cr', 'sph_deg', 'sph_ord', 'central_bodies',
    #                                              'bodies_to_create']}
    area_initial_guess = all_data[i]['state_params']['area']
    predicted_observations = np.zeros((len(tk_list), 2))

    def callback(x):
       print("Current estimate:", x)
    #    print("Current RMS:", rms_error(x, state, t_vec_prop, state_params, optical_sensor_params, tk_list, Yk_arr))
    
    def rms_error(area, state, t_vec_prop, state_params, optical_sensor_params, tk_list, Yk_arr):
        state_params['area'] = area
        _, X_out = TudatPropagator.propagate_orbit_mod_output(state, t_vec_prop, state_params,int_params_rk4_fixed)
        predicted_observations = np.array([EstUtil.compute_measurement(tk, X_out[j, :], optical_sensor_params).flatten()
                                        for j, tk in enumerate(tk_list)])
        
        angular_difference = np.arccos(np.cos(np.pi - predicted_observations[:, 1]) * np.cos(np.pi - Yk_arr[:, 1]) +
                                    np.sin(np.pi - predicted_observations[:, 1]) * np.sin(np.pi - Yk_arr[:, 1]) *
                                    np.cos(predicted_observations[:, 0] - Yk_arr[:, 0]))
        return np.sqrt(np.mean(angular_difference ** 2))

    def optimize_area(initial_area, state, t_vec_prop, state_params, optical_sensor_params, tk_list, Yk_arr):
        result = minimize(rms_error, x0=[initial_area], args=(state, t_vec_prop, state_params, optical_sensor_params, tk_list, Yk_arr),
                        method='Nelder-Mead', options={'xatol': 1e-6, 'disp': True},callback=callback)
        return result.x[0]
    # optimized_area = optimize_area(100, state, t_vec_prop, state_params, optical_sensor_params, tk_list, Yk_arr)
    # print('Optimized Area:', optimized_area)
    def optimize_area_with_bounds(initial_area, state, t_vec_prop, state_params, optical_sensor_params, tk_list, Yk_arr):
        bounds = [(1e-3, None)]  # enforce the area to be non-negative
        result = minimize(rms_error, x0=[initial_area], args=(state, t_vec_prop, state_params, optical_sensor_params, tk_list, Yk_arr),
                        method='TNC', bounds=bounds, options={'disp': True})
        return result.x[0]

    # Example usage:
    # optimized_area = optimize_area_with_bounds(100, state, t_vec_prop, state_params, optical_sensor_params, tk_list, Yk_arr)
    # print('Optimized Area:', optimized_area)
    # areas.append(optimized_area)
# print(areas)


# Data structure to store all the returned values from each file
ukf_results = []
# print(areas)
# Loop through each file in the directory and process it with the provided function
k = 0
for filename in os.listdir(directory_path):
    if filename.endswith('.pkl') and filename != 'q3_meas_iod_99004.pkl' and filename != 'q3_meas_rso_99004.pkl' and filename != 'q1_meas_objchar_91861.pkl' and filename != 'q2_meas_maneuver_91104.pkl' and filename!= 'q3_meas_rso_99004':
        print(filename)
        
        filepath = os.path.join(directory_path, filename)
        
        # Read the measurement file
        state_params, meas_dict, sensor_params = EstimationUtilities.read_measurement_file(filepath)
        # state_params['area'] = areas[k]
        # Now run the Unscented Kalman Filter with the parameters
        # filter_params['area'] = areas[k]
        if k ==0 or k==9:
            state_params['covar'] = state_params['covar']*2.5
            filter_params
            print(k)
        ukf_result = EstimationUtilities.ukf(state_params, meas_dict, sensor_params, intsettings, filter_params, bodies, verbose=False)
        
        # Append the result of the UKF for this file to the results list
        ukf_results.append({
            'filename': filename,
            'ukf_result': ukf_result
        })
        k+=1
        # print(k)
# print(ukf_results)



residuals = []
residuals_rms = []
residuals_rms_ra = []
residuals_rms_dec = []
final_states = np.zeros((len(ukf_results), 6))
final_covars = np.zeros((len(ukf_results), 6, 6))
final_time = np.zeros((len(ukf_results), 1))

for i in range(len(ukf_results)):
    tk_list = all_data[i]['meas_dict']['tk_list']
    residuals_ra_dec = np.zeros((len(tk_list), 2))
    last_key = list(ukf_results[i]['ukf_result'].keys())[-1]
    final_states[i, :] = ukf_results[i]['ukf_result'][last_key]['state'].flatten()
    final_covars[i, :, :] = ukf_results[i]['ukf_result'][last_key]['covar']
    final_time[i] = last_key
    for x, key in enumerate(ukf_results[i]['ukf_result'].keys()):
        residuals_ra_dec[x, :] = (ukf_results[i]['ukf_result'][key]['resids'].flatten())
    residuals_temp = np.arccos( np.cos(residuals_ra_dec[:, 0] * np.cos(residuals_ra_dec[:, 1])) )
    residuals.append(residuals_temp)
    residuals_rms_ra.append(np.sqrt(np.mean(residuals_ra_dec[:, 0]**2)))
    residuals_rms_dec.append(np.sqrt(np.mean(residuals_ra_dec[:, 1]**2)))                            
    residuals_rms.append(np.sqrt(np.mean(residuals_temp**2)))
    
    
print(residuals_rms)
print(residuals_rms_ra)
print(residuals_rms_dec)
rmss ={
    'residuals_rms': residuals_rms,
    'residuals_rms_ra': residuals_rms_ra,
    'residuals_rms_dec': residuals_rms_dec
}

data = {
    'final_states': final_states,
    'final_covars': final_covars,
    'final_time': final_time
}

# Use a binary file to store your data
with open('new_states_measurements.pkl', 'wb') as f:
    pickle.dump(data, f)
with open('rms_results.pkl', 'wb') as f:
    pickle.dump(data, f)



for i in range(len(ukf_results)):
    plt.close()
    tk_list = all_data[i]['meas_dict']['tk_list']
    tk_arr = np.array(tk_list)
    UTC = all_data[i]['state_params']['UTC']
    initial_epoch = DateTime(UTC.year, UTC.month, UTC.day, UTC.hour).epoch()
    
    
    
    plt.plot((tk_arr - initial_epoch)/3600, residuals[i], 'o', label='residuals')
    plt.xlabel('time after initial state [hours]')
    plt.ylabel('residual [rad]')
    plt.yscale('log')
    plt.title('Post fit angular residuals')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(os.path.join(dir_path, "residuals_q4_"+str(ukf_results[i]['filename'])+"_test.png"))

    plt.close()


    std_dev_ric = np.zeros((len(tk_list), 3))
    for j, key in enumerate(ukf_results[i]['ukf_result'].keys()):
        covar_eci = ukf_results[i]['ukf_result'][key]['covar']
        eci_to_ric_3 = get_inertial_to_ric_matrix(ukf_results[i]['ukf_result'][key]['state'].flatten())
        eci_to_ric_6 = np.block([[eci_to_ric_3, np.zeros((3, 3))], [np.zeros((3, 3)), eci_to_ric_3]])
        covar_ric = np.dot(np.dot(eci_to_ric_6, covar_eci), eci_to_ric_6.T)
        std_dev_ric[j, :] = np.sqrt(np.diag(covar_ric[:3, :3]))




    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(tk_arr - initial_epoch, std_dev_ric[:, 0])
    ax[0].set_ylabel('std dev R [m]')
    ax[1].plot(tk_arr - initial_epoch, std_dev_ric[:, 1])
    ax[1].set_ylabel('std dev I [m]')
    ax[2].plot(tk_arr - initial_epoch, std_dev_ric[:, 2])
    ax[2].set_ylabel('std dev C [m]')
    plt.xlabel('time after initial state [s]')
    plt.suptitle('Standard deviation in RIC frame')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(os.path.join(dir_path, "Std_dev_q4_"+str(ukf_results[i]['filename'])+"_test.png"))
    plt.close()
    
    
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
    dir_path = r"assignment4\plot"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(os.path.join(dir_path, "3sigma_bounds_ric_"+str(ukf_results[i]['filename'])+"_test.png"))
    plt.close()
    std_dev_ra_dec = np.zeros((len(tk_list), 2))
    residuals_ra_dec = np.zeros((len(tk_list), 2))
    for j, key in enumerate(ukf_results[i]['ukf_result'].keys()):
        std_dev_ra_dec[j,0] = ukf_results[i]['ukf_result'][key]['sigma_dict_meas']['ra']
        std_dev_ra_dec[j,1] = ukf_results[i]['ukf_result'][key]['sigma_dict_meas']['dec']
        residuals_ra_dec[j, :] = (ukf_results[i]['ukf_result'][key]['resids'].flatten())
        
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(tk_arr - initial_epoch, 3*std_dev_ra_dec[:, 0])
    ax[0].plot(tk_arr - initial_epoch, -3*std_dev_ra_dec[:, 0])
    ax[0].plot(tk_arr - initial_epoch, residuals_ra_dec[:, 0], 'o', label='residuals')
    ax[0].set_ylabel('std dev RA [rad]')
    ax[1].plot(tk_arr - initial_epoch, 3*std_dev_ra_dec[:, 1])
    ax[1].plot(tk_arr - initial_epoch, residuals_ra_dec[:, 1], 'o', label='residuals')
    ax[1].plot(tk_arr - initial_epoch, -3*std_dev_ra_dec[:, 1])
    ax[1].set_ylabel('std dev DEC [rad]')
    
    plt.xlabel('Time after initial state [hours]')
    plt.suptitle('3 $\sigma$ covariance bounds for RA and DEC')
    fig.tight_layout()
    dir_path = r"assignment4\plot"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(os.path.join(dir_path, "3sigma_bounds_ra_de"+str(ukf_results[i]['filename'])+"_test.png"))
    plt.close()
    
    states = np.zeros((len(tk_list), 6))
    for j, key in enumerate(ukf_results[i]['ukf_result'].keys()):
        states[j, :] = ukf_results[i]['ukf_result'][key]['state'].flatten()    
    fig, ax = plt.subplots(6, 1, sharex=True)
    ax[0].plot(tk_arr - initial_epoch, states[:, 0])
    ax[0].set_ylabel('x [m]')
    ax[1].plot(tk_arr - initial_epoch, states[:, 1])
    ax[1].set_ylabel('y [m]')
    ax[2].plot(tk_arr - initial_epoch, states[:, 2])
    ax[2].set_ylabel('z [m]')
    ax[3].plot(tk_arr - initial_epoch, states[:, 3])
    ax[3].set_ylabel('vx [m/s]')
    ax[4].plot(tk_arr - initial_epoch, states[:, 4])
    ax[4].set_ylabel('vy [m/s]')
    ax[5].plot(tk_arr - initial_epoch, states[:, 5])
    ax[5].set_ylabel('vz [m/s]')
    plt.xlabel('time after initial state [s]')
    plt.suptitle('State evolution')
    fig.tight_layout()
    dir_path = r"assignment4\plot"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(os.path.join(dir_path, "state_evolution_"+str(ukf_results[i]['filename'])+"_test.png"))
        
    plt.close()
    
# print('UKF processing complete.')

