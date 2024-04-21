import matplotlib
import scienceplots
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import pickle
import os
import pandas as pd
import sys
import EstimationUtilities
import ukf_tuning
from datetime import datetime
from TudatPropagator import *
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.astro import time_conversion


settings = ukf_tuning.UkfSettings()
intsettings = settings.get_int_params()
filter_params = settings.get_filter_params()
bodies = ukf_tuning.tudat_initialize_bodies()

intsettings = {}
intsettings['tudat_integrator'] = "rk4"
intsettings['step'] = 10
# Set the font family and size to use for Matplotlib figures.

# Set the initial time
t0_seconds = time_conversion.epoch_from_date_time_components(2024, 3, 21, 12, 0, 0)


file_path = "data\group4\q2_meas_maneuver_91104.pkl"

# Read the measurement file
state_params, meas_dict, sensor_params = EstimationUtilities.read_measurement_file(file_path)

# # Now run the Unscented Kalman Filter with the parameters
# ukf_result = EstimationUtilities.ukf(state_params, meas_dict, sensor_params, intsettings, filter_params, bodies)

# # Save the UKF result to a pickle file
# with open('ukf_result.pkl', 'wb') as f:
#     pickle.dump(ukf_result, f)

# Read the UKF result from the pickle file
with open('ukf_result.pkl', 'rb') as f:
    ukf_result = pickle.load(f)

# Get the variances and residuals for each state
residuals = np.zeros((len(ukf_result), 2))
state_variances = np.zeros((len(ukf_result), 6))
for i, time in enumerate(ukf_result.keys()):
    state_variances[i] = np.sqrt(np.diag(ukf_result[time]['covar']))
    residuals[i] = np.array(ukf_result[time]['resids']).reshape(2,)

times = ukf_result.keys()

# Convert the times to a list
times = list(times)

time_datetime = []
# Convert the times in seconds since J2000 to UTC time
for i, time in enumerate(times):
    # Convert the DateTime object to a string
    time_string = time_conversion.date_time_from_epoch(time).iso_string()

    # Convert the string to a datetime object
    time_datetime_object = datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S.%f000000000")

    time_datetime.append(time_datetime_object)

# Get the time in days
times_days = np.array([time / 86400 - t0_seconds / 86400 for time in times])

# Get the initial state and covariance matrix
X0_forward = state_params['state']
P0_forward = state_params['covar']

# Get the final state from the UKF result
final_state = ukf_result[times[-1]]['state']

# Propagate the state and covariance matrix from the initial time to the final time
tvec = [times[0], times[-1]]
tf_forward, Xf_forward, Pf_forward = propagate_state_and_covar_original(X0_forward, P0_forward, tvec, state_params, intsettings)

# Get the final state and covariance from the UKF result
X0_backward = ukf_result[times[-1]]['state']
P0_backward = ukf_result[times[-1]]['covar']

# Propagate the state and covariance matrix from the final time to the initial time
tvec = [times[-1], t0_seconds]
tf_backward, Xf_backward, Pf_backward = propagate_state_and_covar_backward(X0_backward, P0_backward, tvec, state_params, intsettings)

# Convert tf_forward to days
tf_forward_days = np.array([time / 86400 - t0_seconds / 86400 for time in tf_forward])

# Convert tf_backward to days
tf_backward_days = np.array([time / 86400 - t0_seconds / 86400 for time in tf_backward])

kep_state = np.zeros((len(times), 6))
# Convert the ukf state from cartesian to keplerian
for i, time in enumerate(times):
    kep_state[i] = element_conversion.cartesian_to_keplerian(ukf_result[time]['state'], 398600.4415*1e9)

####################################################################################################
# Determine time of maneuver
####################################################################################################
SMA_residuals = kep_state[:, 0] - kep_state[0, 0]

# Find the first time where the SMA residual is greater than 100m
maneuver_time = times[np.where(SMA_residuals > 100)[0][0]]
maneuver_time_days = maneuver_time / 86400 - t0_seconds / 86400

maneuver_time_UTC = time_conversion.date_time_from_epoch(maneuver_time).iso_string()
maneuver_time_UTC = datetime.strptime(maneuver_time_UTC, "%Y-%m-%d %H:%M:%S.%f000000000")

print("The time of the maneuver is: ", maneuver_time)
print("The time of the maneuver in days is: ", maneuver_time_days)
print("The time of the maneuver in UTC is: ", maneuver_time_UTC)


####################################################################################################
# Rerun the UKF from the time of the maneuver to the end
####################################################################################################
# Get the state and covariance at the time of the maneuver
X0_maneuver = ukf_result[maneuver_time]['state']
P0_maneuver = ukf_result[maneuver_time]['covar']

# Create the state_params
state_params_maneuver = state_params.copy()
state_params_maneuver['UTC'] = maneuver_time_UTC
state_params_maneuver['state'] = X0_maneuver
state_params_maneuver['covar'] = P0_maneuver

# Trim the measurements to only include those after the maneuver
# For all values of key tk_list equal or larger than the maneuver time, keep the measurements

# Convert meas_dict to a pandas dataframe
meas_dict_df = pd.DataFrame(meas_dict)

# Only keep the measurements after the maneuver time
meas_dict_df = meas_dict_df[meas_dict_df['tk_list'] >= maneuver_time]

# Convert the dataframe back to a dictionary
meas_dict_maneuver = meas_dict_df.to_dict(orient='list')

# # Now run the Unscented Kalman Filter with the parameters
# ukf_result_maneuver = EstimationUtilities.ukf(state_params_maneuver, meas_dict_maneuver, sensor_params, intsettings, filter_params, bodies)
#
# # Save the UKF result to a pickle file
# with open('ukf_result_maneuver.pkl', 'wb') as f:
#     pickle.dump(ukf_result_maneuver, f)

# Read the UKF result from the pickle file
with open('ukf_result_maneuver.pkl', 'rb') as f:
    ukf_result_maneuver = pickle.load(f)

# Get the final state from the new UKF result
final_state_maneuver = ukf_result_maneuver[times[-1]]['state']

# Get the times for the UKF maneuver result
times_maneuver = ukf_result_maneuver.keys()

# Convert the times to a list
times_maneuver = list(times_maneuver)

# Get the time in days
times_maneuver_days = np.array([time / 86400 - t0_seconds / 86400 for time in times_maneuver])

# Get the position at the time of the maneuver
position_maneuver = ukf_result[maneuver_time]['state'][0:3]

# Get the velocity at the time of the maneuver
velocity_maneuver = np.array(ukf_result[maneuver_time]['state'][3:6]).reshape(3,)

# ####################################################################################################
# # Simulate the maneuver with different delta v values and propagate the state and covariance forward
# ####################################################################################################
# # Define the delta v values to simulate
# delta_v_x_array = np.linspace(-260, 280, 6)
# delta_v_y_array = np.linspace(380, 400, 6)
# delta_v_z_array = np.linspace(-520, -500, 6)
#
# tvec = [maneuver_time, times[-1]]
#
# # Initiate array to store all the results
# Xf_results = []
# Pf_results = []
# delta_v_results = []
#
# intsettings['step'] = 100
#
# i = 0
# for delta_v_x in delta_v_x_array:
#     for delta_v_y in delta_v_y_array:
#         for delta_v_z in delta_v_z_array:
#
#             print("\n-----------------------------------")
#             print(f"Case {i}")
#             print(f"Simulating delta v: {delta_v_x}, {delta_v_y}, {delta_v_z}")
#
#             X0_case = X0_maneuver.copy()
#             X0_case[3] += delta_v_x
#             X0_case[4] += delta_v_y
#             X0_case[5] += delta_v_z
#
#             tf_results, Xf_case, Pf_case = propagate_state_and_covar_original(X0_case, P0_maneuver, tvec, state_params_maneuver, intsettings, bodies=None)
#             Xf_results.append(Xf_case[:, 1:7])
#             Pf_results.append(Pf_case)
#             delta_v_results.append([delta_v_x, delta_v_y, delta_v_z])
#
#             i += 1
#
# import numpy as np
#
# # Convert tf_results to days
# tf_results_days = np.array([time / 86400 - t0_seconds / 86400 for time in tf_results])
#
# # Get the final position error for each component
# final_position_error = np.zeros((len(Xf_results),))
#
# # Get the final position error
# for l in range(len(Xf_results)):
#     final_position_error[l] = np.linalg.norm(np.array(Xf_results[l][-1][0:3]).reshape(3,) - np.array(final_state_maneuver[0:3]).reshape(3,))
#
# # Only keep the cases with the 20 least final position error
# indices = np.argsort(final_position_error)[0:20]
#
# # Calculate the error in each component for the cases with the least final position error
# final_position_error_components = np.array(Xf_results[indices[0]][-1][0:3]).reshape(3,) - np.array(final_state_maneuver[0:3]).reshape(3,)
#
# print('\n====================================')
# print(f"The index with the least final position error is: {indices[0]}")
# print(f"The final position error is: {final_position_error[indices[0]]}")
# print(f"The final position error components are: {final_position_error_components}")
# print(f"The delta v for this case is: {delta_v_results[indices[0]]}")
# # Convert Xf_results to a numpy array
# Xf_results_np = np.array(Xf_results)
#
# # Use the indices to filter the results
# Xf_results_filtered = Xf_results_np[indices.tolist()]
#
#
# labels = ["x position [m]", "y position [m]", "z position [m]"]
# # Make a plot of the positions over time for all cases and compare it to the UKF result
# fig, axs = plt.subplots(3, 1, figsize=(10, 6))
# for i in range(3):
#     for j in range(len(Xf_results_filtered)):
#         axs[i].plot(tf_results_days, [state[i] for state in Xf_results_filtered[j]], label="Case " + str(j))
#     axs[i].scatter(times_maneuver_days, [ukf_result_maneuver[time]['state'][i] for time in times_maneuver], c = 'red', s=5)
#     axs[i].set_xlabel("Time [days]")
#     axs[i].set_ylabel(labels[i])
#
# # axs[0].legend()
# plt.tight_layout()
# plt.show()
#
#
# # Make a plot for time index 167
# fig, axs = plt.subplots(3, 1, figsize=(10, 6))
# for i in range(3):
#
#     axs[i].plot(tf_results_days, [state[i] for state in Xf_results[indices[0]]], label="Optimal Impulsive Maneuver")
#     axs[i].scatter(times_maneuver_days, [ukf_result_maneuver[time]['state'][i] for time in times_maneuver], label="Post-Maneuver UKF Results", c = 'red', s=5)
#     axs[i].set_xlabel("Time [days]")
#     axs[i].set_ylabel(labels[i])
#
# axs[0].legend()
# plt.tight_layout()
# plt.show()
#
# # At what times is the backward propagated state closest to the position at the time of the maneuver
# position_backward = np.array([state[1:4] for state in Xf_backward])
# position_maneuver = np.reshape(position_maneuver, (1, 3))
# position_maneuver_loop = np.repeat(position_maneuver, len(position_backward), axis=0)
# position_difference = np.linalg.norm(position_backward - position_maneuver_loop, axis=1)
#
# # Find the time where the position is closest to the position at the time of the maneuver
# closest_time = tf_backward[np.argmin(position_difference)]
#
# print("The time of the closest position to the maneuver time is: ", closest_time, "with a position difference of: ", np.min(position_difference))
#
# # Get the velocity of the backward propagated state at the time of the maneuver
# velocity_maneuver_backward = np.array([state[4:7] for state in Xf_backward if state[0] == maneuver_time]).reshape(3,)
#
# # Compute the delta v needed for the maneuver
# delta_v = velocity_maneuver_backward - velocity_maneuver
#
# # Make a 3d plot of the orbit at the time of the maneuver using the backward propagated state
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot([state[1] for state in Xf_backward], [state[2] for state in Xf_backward], [state[3] for state in Xf_backward], label="Backward")
#
# # Include the position at the time of the maneuver
# ax.scatter(position_maneuver[0, 0], position_maneuver[0, 1], position_maneuver[0, 2], label="Position at maneuver", color='r')
#
# ax.set_xlabel("X [m]")
# ax.set_ylabel("Y [m]")
# ax.set_zlabel("Z [m]")
# ax.legend()
# plt.show()
#
#
#
#
# # Plot the x, y and z componennts of the UKF state and the backward propagated state for all times
# fig, axs = plt.subplots(3, 1, figsize=(10, 6))
# for i in range(3):
#     axs[i].scatter(times_days, [ukf_result[time]['state'][i] for time in times], label="UKF", s=2.5)
#     axs[i].scatter(tf_backward_days, [state[i+1] for state in Xf_backward], label="Backward", s=2.5)
#     axs[i].set_xlabel("Time [days]")
#     axs[i].set_ylabel("State component")
#     axs[i].set_title("State component " + str(i))
#     axs[i].legend()
#
# plt.tight_layout()
#
#
# # Plot the vx, vy and vz componennts of the UKF state and the backward propagated state for all times
# fig, axs = plt.subplots(3, 1, figsize=(10, 6))
# for i in range(3):
#     axs[i].scatter(times_days, [ukf_result[time]['state'][i+3] for time in times], label="UKF", s=2.5)
#     axs[i].scatter(tf_backward_days, [state[i+4] for state in Xf_backward], label="Backward", s=2.5)
#
#     # Include vertical lines to indicate the time of the maneuver
#     axs[i].axvline(x=maneuver_time_days, color='r', linestyle='--', label="Maneuver time")
#
#     axs[i].set_xlabel("Time [days]")
#     axs[i].set_ylabel("State component")
#     axs[i].set_title("State component " + str(i + 3))
#     axs[i].legend()
#
# plt.tight_layout()
# plt.show()
#
#
# # Plot the velocities of the UKF state and and the UKF state after the maneuver
# fig, axs = plt.subplots(3, 1, figsize=(10, 6))
# for i in range(3):
#     axs[i].scatter(times_days, [ukf_result[time]['state'][i+3] for time in times], label="UKF", s=2.5)
#     axs[i].scatter(times_maneuver_days, [ukf_result_maneuver[time]['state'][i+3] for time in times_maneuver], label="UKF maneuver", s=2.5)
#     axs[i].set_xlabel("Time [days]")
#     axs[i].set_ylabel("Velocity component")
#     axs[i].set_title("Velocity component " + str(i + 3))
#     axs[i].legend()
#
# plt.tight_layout()
# plt.show()
#











####################################################################################################
# Plots to help visualize
####################################################################################################

# Get the residuals for the UKF result after the maneuver
residuals_maneuver = np.zeros((len(ukf_result_maneuver), 2))
for i, time in enumerate(ukf_result_maneuver.keys()):
    residuals_maneuver[i] = np.array(ukf_result_maneuver[time]['resids']).reshape(2,)

# Calculate the RMS of each component of the residuals
rms_residuals = np.sqrt(np.mean(residuals**2, axis=0))

print("The RMS of the residuals in the RA component is: ", rms_residuals[0])
print("The RMS of the residuals in the DEC component is: ", rms_residuals[1])

labels = ["RA [rad]", "DEC [rad]"]

# Plot the residuals
fig, axs = plt.subplots(2, 1, figsize=(10, 6))
for i in range(2):
    axs[i].scatter(times_days, residuals[:, i], s=2.5, label = "Full UKF")
    axs[i].axvline(x=maneuver_time_days, color='r', linestyle='--', label="Maneuver time")
    axs[i].scatter(times_maneuver_days, residuals_maneuver[:, i], s=2.5, label = "Post-Maneuver UKF")
    axs[i].set_xlabel("Time [days]")
    axs[i].set_ylabel(labels[i])

axs[0].legend()
plt.tight_layout()
plt.show()


####################################################################################################
# Plot the 3-sigma covariance bounds for each component of the position in the RIC frame
####################################################################################################

position_ric = np.zeros((len(ukf_result), 3))
sigma_array = np.zeros((len(ukf_result), 3))

position_ric_maneuver = np.zeros((len(ukf_result_maneuver), 3))
sigma_array_maneuver = np.zeros((len(ukf_result_maneuver), 3))

k = 0
for i, time in enumerate(ukf_result.keys()):
    # Get the position and velocity in ECI frame
    position_eci = np.array(ukf_result[time]['state'][0:3]).reshape(3,)
    velocity_eci = np.array(ukf_result[time]['state'][3:6]).reshape(3,)

    # Get the position covariance
    P_eci = np.array(ukf_result[time]['covar'][0:3, 0:3]).reshape(3, 3)

    # Get the std deviation of the position
    std_dev_position = np.sqrt(np.diag(P_eci))

    # Save the position in the RIC frame
    Q_ric = EstimationUtilities.eci2ric(position_eci, velocity_eci)
    position_ric[i] = np.dot(Q_ric, position_eci)

    # Convert the position covariance to the RIC frame
    sigma_array[i, :] = np.sqrt(np.diag(Q_ric.T @ P_eci @ Q_ric))


    # If the time is after the maneuver, redo the calculation with the ukf result after the maneuver
    if time > maneuver_time:
        print("Also calculating the 3-sigma bounds for the post-maneuver UKF result")
        # Get the position and velocity in ECI frame
        position_eci_maneuver = np.array(ukf_result_maneuver[time]['state'][0:3]).reshape(3,)
        velocity_eci_maneuver = np.array(ukf_result_maneuver[time]['state'][3:6]).reshape(3,)

        # Get the position covariance
        P_eci_maneuver = np.array(ukf_result_maneuver[time]['covar'][0:3, 0:3]).reshape(3, 3)

        # Get the std deviation of the position
        std_dev_position_maneuver = np.sqrt(np.diag(P_eci_maneuver))

        # Save the position in the RIC frame
        Q_ric_maneuver = EstimationUtilities.eci2ric(position_eci_maneuver, velocity_eci_maneuver)
        position_ric_maneuver[k] = np.dot(Q_ric_maneuver, position_eci_maneuver)

        # Convert the position covariance to the RIC frame
        sigma_array_maneuver[k, :] = np.sqrt(np.diag(Q_ric_maneuver.T @ P_eci_maneuver @ Q_ric_maneuver))

        k += 1



labels = [r"$3\sigma$ bounds in $R$ [m]", r"$3\sigma$ bounds in $I$ [m]", r"$3\sigma$ bounds in $C$ [m]"]
# Plot the 3-sigma bounds
fig, axs = plt.subplots(3, 1, figsize=(10, 6))
for i in range(3):
    axs[i].fill_between(times_days, -3*sigma_array[:, i], 3*sigma_array[:, i], color='dodgerblue', alpha=0.5, label="Full UKF")
    # axs[i].fill_between(times_maneuver_days, -3*sigma_array_maneuver[:, i], 3*sigma_array_maneuver[:, i], color='orangered', alpha=0.5, label="Post-Maneuver UKF")
    axs[i].set_ylabel(labels[i])

axs[i].set_xlabel("Time [days]")

plt.tight_layout()
plt.show()





N = 1000

# Loop through the UKF result and calculate the 3-sigma bounds for each component of the position in the RIC frame
position_ric = np.zeros((len(ukf_result), 3))
sigma_array = np.zeros((len(ukf_result), 3))

for i, time in enumerate(ukf_result.keys()):
    # Get the position and velocity in ECI frame
    position_eci = np.array(ukf_result[time]['state'][0:3]).reshape(3,)
    velocity_eci = np.array(ukf_result[time]['state'][3:6]).reshape(3,)

    # Get the position covariance
    P_eci = np.array(ukf_result[time]['covar'][0:3, 0:3]).reshape(3, 3)

    # Get the std deviation of the position
    std_dev_position = np.sqrt(np.diag(P_eci))

    # Save the position in the RIC frame
    Q_ric = EstimationUtilities.eci2ric(position_eci, velocity_eci)
    position_ric[i] = np.dot(Q_ric, position_eci)

    # Initialize an array to store the Monte Carlo samples
    position_ric_MC = np.zeros((3, N))

    # Perform a Monte Carlo simulation for 1000 samples to get the RIC position and calculate the 3-sigma bounds
    for j in range(1000):
        # Get a sample of the position
        position_sample = np.random.multivariate_normal(position_eci, P_eci)

        # Get the ECI to RIC transformation matrix
        Q_ric = EstimationUtilities.eci2ric(position_sample, velocity_eci)

        # Convert the sample to the RIC frame
        position_ric_MC[:, j] = np.dot(Q_ric, position_sample)


    # Get the variance per component for each time
    position_ric_variance = np.var(position_ric_MC, axis=1)

    # Get the sigma
    sigma = np.sqrt(position_ric_variance)

    # Save the sigma
    sigma_array[i] = sigma

# Plot the 3-sigma bounds
fig, axs = plt.subplots(3, 1, figsize=(10, 6))
for i in range(3):
    axs[i].plot(times_days, position_ric[:, i], label="UKF", color='b')
    axs[i].fill_between(times_days, position_ric[:, i] - 3*sigma_array[:, i], position_ric[:, i] + 3*sigma_array[:, i], color='b', alpha=0.3)
    axs[i].set_xlabel("Time [days]")
    axs[i].set_ylabel("Position component")
    axs[i].set_title("Position component " + str(i))
    axs[i].legend()

plt.tight_layout()
plt.show()








# Plot the keplerian elements
fig, axs = plt.subplots(3, 2, figsize=(10, 6))
kep_elements = ['SMA [m]', 'ECC [-]', 'INC [rad]', 'RAAN [rad]', 'AOP [rad]', 'TA [rad]']
for i in range(3):
    axs[i, 0].scatter(times_days, kep_state[:, i], label="UKF", s=2.5)

    # Include vertical lines to indicate the time of the maneuver
    axs[i, 0].axvline(x=maneuver_time_days, color='r', linestyle='--', label="Maneuver time")

    axs[i, 0].set_xlabel("Time [days]")
    axs[i, 0].set_ylabel(kep_elements[i])
    axs[i, 0].legend()

    axs[i, 1].scatter(times_days, kep_state[:, i + 3], label="UKF", s=2.5)

    # Include vertical lines to indicate the time of the maneuver
    axs[i, 1].axvline(x=maneuver_time_days, color='r', linestyle='--', label="Maneuver time")

    axs[i, 1].set_xlabel("Time [days]")
    axs[i, 1].set_ylabel(kep_elements[i + 3])
    axs[i, 1].legend()

plt.tight_layout()
plt.show()

# Convert tf_forward to days
tf_forward_days = np.array([time / 86400 for time in tf_forward])


# Plot the x, y and z componennts of the UKF state and the forward propagated state for all times
fig, axs = plt.subplots(3, 1, figsize=(10, 6))
for i in range(3):
    axs[i].scatter(times_days, [ukf_result[time]['state'][i+1] for time in times], label="UKF", s=2.5)
    axs[i].plot(tf_forward_days, [state[i+1] for state in Xf_forward], label="Forward")
    axs[i].set_xlabel("Time [days]")
    axs[i].set_ylabel("State component")
    axs[i].set_title("State component " + str(i + 1))
    axs[i].legend()

plt.tight_layout()
plt.show()

# Plot the x, y and z componennts of the state for both cases for all times
fig, axs = plt.subplots(3, 1, figsize=(10, 6))
for i in range(3):
    axs[i].plot(tf_forward, [state[i+1] for state in Xf_forward], label="Forward")
    axs[i].plot(tf_backward, [state[i+1] for state in Xf_backward], label="Backward")
    axs[i].set_xlabel("Time")
    axs[i].set_ylabel("State component")
    axs[i].set_title("State component " + str(i + 1))
    axs[i].legend()

plt.tight_layout()
plt.show()

# Plot the x, y and z components of the velocities
fig, axs = plt.subplots(3, 1, figsize=(10, 6))
for i in range(3):
    axs[i].plot(tf_forward, [state[i+4] for state in Xf_forward], label="Forward")
    axs[i].plot(tf_backward, [state[i+4] for state in Xf_backward], label="Backward")
    axs[i].set_xlabel("Time")
    axs[i].set_ylabel("State component")
    axs[i].set_title("State component " + str(i + 4))
    axs[i].legend()

plt.tight_layout()
plt.show()

# Make a 3d plot of both orbits
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([state[1] for state in Xf_forward], [state[2] for state in Xf_forward], [state[3] for state in Xf_forward], label="Forward")
ax.plot([state[1] for state in Xf_backward], [state[2] for state in Xf_backward], [state[3] for state in Xf_backward], label="Backward")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.legend()
plt.show()
