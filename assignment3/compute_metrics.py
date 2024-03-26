import os
import json
import math
import time
import datetime
import numpy as np
import pandas as pd
import pickle as pkl
import TudatPropagator as prop
from scipy.integrate import dblquad




# constants
r_bw_tumbling = 2.090746

central_bodies = ['Earth']
bodies_to_create = ['Earth', 'Moon', 'Sun']
bodies = prop.tudat_initialize_bodies(bodies_to_create)

# integrator settings
# Integration parameters
int_params = {}
# for the RK4 integrator
step_rk4 = 0.0001        # initial step size [s]

# for the RKF78 integrator
step_rkf78 = 10       # initial step size
max_step_rkf78 = 50   # maximum step size
min_step_rkf78 = 1    # minimum step size
atol_rkf78 = 1e-8     # absolute tolerance
rtol_rkf78 = 1e-8     # relative tolerance

integrator_type = 'rk4'

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


def compute_euclidean_distance(r_sat, r_debris):
    d_eucl = np.linalg.norm(r_sat - r_debris)
    return d_eucl


def compute_mahalanobis_distance(s_sat, s_deb, P_sat, P_deb):
    d_maha = np.sqrt((s_sat[:3] - s_deb[:3]).T @ np.linalg.inv(P_sat[:3, :3] + P_deb[:3, :3]) @ (s_sat[:3] - s_deb[:3]))

    return d_maha


def Pc2D_Foster(X1, P1, X2, P2, HBR, rtol=1e-8, HBR_type='circle'):
    '''
    This function computes the probability of collision (Pc) in the 2D
    encounter plane following the method of Foster. The code has been ported
    from the MATLAB library developed by the NASA CARA team, listed in Ref 3.
    The function supports 3 types of hard body regions: circle, square, and
    square equivalent to the area of the circle. The input covariance may be
    either 3x3 or 6x6, but only the 3x3 position covariance will be used in
    the calculation of Pc.


    Parameters
    ------
    X1 : 6x1 numpy array
        Estimated mean state vector
        Cartesian position and velocity of Object 1 in ECI [m, m/s]
    P1 : 6x6 numpy array
        Estimated covariance of Object 1 in ECI [m^2, m^2/s^2]
    X2 : 6x1 numpy array
        Estimated mean state vector
        Cartesian position and velocity of Object 2 in ECI [m, m/s]
    P2 : 6x6 numpy array
        Estimated covariance of Object 2 in ECI [m^2, m^2/s^2]
    HBR : float
        hard-body region (e.g. radius for spherical object) [m]
    rtol : float, optional
        relative tolerance for numerical quadrature (default=1e-8)
    HBR_type : string, optional
        type of hard body region ('circle', 'square', or 'squareEqArea')
        (default='circle')

    Returns
    ------
    Pc : float
        probability of collision

    '''

    # Retrieve and combine the position covariance
    Peci = P1[0:3, 0:3] + P2[0:3, 0:3]

    # Construct the relative encounter frame
    r1 = np.reshape(X1[0:3], (3, 1))
    v1 = np.reshape(X1[3:6], (3, 1))
    r2 = np.reshape(X2[0:3], (3, 1))
    v2 = np.reshape(X2[3:6], (3, 1))
    r = r1 - r2
    v = v1 - v2
    h = np.cross(r, v, axis=0)

    # Unit vectors of relative encounter frame
    yhat = v / np.linalg.norm(v)
    zhat = h / np.linalg.norm(h)
    xhat = np.cross(yhat, zhat, axis=0)

    # Transformation matrix
    eci2xyz = np.concatenate((xhat.T, yhat.T, zhat.T))

    # Transform combined covariance to relative encounter frame (xyz)
    Pxyz = np.dot(eci2xyz, np.dot(Peci, eci2xyz.T))

    # 2D Projected covariance on the x-z plane of the relative encounter frame
    red = np.array([[1., 0., 0.], [0., 0., 1.]])
    Pxz = np.dot(red, np.dot(Pxyz, red.T))

    # Exception Handling
    # Remediate non-positive definite covariances
    Lclip = (1e-4 * HBR) ** 2.
    Pxz_rem, Pxz_det, Pxz_inv, posdef_status, clip_status = remediate_covariance(Pxz, Lclip)

    # Calculate Double Integral
    x0 = np.linalg.norm(r)
    z0 = 0.

    # Set up quadrature
    atol = 1e-13
    Integrand = lambda z, x: math.exp(
        -0.5 * (Pxz_inv[0, 0] * x ** 2. + Pxz_inv[0, 1] * x * z + Pxz_inv[1, 0] * x * z + Pxz_inv[1, 1] * z ** 2.))

    if HBR_type == 'circle':
        lower_semicircle = lambda x: -np.sqrt(HBR ** 2. - (x - x0) ** 2.) * (abs(x - x0) <= HBR)
        upper_semicircle = lambda x: np.sqrt(HBR ** 2. - (x - x0) ** 2.) * (abs(x - x0) <= HBR)
        Pc = (1. / (2. * math.pi)) * (1. / np.sqrt(Pxz_det)) * float(
            dblquad(Integrand, x0 - HBR, x0 + HBR, lower_semicircle, upper_semicircle, epsabs=atol, epsrel=rtol)[0])

    elif HBR_type == 'square':
        Pc = (1. / (2. * math.pi)) * (1. / np.sqrt(Pxz_det)) * float(
            dblquad(Integrand, x0 - HBR, x0 + HBR, z0 - HBR, z0 + HBR, epsabs=atol, epsrel=rtol)[0])

    elif HBR_type == 'squareEqArea':
        HBR_eq = HBR * np.sqrt(math.pi) / 2.
        Pc = (1. / (2. * math.pi)) * (1. / np.sqrt(Pxz_det)) * float(
            dblquad(Integrand, x0 - HBR_eq, x0 + HBR_eq, z0 - HBR_eq, z0 + HBR_eq, epsabs=atol, epsrel=rtol)[0])

    else:
        print('Error: HBR type is not supported! Must be circle, square, or squareEqArea')
        print(HBR_type)

    return Pc


def remediate_covariance(Praw, Lclip, Lraw=[], Vraw=[]):
    '''
    This function provides a level of exception handling by detecting and
    remediating non-positive definite covariances in the collision probability
    calculation, following the procedure in Hall et al. (Ref 2). This code has
    been ported from the MATLAB library developed by the NASA CARA team,
    listed in Ref 3.

    The function employs an eigenvalue clipping method, such that eigenvalues
    below the specified Lclip value are reset to Lclip. The covariance matrix,
    determinant, and inverse are then recomputed using the original
    eigenvectors and reset eigenvalues to ensure the output is positive (semi)
    definite. An input of Lclip = 0 will result in the output being positive
    semi-definite.

    Parameters
    ------
    Praw : nxn numpy array
        unremediated covariance matrix



    Returns
    ------


    '''

    # Ensure the covariance has all real elements
    if not np.all(np.isreal(Praw)):
        print('Error: input Praw is not real!')
        print(Praw)
        return

    # Calculate eigenvectors and eigenvalues if not input
    if len(Lraw) == 0 and len(Vraw) == 0:
        Lraw, Vraw = np.linalg.eig(Praw)

    # Define the positive definite status of Praw
    posdef_status = np.sign(min(Lraw))

    # Clip eigenvalues if needed, and record clipping status
    Lrem = Lraw.copy()
    if min(Lraw) < Lclip:
        clip_status = True
        Lrem[Lraw < Lclip] = Lclip
    else:
        clip_status = False

    # Determinant of remediated covariance
    Pdet = np.prod(Lrem)

    # Inverse of remediated covariance
    Pinv = np.dot(Vraw, np.dot(np.diag(1. / Lrem), Vraw.T))

    # Remediated covariance
    if clip_status:
        Prem = np.dot(Vraw, np.dot(np.diag(Lrem), Vraw.T))
    else:
        Prem = Praw.copy()

    return Prem, Pdet, Pinv, posdef_status, clip_status

def read_JSON_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

        # extract the data and turn them into numpy arrays
        times = np.array(data['Times'])
        positions = np.array(data['Positions [m]'])
        covariances = np.array(data['Covariances'])
    return times, positions, covariances


def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data


def get_GPS_information(data_RSO_catalog):

    GPS_data = data_RSO_catalog[36585]
    times, states, covariances = read_JSON_file(f"JSON_files\\{str(36585)}.json")


    print('break')
    return GPS_data, times, states, covariances

# Read TCA data
file_path = "results_tca_updated.csv"
data_TCA = pd.read_csv(file_path)

# Get RSO data
file_path_RSO_catalog = "data\group4\estimated_rso_catalog.pkl"
data_RSO_catalog = read_pkl(file_path_RSO_catalog)[0]
keys_RSO_catalog = data_RSO_catalog.keys()

# Get GPS data
GPS_data, GPS_times, GPS_states, GPS_covariances = get_GPS_information(data_RSO_catalog)

# read all the JSON files in the JSON_files directory
JSON_files = os.listdir("JSON_files")
JSON_files = [file for file in JSON_files if file.endswith(".json")]

# filter out the file with key 36585
JSON_files = [file for file in JSON_files if file.split(".")[0] != '36585']

def create_metrics_files():
    # loop through the JSON files compute the metrics
    for JSON_file in JSON_files:
        # extract the key from the JSON file
        key = int(JSON_file.split(".")[0])

        if key != 91861:
            print(f'skipped {key}')
            continue

        # get the data from the JSON file
        times, states, covariances = read_JSON_file(f"JSON_files\\{JSON_file}")

        # Get the TCA for this key
        TCA = data_TCA[data_TCA['Key'] == key]

        # Get the index of the TCA in the times array
        idx = np.where(times == TCA['TCA [s]'].values[0].round(0))[0][0]

        t0_short = TCA['TCA [s]'].values[0]-1
        tf_short = TCA['TCA [s]'].values[0]+1

        trange_short = np.array([t0_short, tf_short])

        # Get the GPS satellite state and covariance at t0 - 10s
        X_GPS = GPS_states[idx-1].reshape(6,1)
        P_GPS = GPS_covariances[idx-1]

        # apply a correction to the covariance matrix to ensure it is positive definite
        adj = np.eye(6) * 1e-6
        P_GPS = P_GPS + adj

        # Object parameters
        GPS_params = data_RSO_catalog[36585]
        GPS_params['sph_deg'] = 8
        GPS_params['sph_ord'] = 8
        GPS_params['central_bodies'] = central_bodies
        GPS_params['bodies_to_create'] = bodies_to_create

        # Get the debris state and covariance at t0 - 10s
        X_debris = states[idx-1].reshape(6, 1)
        P_debris = covariances[idx-1]

        # Add a correction to the covariance matrix to ensure it is positive definite
        P_debris = P_debris + adj

        # Object parameters
        debris_params = data_RSO_catalog[key]
        debris_params['sph_deg'] = 8
        debris_params['sph_ord'] = 8
        debris_params['central_bodies'] = central_bodies
        debris_params['bodies_to_create'] = bodies_to_create


        # get start time
        time_start = time.time()

        print("Propagating the GPS satellite's covariance...")
        GPS_times_short, GPS_states_short, GPS_covariances_short = prop.propagate_state_and_covar(X_GPS, P_GPS, trange_short, GPS_params, int_params)

        print("Propagating the GPS satellite's state...\n")
        GPS_times_short, GPS_states_short = prop.propagate_orbit(X_GPS, trange_short, GPS_params, int_params)

        print(f"Propagation of the GPS satellite took {time.time() - time_start} seconds")

        # get start time
        time_start = time.time()

        print("Propagating the debris satellite's covariance...")
        times_short, states_short, covariances_short = prop.propagate_state_and_covar(X_debris, P_debris, trange_short, debris_params, int_params)

        print("Propagating the debris satellite's state...\n")
        times_short, states_short = prop.propagate_orbit(X_debris, trange_short, debris_params, int_params)

        print(f"Propagation of the debris satellite took {time.time() - time_start} seconds")

        # get the hard body region
        HBR = np.sqrt(GPS_data['area']/math.pi) + np.sqrt(data_RSO_catalog[key]['area']/math.pi)

        d_eucl_array = np.zeros(len(times_short))
        d_maha_array = np.zeros(len(times_short))
        Pc_array = np.zeros(len(times_short))

        # loop over the times and compute the metrics
        for i, t in enumerate(times_short):
            # compute the metrics
            d_eucl = compute_euclidean_distance(states_short[i][:3], GPS_states_short[i][:3])
            d_maha = compute_mahalanobis_distance(states_short[i], GPS_states_short[i], covariances_short[i], GPS_covariances_short[i])
            Pc = Pc2D_Foster(states_short[i], covariances_short[i], GPS_states_short[i], GPS_covariances_short[i], HBR)

            # save the results to the arrays
            d_eucl_array[i] = d_eucl
            d_maha_array[i] = d_maha
            Pc_array[i] = Pc

        # save the results to a dataframe
        df = pd.DataFrame()
        df['Times'] = times_short
        df['Euclidean Distance [m]'] = d_eucl_array
        df['Mahalanobis Distance [m]'] = d_maha_array
        df['Pc'] = Pc_array.round(15)


        # save the dataframe to a csv file
        df.to_csv(f"metrics\\metrics_{key}.csv", index=False)


# Create metrics files
# create_metrics_files()


############################################################################################################
# Create CDM messages
############################################################################################################

# Loop through the objects in the metrics folder
for i, file in enumerate(os.listdir("metrics")):
    # read the csv file
    df = pd.read_csv(f"metrics\\{file}")

    # get the key from the file name
    key = int(file.split("_")[1].split(".")[0])

    # Get RSO data for this object
    RSO_data = data_RSO_catalog[key]

    # Get GPS data
    GPS_data = data_RSO_catalog[36585]

    # get the time and index of the time array at which the euclidean miss distance is the lowest
    idx = df['Euclidean Distance [m]'].idxmin()
    TCA_accurate = df['Times'][idx]

    # At this time get the Euclidean, Mahalanobis and Pc
    d_eucl_accurate = df['Euclidean Distance [m]'][idx]
    d_maha_accurate = df['Mahalanobis Distance [m]'][idx]
    Pc_accurate = df['Pc'][idx]

    # create Conjuction Data Message dictionary
    CDM = pd.DataFrame()
    CDM['CDM ID'] = i
    CDM['Creation Time'] = datetime.datetime(2024, 3, 21, 12, 0)
    CDM['Danger Flag'] = "yes"
    CDM['TCA since J2000 [s]'] = TCA_accurate
    CDM['Euclidean miss distance'] = d_eucl_accurate
    CDM['Mahalanobis miss distance'] = d_maha_accurate
    CDM['Probability of collision'] = Pc_accurate
    CDM['NORAD ID 1'] = 36585
    CDM['NORAD ID 2'] = key
    CDM['Radar Cross-Section 1 [m^2]'] = GPS_data['area']
    CDM['Radar Cross-Section 2 [m^2]'] = RSO_data['area']
    CDM['Volume 1 [m^3]'] = 4/3 * math.pi * (np.sqrt(GPS_data['area']/math.pi))**3
    CDM['Volume 2 [m^3]'] = 4/3 * math.pi * (np.sqrt(RSO_data['area']/math.pi))**3

    print('break')





