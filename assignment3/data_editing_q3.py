import numpy as np
import EstimationUtilities as EstUtil


# %%
# define optical sensor

sensor = 'diego garcia'
# data for diego garcia
if sensor == 'diego garcia':
    latitude_radians = np.deg2rad(7.411754)
    longitude_radians = np.deg2rad(72.452285)
    geoid_height_meters = -73.58
    altitude_meters = 1.2192
    geodetic_height_meters = geoid_height_meters + altitude_meters

optical_sensor_params = EstUtil.define_optical_sensor(latitude_radians, longitude_radians, geodetic_height_meters)
# define a radar sensor to get the elevation of the objects and of the sun
radar_sensor_params = EstUtil.define_radar_sensor(latitude_radians, longitude_radians, geodetic_height_meters)
# norad ids to track
norad_list = ['91104', '91368', '91401', '91438', '91447', '91518', '91714', '91787', '91821', '91861']


# %% load all data
data_all = np.zeros((8641, 14, 10))  # 3d array, width is for each object
i = 0
for NORAD in norad_list:
    csv_file = 'dep_var_' + NORAD + '.csv'  # name of csv file
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)  # load data in numpy array
    data_all[:, :, i] = data
    i += 1
print('data loaded')
time_J2000 = data[:, 1]
state_0 = np.transpose(data_all[1, 8:14, :])

# srp check (illumination): independent of gs
a_srp_all = data_all[:, 2:5, :]
a_srp_check = np.linalg.norm(a_srp_all, axis=1)
a_srp_check = (a_srp_check != 0).astype(int)
np.savetxt('a_srp_check.dat', a_srp_check, fmt='%d')

# sun elevation check
sun_pos = data[:, 5:8]
sun_el_check = np.zeros(len(time_J2000))
for i in range(len(time_J2000)):
    rho_az_el_sun = EstUtil.compute_measurement(time_J2000[i], sun_pos[i, :], radar_sensor_params)
    print(i)
    if rho_az_el_sun[-1] < -12 * np.pi / 180:
        sun_el_check[i] = 1
sun_el_check = np.tile(sun_el_check, (10, 1)).T
# save this
if sensor == 'diego garcia':
    np.savetxt('sun_el_check_diego_garcia.dat', sun_el_check, fmt='%d')

if sensor == 'australia':
    np.savetxt('sun_el_check_australia.dat', sun_el_check, fmt='%d')


# elevation mask check
elevation_check = np.zeros((len(time_J2000), len(norad_list)))
counter = 0
for NORAD in norad_list:
    for i in range(len(time_J2000)):
        el = EstUtil.compute_measurement(time_J2000[i], data_all[i, 8:14, counter], radar_sensor_params)[-1, 0]
        if el > 15 * np.pi / 180:
            elevation_check[i, counter] = 1
    counter += 1
    print(counter)

if sensor == 'diego garcia':
    np.savetxt('elevation_check_diego_garcia.dat', elevation_check, fmt='%d')

if sensor == 'australia':
    np.savetxt('elevation_check_australia.dat', elevation_check, fmt='%d')


# measurability logical table
measurability_check = (sun_el_check[:8640, :]*a_srp_check[:8640, :]*elevation_check[:8640, :]).astype(int)
if sensor == 'diego garcia':
    np.savetxt('measurability_check_diego_garcia.dat', measurability_check, fmt='%d')

if sensor == 'australia':
    np.savetxt('measurability_check_australia.dat', measurability_check, fmt='%d')

print('observability from ground station: ' + sensor)
for i in range(10):
    obj = np.nonzero(measurability_check[:, i])
    print('obj ' + str(i) + ' starts at ' + str(obj[0][0]) + ' and ends at ' + str(obj[0][-1]))

# %% redo computations for sensor in australia. Code is commented because the final choice was Diego Garcia
# sensor = 'australia'
# # data for australia southern obs
# if sensor == 'australia':
#     latitude_radians = np.deg2rad(21.912222222)
#     longitude_radians = np.deg2rad(114.09)
#     height_meters = 64.9224
#     geoid_height_meters = -14.69
#     geodetic_height_meters = height_meters + geoid_height_meters
#
# optical_sensor_params = EstUtil.define_optical_sensor(latitude_radians, longitude_radians, geodetic_height_meters)
# # define a radar sensor to get the elevation of the objects and of the sun
# radar_sensor_params = EstUtil.define_radar_sensor(latitude_radians, longitude_radians, geodetic_height_meters)
#
# # sun elevation check
# sun_pos = data[:, 5:8]
# sun_el_check = np.zeros(len(time_J2000))
# for i in range(len(time_J2000)):
#     rho_az_el_sun = EstUtil.compute_measurement(time_J2000[i], sun_pos[i, :], radar_sensor_params)
#     print(i)
#     if rho_az_el_sun[-1] < -12 * np.pi / 180:
#         sun_el_check[i] = 1
# sun_el_check = np.tile(sun_el_check, (10, 1)).T
# # save this
# if sensor == 'diego garcia':
#     np.savetxt('sun_el_check_diego_garcia.dat', sun_el_check, fmt='%d')
#
# if sensor == 'australia':
#     np.savetxt('sun_el_check_australia.dat', sun_el_check, fmt='%d')
#
#
# # elevation mask check
# elevation_check = np.zeros((len(time_J2000), len(norad_list)))
# counter = 0
# for NORAD in norad_list:
#     for i in range(len(time_J2000)):
#         el = EstUtil.compute_measurement(time_J2000[i], data_all[i, 8:14, counter], radar_sensor_params)[-1, 0]
#         if el > 15 * np.pi / 180:
#             elevation_check[i, counter] = 1
#     counter += 1
#     print(counter)
#
# if sensor == 'diego garcia':
#     np.savetxt('elevation_check_diego_garcia.dat', elevation_check, fmt='%d')
#
# if sensor == 'australia':
#     np.savetxt('elevation_check_australia.dat', elevation_check, fmt='%d')
#
#
# # measurability logical table
# measurability_check = (sun_el_check[:8640, :]*a_srp_check[:8640, :]*elevation_check[:8640, :]).astype(int)
# if sensor == 'diego garcia':
#     np.savetxt('measurability_check_diego_garcia.dat', measurability_check, fmt='%d')
#
# if sensor == 'australia':
#     np.savetxt('measurability_check_australia.dat', measurability_check, fmt='%d')
#
# print('observability from ground station: ' + sensor)
# for i in range(10):
#     obj = np.nonzero(measurability_check[:, i])
#     if obj[0].size > 0:
#         print('obj ' + str(i) + ' starts at ' + str(obj[0][0]) + ' and ends at ' + str(obj[0][-1]))
