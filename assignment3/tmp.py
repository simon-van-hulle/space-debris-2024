"""A simple function to get the transformation matrix from inertial to RSW frame without TUDAT"""

import numpy as np
from numpy.linalg import norm
from tudatpy.astro import frame_conversion

def get_RSW_matrix(r, v):
        r = r / norm(r)
        h = np.cross(r, v)
        h = h / norm(h)
        w = np.cross(h, r)
        w = w / norm(w)
        return np.vstack((r, w, h)).T

def get_inertial_to_ric_matrix(state_ref):
    r = state_ref[:3]
    v = state_ref[3:]

    e_r = r / np.linalg.norm(r)
    h = np.cross(r, v)
    e_c = h / np.linalg.norm(h)
    e_i = np.cross(h, r)
    return np.vstack((r, w, h)).T

def geodetic_lat_long_alt_to_ecef(lat, long, alt):
    """Don't use Tudat"""
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    X = (N + alt) * np.cos(lat) * np.cos(long)
    Y = (N + alt) * np.cos(lat) * np.sin(long)
    Z = (N * (1 - e2) + alt) * np.sin(lat)
    return np.array([X, Y, Z])

# Test
# r = np.array([1, 0, 1.5])
# v = np.array([0, 0.9, 1.9])
# RSW = get_RSW_matrix(r, v)
# RSW2 = getRSW(r, v)
#
#
#
# print(RSW)
# print(RSW2)


lat = np.deg2rad(52.00667)
long = np.deg2rad(4.35556)
alt = 0.0

print(geodetic_lat_long_alt_to_ecef(lat, long, alt))