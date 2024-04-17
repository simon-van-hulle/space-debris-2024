"""
Initial Orbit Determination
"""

###############################################################################
# Imports
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

from datetime import datetime, timedelta

from tudatpy.astro.two_body_dynamics import LambertTargeterIzzo, propagate_kepler_orbit
from tudatpy.interface import spice
from tudatpy import constants
from tudatpy.astro import element_conversion
from tudatpy.numerical_simulation import environment

from sutils.utils import *
from sutils.logging import *
from sutils.style import *

from scipy.stats import gaussian_kde
import seaborn as sns

###############################################################################
# Setup
###############################################################################

mpl.use("tkagg")
np.set_printoptions(linewidth=150, suppress=True, threshold=np.inf, formatter={"float": "{: 0.5E}".format})


# def savefig(*args, **kwargs):
#     WARN("IGNORING SAVEFIG")
#     return


###############################################################################
# Constants
###############################################################################

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = f"{FILE_DIR}/data/group4"
# DATA_DIR = f"{FILE_DIR}/data/group5"
FIG_DIR = f"{FILE_DIR}/fig"
CACHE_DIR = f"{FILE_DIR}/cache"

IOD_MEAS_FILENAME = "q3_meas_iod_99004.pkl"
IOD_RESULT_FILENAME = "q3_meas_rso_99004.pkl"
# IOD_MEAS_FILENAME = "q3_meas_iod_99005.pkl"
IOD_MEAS_PATH = os.path.join(DATA_DIR, IOD_MEAS_FILENAME)

MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6378000
ARCSEC = np.deg2rad(1 / 3600)

###############################################################################
# Defaults
###############################################################################

DEFAULT_STATE_PARAMS = {
    "UTC": None,
    "state": None,
    "covar": None,
    "mass": 1000,
    "area": 1,
    "Cd": 1.3,
    "Cr": 1.3,
    "sph_deg": 8,
    "sph_ord": 8,
    "central_bodies": ["Earth"],
    "bodies_to_create": ["Earth", "Sun", "Moon"],
}


###############################################################################
# Utility Functions
###############################################################################


def within_limits(value, limits):
    return limits[0] <= value <= limits[1]


def sec_j2000_to_datetime(sec_j2000):
    return datetime(2000, 1, 1) + timedelta(seconds=sec_j2000)


def datetime_to_sec_j2000(dt):
    return (dt - datetime(2000, 1, 1)).total_seconds()


def get_kepler_interpolation(state, n_steps=100):
    kep_state = element_conversion.cartesian_to_keplerian(state, MU_EARTH)
    a = kep_state[0]

    period = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
    times = np.arange(0, period, period / n_steps)

    kep_states = [propagate_kepler_orbit(kep_state, t, MU_EARTH) for t in times]
    states = np.array([element_conversion.keplerian_to_cartesian(kep_state, MU_EARTH) for kep_state in kep_states])
    return states


def get_inertial_to_ric_matrix(state_ref):
    pos = state_ref[:3]
    vel = state_ref[3:]

    u_r = pos / np.linalg.norm(pos)
    angular_momentum = np.cross(pos, vel)
    u_c = angular_momentum / np.linalg.norm(angular_momentum)
    u_i = np.cross(u_c, u_r)

    return np.vstack((u_r, u_i, u_c)).T


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="testcmap"):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    mpl.colormaps.register(cmap=newcmap, force=True)

    return newcmap


###############################################################################
# Data structures
###############################################################################


class RsoFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw = load_pkl(file_path)
        self.state_params = self.raw[0]
        self.sensor_params = self.raw[1]
        self.meas_dict = self.raw[2]
        self.tk = np.array(self.meas_dict["tk_list"])
        self.Yk = np.array(self.meas_dict["Yk_list"]).squeeze()
        self.sensor_itrf = self.sensor_params["sensor_itrf"]

    def update_raw(self):
        self.raw[0] = self.state_params

    def save(self, *dirs, filename=None):
        if filename is None:
            filename = self.file_path.split("/")[-1]

        self.update_raw()
        save_pkl(filename, self.raw, *dirs)

    def plot_state(self, ax=None):
        state = self.state_params["state"]
        cov = self.state_params["covar"]

        return plot_state_cov_earth(state, cov, ax=ax)

    def __repr__(self):
        return f"IODMeasurement({self.file_path})"


def make_state_param_dict(
    utc: datetime,
    state: np.ndarray,
    covar: np.ndarray,
    mass: float,
    area: float,
    Cd: float,
    Cr: float,
    sph_deg: int,
    sph_ord: int,
    central_bodies: list,
    bodies_to_create: list,
) -> dict:
    """Simply a utility function to ensure all the correct data is in the dictionary"""
    state = state.reshape(6, 1)
    covar = covar.reshape(6, 6)

    if state.shape != (6, 1):
        raise ValueError("State must be a 6x1 numpy array")
    if covar.shape != (6, 6):
        raise ValueError("Covariance must be a 6x6 numpy array")
    if len(central_bodies) != 1:
        raise ValueError("Central bodies must be a list of length 1")
    if len(bodies_to_create) < 1:
        raise ValueError("Bodies to create must be a list of length > 0")

    state_params = {
        "UTC": utc,
        "state": state,
        "covar": covar,
        "mass": mass,
        "area": area,
        "Cd": Cd,
        "Cr": Cr,
        "sph_deg": sph_deg,
        "sph_ord": sph_ord,
        "central_bodies": central_bodies,
        "bodies_to_create": bodies_to_create,
    }
    return state_params


###############################################################################
# Gooding Method
###############################################################################


def calc_rho_hat(ra, dec):
    return np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)]).T


class AnglesOnlyData:
    def __init__(self, obs_times, observations, station_positions_inertial):

        self.obs_times = obs_times
        self.observations = observations
        self.right_ascensions = observations[:, 0]
        self.declinations = observations[:, 1]
        self.station_positions_inertial = station_positions_inertial
        self.rho_hats = calc_rho_hat(self.right_ascensions, self.declinations)
        self.check_shapes()
        self.converged = False

    @classmethod
    def from_iod_measurement(cls, meas: RsoFile):
        # Todo
        if not spice.get_total_count_of_kernels_loaded():
            raise ValueError("No SPICE kernels loaded. Necessary for inertial to ITRF conversion")

        obs_times = meas.tk
        station_positions_itrf = meas.sensor_itrf

        station_positions_inertial = np.array(
            [
                spice.compute_rotation_matrix_between_frames("IAU_Earth", "J2000", obs_times[i])
                @ station_positions_itrf
                for i in range(len(obs_times))
            ]
        ).squeeze()

        return cls(obs_times, meas.Yk, station_positions_inertial)

    @classmethod
    def perturbed_realisation(cls, other, obs_covariance_matrix):

        Yk = other.observations + np.random.multivariate_normal(
            np.zeros(2), obs_covariance_matrix, len(other.obs_times)
        )
        new_aod = cls(other.obs_times, Yk, other.station_positions_inertial)

        return new_aod

    def check_shapes(self):
        if not (len(self.obs_times) == len(self.right_ascensions) == len(self.declinations) == 3):
            raise ValueError("All lists must be the same length and equal to 3")

    def r_vec(self, index, range):
        return self.station_positions_inertial[index] + range * self.rho_hats[index]

    def r1_r3_vec(self, range1, range3):
        return self.r_vec(0, range1), self.r_vec(2, range3)

    def f_g(self, rho2_vec_calc):
        nvec = np.cross(self.rho_hats[1], rho2_vec_calc)
        nhat = nvec / np.linalg.norm(nvec)

        pvec = np.cross(nvec, self.rho_hats[1])
        phat = pvec / np.linalg.norm(pvec)

        f = np.dot(phat, rho2_vec_calc)
        g = np.dot(nhat, rho2_vec_calc)

        return f, g

    def get_lambert_targeter(self, range1, range3):
        r1_vec, r3_vec = self.r1_r3_vec(range1, range3)
        return LambertTargeterIzzo(r1_vec, r3_vec, self.obs_times[2] - self.obs_times[0], MU_EARTH)

    def get_kepler_initial_state(self, range1, range3):
        r1_vec, r3_vec = self.r1_r3_vec(range1, range3)
        lt = LambertTargeterIzzo(r1_vec, r3_vec, self.obs_times[2] - self.obs_times[0], MU_EARTH)
        v1_vec, v3_vec = lt.get_velocity_vectors()
        state1 = np.block([r1_vec, v1_vec])
        kep_state1 = element_conversion.cartesian_to_keplerian(state1, MU_EARTH)
        return kep_state1

    def get_semimajor_axis(self, range1, range3):
        kep_state1 = self.get_kepler_initial_state(range1, range3)
        return kep_state1[0]

    def get_kepler_interpolation(self, range1, range3, time_step=10):
        kep_state1 = self.get_kepler_initial_state(range1, range3)
        a = kep_state1[0]

        period = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
        times = np.arange(0, period, time_step)

        kep_states = [propagate_kepler_orbit(kep_state1, t, MU_EARTH) for t in times]
        states = np.array([element_conversion.keplerian_to_cartesian(kep_state, MU_EARTH) for kep_state in kep_states])
        return states

    def get_kepler_interpolated_state(self, range1, range3, time):
        kep_state1 = self.get_kepler_initial_state(range1, range3)
        kep_state = propagate_kepler_orbit(kep_state1, time - self.obs_times[0], MU_EARTH)
        state = element_conversion.keplerian_to_cartesian(kep_state, MU_EARTH)
        return state

    def calc_rho2_vec(self, range1, range3):
        state2 = self.get_kepler_interpolated_state(range1, range3, self.obs_times[1])
        r2_vec_calc = state2[:3]
        rho2_vec_calc = r2_vec_calc - self.station_positions_inertial[1]
        return rho2_vec_calc

    def newton_raphson_step(self, range1, range3, increment_factor=1e-6):

        delta_range1 = increment_factor * range1
        delta_range3 = increment_factor * range3

        # TODO: Use a better method to calculate the partial derivatives (TUDAT)

        rho2_vec_calc = self.calc_rho2_vec(range1, range3)
        f00, g00 = self.f_g(rho2_vec_calc)

        f01, g01 = self.f_g(self.calc_rho2_vec(range1, range3 + delta_range3))
        f0_1, g0_1 = self.f_g(self.calc_rho2_vec(range1, range3 - delta_range3))

        f10, g10 = self.f_g(self.calc_rho2_vec(range1 + delta_range1, range3))
        f11, g11 = self.f_g(self.calc_rho2_vec(range1 + delta_range1, range3 + delta_range3))
        f1_1, g1_1 = self.f_g(self.calc_rho2_vec(range1 + delta_range1, range3 - delta_range3))

        f_10, g_10 = self.f_g(self.calc_rho2_vec(range1 - delta_range1, range3))
        f_11, g_11 = self.f_g(self.calc_rho2_vec(range1 - delta_range1, range3 + delta_range3))
        f_1_1, g_1_1 = self.f_g(self.calc_rho2_vec(range1 - delta_range1, range3 - delta_range3))

        fx = (f10 - f_10) / (2 * delta_range1)
        fy = (f01 - f0_1) / (2 * delta_range3)
        gx = (g10 - g_10) / (2 * delta_range1)
        gy = (g01 - g0_1) / (2 * delta_range3)

        fxx = (f10 - 2 * f00 + f_10) / delta_range1**2
        fyy = (f01 - 2 * f00 + f0_1) / delta_range3**2
        fxy = (f11 - f_11 - f1_1 + f_1_1) / (4 * delta_range1 * delta_range3)

        hx = f00 * fx
        hy = f00 * fy
        hxx = f00 * fxx + fx * fx + gx * gx
        hyy = f00 * fyy + fy * fy + gy * gy
        hxy = f00 * fxy + fx * fy + gx * gy

        H = np.array([[hxx, hxy], [hxy, hyy]])
        H_inv = np.linalg.inv(H)
        delta = -H_inv @ np.array([hx, hy])

        # Check convergence
        r2_vec_calc = self.calc_rho2_vec(range1, range3) + self.station_positions_inertial[1]
        r2_calc = np.linalg.norm(r2_vec_calc)
        convergence = (abs(hx) + abs(hy)) / max(r2_calc, np.dot(rho2_vec_calc, self.rho_hats[1]))

        return delta, convergence

    def execute_newton_raphson(self, range1_0, range3_0, max_iter=300, tol=1e-9, verbose=True):
        self.converged = False
        range1 = range1_0
        range3 = range3_0
        for i in range(max_iter):
            delta, convergence = self.newton_raphson_step(range1, range3)
            if verbose:
                LOG(f"Iteration {i:0>3}: delta = {delta}, convergence = {convergence:.2e}")

            range1 += delta[0]
            range3 += delta[1]
            if convergence < tol:
                LOG("NR Converged")
                self.converged = True
                break

        return range1, range3


# TODO Check tolerance value
class GoodingIod:
    def __init__(self, initial_ranges=[10e6, 15e6], max_iter=300, tolerance=1e-9, rho_thresholds=[R_EARTH, 5e8]):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.converged = False
        self.rho1 = initial_ranges[0]
        self.rho3 = initial_ranges[1]
        self.rho_thresholds = rho_thresholds

    def run(self, aod, verbose=True):
        self.converged = False
        rho1 = self.rho1
        rho3 = self.rho3
        for i in range(self.max_iter):
            delta, convergence = aod.newton_raphson_step(rho1, rho3)
            if verbose:
                LOG(f"Iteration {i:0>3}: delta = {delta}, convergence = {convergence:.2e}")

            rho1 += delta[0]
            rho3 += delta[1]
            if convergence < self.tolerance:
                LOG("NR Converged")
                self.converged = True
                break

            if not within_limits(rho1, self.rho_thresholds) or not within_limits(rho3, self.rho_thresholds):
                WARN("Rho out of bounds")
                break

        semi_major_axis = aod.get_semimajor_axis(rho1, rho3)
        if semi_major_axis < 0:
            WARN("Negative semi-major axis")
            self.converged = False

        if self.converged:
            self.rho1 = rho1
            self.rho3 = rho3


class GoodingIodMonteCarlo:
    def __init__(
        self, goodingIOD: GoodingIod, aod: AnglesOnlyData, n_samples=100, observation_covariance=np.eye(2) * ARCSEC**2
    ):
        self.goodingIOD = goodingIOD
        self.aod = aod
        self.n_samples = n_samples
        self.observation_cov = observation_covariance

        self.initial_states = None
        self.state_cov = None
        self.rho1_list = None
        self.rho3_list = None
        self.rho1 = None
        self.rho3 = None

    def run(self):

        # Clear previous results
        self.initial_state_list = np.zeros((self.n_samples, 6))
        self.rho1_list = []
        self.rho3_list = []
        self.initial_states = []
        self.rho1_list = []
        self.rho3_list = []

        self.initial_state = np.zeros(6)
        self.state_cov = np.zeros((6, 6))

        for i in range(self.n_samples):
            aod_perturbed = AnglesOnlyData.perturbed_realisation(self.aod, self.observation_cov)
            self.goodingIOD.run(aod_perturbed, verbose=False)
            rho1 = self.goodingIOD.rho1
            rho3 = self.goodingIOD.rho3
            # rho1, rho3 = aod_perturbed.execute_newton_raphson(rho1, rho3, max_iter=500, tol=1e-9, verbose=False)
            if self.goodingIOD.converged:
                self.rho1_list.append(rho1)
                self.rho3_list.append(rho3)
                self.initial_states.append(
                    aod_perturbed.get_kepler_interpolated_state(rho1, rho3, self.aod.obs_times[0])
                )

        self.rho1 = np.mean(self.rho1_list)
        self.rho3 = np.mean(self.rho3_list)

        self.initial_state = np.mean(self.initial_states, axis=0)
        self.state_cov = np.cov(np.array(self.initial_states).T)


###############################################################################
# Plotting
###############################################################################


def plot_covariance(Cov, ax=None, cbar_label=r"Covariance [m$^2$]", labels=None):
    # plt.figure()
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    plt.title("Covariance Matrix")

    min_val = np.min(Cov)
    max_val = np.max(Cov)
    val_range = max_val - min_val
    min_val -= 0.01 * val_range
    max_val += 0.01 * val_range
    zero_point = np.max([-min_val, 0]) / val_range

    SHIFTED_CMAP = shiftedColorMap(mpl.cm.RdBu, midpoint=zero_point, name="shifted")
    # im = plt.imshow(Cov, cmap=SHIFTED_CMAP)
    im = plt.imshow(Cov, cmap="RdBu")
    cbar = plt.colorbar(im)

    N = len(Cov)
    cbar.set_label(cbar_label)

    if labels:
        if len(labels) != N:
            WARN("Labels must be the same length as the covariance matrix")
        else:
            plt.xticks(range(N), labels)
            plt.yticks(range(N), labels)
    plt.grid(False)
    return


def plot_cov_ellipsoid(state, Cov, scaling_factor=1, ax=None, *args, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    state = state[:3]
    Cov = Cov[:3, :3]

    # Find the rotation matrix and radii of the ellipsoid
    eig_val, eig_vec = np.linalg.eig(Cov)
    radii = np.sqrt(eig_val)

    # Generate points on the unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Apply the rotation matrix and scale by the radii
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], eig_vec) * radii

    x = (x + state[0]) * scaling_factor
    y = (y + state[1]) * scaling_factor
    z = (z + state[2]) * scaling_factor

    ax.plot_surface(x, y, z, *args, **kwargs)
    return fig, ax


def scatter_3d_with_projection(x, y, z, ax=None, *args, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    ax.scatter(x, y, z, *args, **kwargs)

    y_boundary = np.min(y) - 0.2 * np.ptp(y)
    x_boundary = np.min(x) - 0.2 * np.ptp(x)
    z_boundary = np.min(z) - 0.2 * np.ptp(z)

    yz = np.vstack([y, z])
    xy = np.vstack([x, z])
    xz = np.vstack([x, y])

    colx = gaussian_kde(yz)(yz)
    coly = gaussian_kde(xz)(xz)
    colz = gaussian_kde(xy)(xy)

    ax.scatter(y, z, c=colx, zdir="x", zs=x_boundary)
    ax.scatter(x, z, c=coly, zdir="y", zs=y_boundary)
    ax.scatter(x, y, c=colz, zdir="z", zs=z_boundary)

    ax.set_ylim(ymin=y_boundary)
    ax.set_xlim(xmin=x_boundary)
    ax.set_zlim(zmin=z_boundary)

    ax.view_init(elev=30, azim=50, roll=0)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return fig, ax


def plot_kepler_interpolation(state, n_steps=100, *args, **kwargs):
    ax = plt.gca()
    states = get_kepler_interpolation(state, n_steps)
    ax.plot(*states[:, :3].T, *args, **kwargs)


def plot_state_cov_earth(state, cov, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    state = state.flatten()
    cov = cov[:3, :3]

    state_line = np.vstack([np.zeros(3), state[:3]])
    ax.plot(*state_line.T, color="k", linestyle="--", label="Initial state")
    plot_cov_ellipsoid(state, cov, ax=ax, label="Uncertainty")
    plot_sphere(radius=R_EARTH, label="Earth")

    plot_kepler_interpolation(state, label="Orbit")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    ax.set_aspect("equal")

    return fig, ax


def plot_aod_solution(aod, range1, range3, ax=None, all_labels=True, **main_kwargs):
    r1_vec, r3_vec = aod.r1_r3_vec(range1, range3)

    rho_2_calc = aod.calc_rho2_vec(range1, range3)
    r2_vec = aod.r_vec(1, np.linalg.norm(rho_2_calc))
    r2_vec_calc = aod.calc_rho2_vec(range1, range3) + aod.station_positions_inertial[1]

    states = aod.get_kepler_interpolation(range1, range3)

    r1_vec = np.vstack([aod.station_positions_inertial[0], r1_vec])
    r2_vec = np.vstack([aod.station_positions_inertial[1], r2_vec])
    r3_vec = np.vstack([aod.station_positions_inertial[2], r3_vec])
    r2_vec_calc = np.vstack([aod.station_positions_inertial[1], r2_vec_calc])

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    # add range points
    m_size = 10
    ax.scatter(*r1_vec[1, :] / 1000, color="k", s=m_size)
    ax.scatter(*r3_vec[1, :] / 1000, color="k", s=m_size)

    plt.title("AOD Solution\n" rf"$\rho_1=${range1 / 1000:.1f} km, $\rho_3 =${range3 / 1000:.2f} km")
    ax.plot(*r1_vec.T / 1000, label="Measurements" if all_labels else None, color="k", linewidth=0.5)
    ax.plot(*r2_vec.T / 1000, color="k", linewidth=0.5)
    ax.plot(*r3_vec.T / 1000, color="k", linewidth=0.5)
    ax.plot(
        *r2_vec_calc.T / 1000,
        label=r"Calculated $\boldsymbol{\rho}_2$" if all_labels else None,
        color="r",
        linestyle="--",
        linewidth=0.5,
    )
    ax.plot(*states[:, :3].T / 1000, **main_kwargs)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")

    plot_sphere(radius=R_EARTH / 1000, label="Earth" if all_labels else None)
    ax.legend()

    ax.set_aspect("equal")
    return fig, ax


###############################################################################
# Testing
###############################################################################


def test_cov_plot():
    cov = np.diag([4, 16, 25])
    # cov = np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]])

    fig, ax = plot_cov_ellipsoid(np.zeros(3), cov)

    # plot_covariance(cov, labels=["x", "y", "z"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.title("Covariance Ellipsoid")

    savefig("covariance.pdf", FIG_DIR, "tmp", close=False)


def test_process_rso(filename="q4_meas_rso_91447.pkl"):
    RSO_PATH = os.path.join(DATA_DIR, filename)
    rso = RsoFile(RSO_PATH)
    fig, ax = rso.plot_state()
    savefig("rso_state.pdf", FIG_DIR, "tmp", close=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_cov_ellipsoid(rso.state_params["state"], rso.state_params["covar"], scaling_factor=1e-3, ax=ax)
    plt.title("RSO Covariance Ellipsoid")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")

    # ax.set_aspect('equal')
    savefig("rso_covariance.pdf", FIG_DIR, "tmp", close=False)


def draw_3d_frame(ax, color="k", **kwargs):

    def new_lim(lims):
        diff = lims[1] - lims[0]
        return [lims[0] - diff * 0.01, lims[1] + diff * 0.01]

    xlim = new_lim(ax.get_xlim())
    ylim = new_lim(ax.get_ylim())
    zlim = new_lim(ax.get_zlim())

    ax.plot(xlim, [ylim[0], ylim[0]], [zlim[0], zlim[0]], linewidth=1, color=color, **kwargs)
    ax.plot(xlim, [ylim[0], ylim[0]], [zlim[1], zlim[1]], linewidth=1, color=color, **kwargs)

    ax.plot([xlim[0], xlim[0]], ylim, [zlim[0], zlim[0]], linewidth=1, color=color, **kwargs)
    ax.plot([xlim[0], xlim[0]], ylim, [zlim[1], zlim[1]], linewidth=1, color=color, **kwargs)

    ax.plot([xlim[0], xlim[0]], [ylim[0], ylim[0]], zlim, linewidth=1, color=color, **kwargs)
    ax.plot([xlim[0], xlim[0]], [ylim[1], ylim[1]], zlim, linewidth=1, color=color, **kwargs)


###############################################################################
# Main
###############################################################################


def simple():
    SECTION("Simple Example")

    meas = RsoFile(IOD_MEAS_PATH)
    aod = AnglesOnlyData.from_iod_measurement(meas)

    ## GOODING METHOD #########################################################

    rho1 = 18e6
    rho3 = 25e6

    fig, ax = plot_aod_solution(aod, rho1, rho3)
    savefig("aod_initial.pdf", FIG_DIR, close=False)

    gooding = GoodingIod(initial_ranges=[rho1, rho3])

    gooding.run(aod)
    rho1, rho3 = gooding.rho1, gooding.rho3

    plot_aod_solution(aod, rho1, rho3, ax=ax)
    savefig("aod_final.pdf", FIG_DIR, close=False)


def aod_monte_carlo(n_samples=100):
    SECTION("Advanced Example")

    # Load measurements

    meas = RsoFile(IOD_MEAS_PATH)
    aod = AnglesOnlyData.from_iod_measurement(meas)

    ## GOODING METHOD #########################################################

    rho1 = 10e6
    rho3 = 20e6

    gooding = GoodingIod(initial_ranges=[rho1, rho3], rho_thresholds=[100e3, 50e6])
    mc = GoodingIodMonteCarlo(gooding, aod, n_samples=n_samples)
    mc.run()
    save_pkl("aod_monte_carlo.pkl", mc, CACHE_DIR)


def process_monte_carlo():
    mc = load_pkl(f"{CACHE_DIR}/aod_monte_carlo.pkl")
    rso = RsoFile(IOD_MEAS_PATH)
    figdirs = [FIG_DIR, "monte_carlo"]

    SECTION("Update RSO")
    rso.state_params = DEFAULT_STATE_PARAMS.copy()
    rso.state_params.update(
        {
            "UTC": sec_j2000_to_datetime(rso.tk[0]),
            "state": mc.initial_state,
            "covar": mc.state_cov,
        }
    )

    rso.save(DATA_DIR, filename=IOD_RESULT_FILENAME)

    SECTION("Monte Carlo Results")
    print("Initial State: ", mc.initial_state)
    LOG("Covariance: \n", mc.state_cov)

    fig, ax = plt.subplots(1, 1, figsize=(2.3, 2))
    plot_covariance(mc.state_cov[:3, :3], ax=ax, labels=["x", "y", "z"])
    savefig("aod_covariance.pdf", *figdirs, close=False)

    # RIC covariance
    rot = get_inertial_to_ric_matrix(mc.initial_state)
    rot = np.block([[rot, np.zeros((3, 3))], [np.zeros((3, 3)), rot]])
    states_ric = np.array([rot @ state for state in mc.initial_states])
    state_cov_ric = rot @ mc.state_cov @ rot.T
    fig, ax = plt.subplots(1, 1, figsize=(2.3, 2))
    plot_covariance(state_cov_ric[:3, :3], ax=ax, labels=["R", "I", "C"])
    savefig("aod_covariance_ric.pdf", *figdirs, close=False)

    fig, ax = plt.subplots(1, 1, figsize=(2.3, 2))
    plot_covariance(
        state_cov_ric[3:, 3:], ax=ax, cbar_label=r"Covariance [m$^2$/s$^2$]", labels=[r"$v_R$", r"$v_I$", r"$v_C$"]
    )
    savefig("aod_covariance_ric_vel.pdf", *figdirs, close=False)

    plot_cov_ellipsoid(np.zeros(3), mc.state_cov, scaling_factor=1e-3)
    savefig("cov_ellipsoid.pdf", *figdirs, close=False)

    fig, ax = plot_aod_solution(mc.aod, mc.rho1, mc.rho3)
    plot_cov_ellipsoid(mc.initial_state, mc.state_cov, scaling_factor=1e-3, ax=ax, alpha=0.5)
    savefig("aod_ellipsoid.pdf", *figdirs, close=False)

    # Plot AOD solution
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    plot_aod_solution(mc.aod, 10e6, 20e6, ax=ax, all_labels=False, label="Initial Guess")
    plot_aod_solution(mc.aod, mc.rho1, mc.rho3, ax=ax, all_labels=True, label="Converged Orbit")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.view_init(elev=30, azim=160, roll=0)

    savefig("aod_final.pdf", *figdirs, close=False)

    # Rho histogram
    plt.figure(figsize=(3, 2.5))
    plt.title("Range Histogram")
    plt.hist(
        np.array(mc.rho1_list) / 1000, bins=10, alpha=0.5, label=r"$\rho_1$", edgecolor="k", zorder=10, linewidth=0.1
    )
    plt.hist(
        np.array(mc.rho3_list) / 1000, bins=10, alpha=0.5, label=r"$\rho_3$", edgecolor="k", zorder=10, linewidth=0.1
    )
    plt.xlabel("Range [km]")
    plt.ylabel("Occurrences")
    plt.legend()

    savefig("aod_histogram.pdf", *figdirs, close=False)

    # Plot all realisations ###################################################
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    cov = mc.state_cov * 0
    fig, ax = plot_state_cov_earth(mc.initial_state, cov, ax=ax)
    i = 0
    for state in mc.initial_states:
        plot_kepler_interpolation(state)
        state_line = np.vstack([np.zeros(3), state[:3]])
        ax.plot(*state_line.T, color="k", linestyle="--")
        i += 1
        if i > 20:
            break

    ax.set_aspect("equal")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.view_init(elev=30, azim=160, roll=0)
    savefig("aod_monte_carlo_all.pdf", *figdirs, close=False)

    # Scatter plot all realisations with projection ###########################
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter_3d_with_projection(*(np.array(mc.initial_states - mc.initial_state).T[:3] / 1e3), ax=ax)
    ax.set_aspect("equal")
    draw_3d_frame(ax)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    savefig("aod_monte_carlo_scatter.pdf", *figdirs, pad_inches=0.3, close=False)

    # Scatter plot all ric realisations with projection #######################
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    mean_state_ric = np.mean(states_ric, axis=0)
    scatter_3d_with_projection(*(np.array(states_ric - mean_state_ric).T[:3] / 1e3), ax=ax)
    ax.set_xlabel("R [km]", fontsize=7)
    ax.set_ylabel("I [km]", fontsize=7)
    ax.set_zlabel("C [km]", fontsize=7)
    # plt.tight_layout()
    ax.set_aspect("equal")
    draw_3d_frame(ax)
    savefig("aod_monte_carlo_ric.pdf", *figdirs, pad_inches=0.3, close=False)

    # Scatter velocity realisations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter_3d_with_projection(*(np.array(states_ric - mean_state_ric).T[3:]), ax=ax)
    ax.set_xlabel("$v_R$ [m/s]", fontsize=7)
    ax.set_ylabel("$v_I$ [m/s]", fontsize=7)
    ax.set_zlabel("$v_C$ [m/s]", fontsize=7)
    ax.set_aspect("equal")
    draw_3d_frame(ax)
    savefig("aod_monte_carlo_ric_vel.pdf", *figdirs, pad_inches=0.3, close=False)

    # Rho covariance
    fig = plt.figure(figsize=(2.3, 2))
    ax = fig.add_subplot(111)
    rho_cov = np.cov(np.array([mc.rho1_list, mc.rho3_list]))
    plot_covariance(rho_cov, ax=ax, labels=[r"$\rho_1$", r"$\rho_3$"])
    savefig("aod_rho_covariance.pdf", *figdirs, close=False)

    # Plot rho correlation
    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_subplot(111)
    rho_corr = np.corrcoef(np.array([mc.rho1_list, mc.rho3_list]))
    xx = np.array(mc.rho1_list) / 1000
    yy = np.array(mc.rho3_list) / 1000
    xxyy = np.vstack([xx, yy])
    col = gaussian_kde(xxyy)(xxyy)
    ax.scatter(xx, yy, c=col, alpha=0.5, zorder=10, label="Samples")
    ax.set_xlabel(r"$\rho_1$ [km]")
    ax.set_ylabel(r"$\rho_3$ [km]")
    ax.axvline(mc.rho1 / 1000, color="k", linestyle="--", zorder=1)
    ax.axhline(mc.rho3 / 1000, color="k", linestyle="--", zorder=1, label="Mean solution")
    ax.set_title(f"Correlation: {rho_corr[0, 1]:.2f}")
    ax.set_aspect("equal")
    ax.legend()
    savefig("aod_rho_correlation.pdf", *figdirs, close=False)


def print_mc_info():
    figdirs = [FIG_DIR, "info"]
    mc = load_pkl(f"{CACHE_DIR}/aod_monte_carlo.pkl")
    LOG("Initial State: ", mc.initial_state)

    # Keplerian elements
    kep_state = element_conversion.cartesian_to_keplerian(mc.initial_state, MU_EARTH)
    LOG("Keplerian Elements: ", kep_state)
    LOG(f"a = {kep_state[0]/1000:.2f} km")
    LOG(f"e = {kep_state[1]:.4f}")
    LOG(f"i = {np.degrees(kep_state[2]):.2f} deg")
    LOG(f"RAAN = {np.degrees(kep_state[3]):.2f} deg")
    LOG(f"AoP = {np.degrees(kep_state[4]):.2f} deg")
    LOG(f"TA = {np.degrees(kep_state[5]):.2f} deg")

    # GPS orbit
    TLE  = environment.Tle(
        "1 36585U 10022A   24106.74392310 -.00000020  00000+0  00000+0 0  9995",
        "2 36585  54.4704 240.5201 0115895  61.7635 323.1802  2.00573615101701",
    )
    
    state_gps =  spice.get_cartesian_state_from_tle_at_epoch(mc.aod.obs_times[0], TLE)
    states_gps = get_kepler_interpolation(state_gps)

    # Plot AOD solution with GPS orbit
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    plot_aod_solution(mc.aod, mc.rho1, mc.rho3, ax=ax, all_labels=False, label="Converged Orbit")
    ax.plot(*states_gps[:, :3].T / 1000, label="NAVSTAR-65")
    ax.plot(*state_gps[:3] / 1000, marker="o", markersize=3)
    
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    
    ax.view_init(elev=30, azim=160, roll=0)
    savefig("aod_gps.pdf", *figdirs, close=False)


if __name__ == "__main__":
    spice.load_standard_kernels()

    TITLE("Assignment 4: Initial Orbit Determination")
    np.random.seed(37)

    # simple()
    # aod_monte_carlo(n_samples=1000)

    # Process
    # process_monte_carlo()
    print_mc_info()

    # test_cov_plot()
    # test_process_rso(IOD_RESULT_FILENAME)
    # test_process_rso("q4_meas_rso_91447.pkl")

    plt.show()
    # END()
