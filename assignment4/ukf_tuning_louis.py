import matplotlib.pyplot as plt
import numpy as np

# from utils.style import *
# from utils.utils import *
import pickle
import os
import sys
sys.path.append('assignment2')
from EstimationUtilities import *
from BreakupUtilities import *
from tudatpy.astro import frame_conversion
from sklearn.utils import Bunch
from dataclasses import dataclass
import time
import copy

########################################################################################################################
# Constants
########################################################################################################################
BASE_DIR = os.path.dirname(__file__)

SUB_QUESTION = "q2a"
# SUB_QUESTION = "q2b"

DATA_DIR = f"{BASE_DIR}/data/group4"
RADAR_FILE = f"group4_{SUB_QUESTION}_gps_sparse_radar_meas.pkl"
OPTICAL_FILE = f"group4_{SUB_QUESTION}_gps_sparse_optical_meas.pkl"

TRUTH_FILE = f"group4_{SUB_QUESTION}_gps_sparse_truth_grav{'_srp' if SUB_QUESTION == 'q2b' else ''}.pkl"
FIG_DIR = f"{BASE_DIR}/figures"

I3 = np.eye(3, 3)
I6 = np.eye(6, 6)
Z3 = np.zeros((3, 3))
Z6 = np.zeros((6, 6))


########################################################################################################################
# Tudat boilerplate
########################################################################################################################

def tudat_initialize_bodies():
    # Load spice kernels
    spice_interface.clear_kernels()
    spice_interface.load_standard_kernels()

    # Define string names for bodies to be created from default.
    bodies_to_create = ["Sun", "Earth", "Moon"]

    # Use "Earth"/"J2000" as global frame origin and orientation.
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"

    # Create default body settings, usually from `spice`.
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation
    )

    # Create system of selected celestial bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    return bodies


########################################################################################################################
# Calculate error between two states in the RIC frame
########################################################################################################################

def get_inertial_to_ric_matrix(state_ref):
    pos = state_ref[:3]
    vel = state_ref[3:]

    u_r = (pos / np.linalg.norm(pos))
    angular_momentum = np.cross(pos, vel)
    u_c = angular_momentum / np.linalg.norm(angular_momentum)
    u_i = np.cross(u_c, u_r)

    return np.vstack((u_r, u_i, u_c)).T


def calc_ric_error(state_est, state_true, covar_est):
    # Calculate the rotation matrix
    # iec2rec = frame_conversion.inertial_to_rsw_rotation_matrix(state_true)
    iec2rec = get_inertial_to_ric_matrix(state_true)
    iec2rec_full = np.block([[iec2rec, Z3], [Z3, iec2rec]])

    # Get the error in the RIC frame
    state_err = state_est - state_true
    state_err_ric = iec2rec_full @ state_err
    covar_ric = iec2rec_full @ covar_est @ iec2rec_full.T

    return state_err_ric, covar_ric


def calc_ric_errors(X_est, X_true, Cov_eci):
    X_err_ric = np.zeros_like(X_est)
    Cov_ric = np.zeros_like(Cov_eci)
    for i in range(len(X_est)):
        err_ric, covar_ric = calc_ric_error(X_est[i], X_true[i], Cov_eci[i])
        X_err_ric[i] = err_ric
        Cov_ric[i] = covar_ric

    return X_err_ric, Cov_ric


########################################################################################################################
# Load data
########################################################################################################################

class Truth:
    def __init__(self, filename, directory=DATA_DIR):
        self.filename = os.path.join(directory, filename)
        self.t_truth, self.X_truth, self.state_params = read_truth_file(self.filename)
        return

    def make_dense(self, int_params, dt=10, n_points=1000):
        LOG(f"Generating Dense orbit with {n_points} time steps")

        t0 = self.t_truth[0]
        self.t_truth = np.linspace(t0, t0 + n_points * dt, n_points)
        bodies = tudat_initialize_bodies()
        Po = np.diag([1e8, 1e8, 1e8, 100., 100., 100.])
        int_params['tudat_integrator'] = 'rk4'
        int_params['step'] = dt
        self.t_truth, self.X_truth, _ = prop.propagate_state_and_covar(
            self.X_truth[0].reshape(6, 1), Po, self.t_truth, self.state_params, int_params, bodies, alpha=1
        )
        self.X_truth = self.X_truth.reshape(-1, 6)


class Measurement:
    def __init__(self, filename, directory=DATA_DIR):
        self.filename = os.path.join(directory, filename)
        print(self.filename)
        self.state_params, self.meas_dict, self.sensor_params = read_measurement_file(self.filename)
        return

    def generate_from_truth(self, truth: Truth):
        """Taken from Steve's code"""

        meas_types = self.sensor_params['meas_types']
        if not ('ra' in meas_types and 'dec' in meas_types):
            raise NotImplementedError("RA and DEC measurements are required for this function")

        # Setup
        bodies = tudat_initialize_bodies()
        m = 2
        Xt_mat = truth.X_truth
        theta0 = 0
        dtheta = 2 * np.pi / (24 * 3600)

        t_obs = truth.t_truth
        obs_data = np.zeros((len(t_obs), m))
        resids = np.zeros((len(t_obs), m))
        for kk in range(len(t_obs)):

            # Current time and true position states
            tk = t_obs[kk]
            r_eci = Xt_mat[kk, 0:3].reshape(3, 1)

            # Compute earth rotation and sensor position in ECI
            earth_rotation_model = bodies.get('Earth').rotation_model
            earth_rotation_at_epoch = earth_rotation_model.body_fixed_to_inertial_rotation(tk)
            sensor_eci = earth_rotation_at_epoch @ self.sensor_params['sensor_itrf']

            # Compute range and line of sight unit vector
            rho_eci = r_eci - sensor_eci
            rho = np.linalg.norm(rho_eci)
            rho_hat_eci = rho_eci / rho

            # Compute topocentric right ascension and declination
            ra = math.atan2(rho_hat_eci[1, 0], rho_hat_eci[0, 0])
            dec = math.asin(rho_hat_eci[2, 0])

            if m == 2:
                obs_data[kk, 0] = ra + np.random.randn() * self.sensor_params['sigma_dict']['ra']
                obs_data[kk, 1] = dec + np.random.randn() * self.sensor_params['sigma_dict']['dec']

                resids[kk, 0] = obs_data[kk, 0] - ra
                resids[kk, 1] = obs_data[kk, 1] - dec
            else:
                raise NotImplementedError("Only 2 measurements are supported")

        # Update!
        self.meas_dict['tk_list'] = t_obs
        self.meas_dict['Yk_list'] = [observation.reshape(2, 1) for observation in obs_data]

        return


########################################################################################################################
# Math utilities
########################################################################################################################

def Std_from_Cov(Cov):
    return np.sqrt(np.diagonal(Cov, axis1=-2, axis2=-1))


########################################################################################################################
# UKF
########################################################################################################################

class UkfSettings:
    def __init__(self):
        self.Q_eci = I3 * 0
        self.Q_ric = I3 * 0
        self.alpha = 0.1
        self.gap_seconds = 600

        self.tudat_integrator = "rkf78"
        self.step = 10.
        self.max_step = 1000.
        self.min_step = 1e-4
        self.rtol = 1e-12
        self.atol = 1e-12
        self.measurements = None

    def get_filter_params(self):
        return {"Qeci": self.Q_eci, "Qric": self.Q_ric, "alpha": self.alpha, "gap_seconds": self.gap_seconds}

    def get_int_params(self):
        return {
            "tudat_integrator": self.tudat_integrator,
            "step": self.step,
            "max_step": self.max_step,
            "min_step": self.min_step,
            "rtol": self.rtol,
            "atol": self.atol
        }

    def run(self, measurements: Measurement, perturb_x0=False, **kwargs):
        bodies = tudat_initialize_bodies()
        self.measurements = measurements
        print(self.get_filter_params())

        state_params = measurements.state_params
        state_params['Cd'] = 0.0
        state_params['Cr'] = 0.0

        if perturb_x0:
            state_params['state'] = np.random.multivariate_normal(
                state_params['state'].flatten(), perturb_x0 * state_params['covar']
            ).reshape(6, 1)
            state_params['covar'] = perturb_x0 * state_params['covar']

        return ukf(
            state_params,
            measurements.meas_dict,
            measurements.sensor_params,
            self.get_int_params(),
            self.get_filter_params(),
            bodies,
            **kwargs
        )


class UkfResult:
    def __init__(self, ukf_output: dict, filter_settings: UkfSettings):
        self.times = np.array(list(ukf_output.keys()))
        self.X_est = np.array([ukf_output[t]["state"] for t in self.times]).reshape(-1, 6)
        self.Cov = np.array([ukf_output[t]["covar"] for t in self.times])

        n_resid = ukf_output[self.times[0]]["resids"].shape[0]
        self.Resid = np.array([ukf_output[t]["resids"] for t in self.times]).reshape(-1, n_resid)

        self.time_step = self.times[1] - self.times[0]
        self.filter_settings = filter_settings
        self.Std = None
        self.Std_ric = None
        self.X_err = None
        self.X_err_ric = None
        self.Cov_ric = None
        self.rms_pos = None
        self.rms_vel = None
        self.rms_resid = None

    def get_indices_start_pass(self):
        """Find the indices where a pass starts"""
        pass_indices = [0]
        time_idx = 0
        for _ in range(len(self.times) - 1):
            time_idx += 1
            if self.times[time_idx] - self.times[time_idx - 1] > 100 * self.time_step:
                pass_indices.append(time_idx)

        pass_indices.append(time_idx + 1)
        return pass_indices

    def update_errors(self, X_truth, n_ignore_rms=0):
        self.X_err = self.X_est - X_truth
        self.Std = Std_from_Cov(self.Cov)

        self.X_err_ric, self.Cov_ric = calc_ric_errors(self.X_est, X_truth, self.Cov)
        self.Std_ric = Std_from_Cov(self.Cov_ric)

        self.rms_pos = np.sqrt(np.mean(np.linalg.norm(self.X_err[n_ignore_rms:, :3], axis=1) ** 2, axis=0))
        self.rms_vel = np.sqrt(np.mean(np.linalg.norm(self.X_err[n_ignore_rms:, 3:], axis=1) ** 2, axis=0))
        self.rms_resid = np.sqrt(np.mean(self.Resid ** 2, axis=0))


class UkfResultCollection:
    def __init__(self, fig_dirs):
        self.settings = []
        self.rms_pos = []
        self.rms_vel = []
        self.rms_resid = []
        self.fig_dirs = fig_dirs or [FIG_DIR]

    def add_result(self, ukf_result: UkfResult):
        self.settings.append(copy.copy(ukf_result.filter_settings))
        self.rms_pos.append(ukf_result.rms_pos)
        self.rms_vel.append(ukf_result.rms_vel)
        self.rms_resid.append(ukf_result.rms_resid)

    def plot_rms_var_ric(self):
        var_ric = [s.Q_ric[0, 0] for s in self.settings]
        meas_types = self.settings[0].measurements.sensor_params['meas_types']
        n = len(meas_types)
        fig, axs = plt.subplots(2, 1, figsize=(4.5, n * 1.6), dpi=600, sharex=True)
        axs[0].plot(var_ric, self.rms_pos, '.-k')
        axs[0].set_ylabel("RMS pos err [m]")
        axs[1].plot(var_ric, np.array(self.rms_vel) * 1000, '.-k')
        axs[1].set_ylabel("RMS vel err [mm/s]")
        axs[-1].set_xlabel(r"$\sigma_{ric}^2$ [m$^2$/s$^2$]")
        for ax in axs:
            ax.grid(True)
            ax.yaxis.set_label_coords(-0.2, 0.5)
        plt.xscale("log")
        savefig("rms_err_var_ric", *self.fig_dirs)

        fig, axs = plt.subplots(n, 1, figsize=(4.5, n * 1.6), dpi=600, sharex=True)
        print(meas_types)
        for i, meas_type in enumerate(meas_types):
            if meas_type in ["el", "az"]:
                y_label = f"RMS {meas_type} res [deg]"
                def convert(x): return np.rad2deg(x)
            elif meas_type in ["dec", "ra"]:
                y_label = f"RMS {meas_type} res ['']"
                def convert(x): return rad2arcsec(x)
            else:
                y_label = f"RMS {meas_type} res [m]"
                def convert(x): return x

            axs[i].plot(var_ric, convert(np.array(self.rms_resid)[:, i]), '.-', color=COLOR(i), label=meas_type)
            axs[i].legend()
            axs[i].set_ylabel(y_label)

        axs[-1].set_xlabel(r"$\sigma_{ric}^2$ [m$^2$/s$^2$]")

        for ax in axs:
            ax.grid(True)
            ax.yaxis.set_label_coords(-0.2, 0.5)
        fig.tight_layout()
        plt.xscale("log")
        savefig("rms_res_var_ric", *self.fig_dirs)

    def plot_rms_var_ric2d(self, N_max_x=5, N_max_y=5):
        var_ric_x = np.array([s.Q_ric[0, 0] for s in self.settings])
        var_ric_y = np.array([s.Q_ric[1, 1] for s in self.settings])
        N = np.unique(var_ric_x).shape[0]
        var_ric_x = var_ric_x.reshape(N, -1)
        var_ric_y = var_ric_y.reshape(N, -1)
        rms_pos = np.array(self.rms_pos).reshape(N, -1)
        rms_vel = np.array(self.rms_vel).reshape(N, -1) * 1000  # mm/s

        filter = (var_ric_x < 1e-3) * (var_ric_y < 1e-3)

        meas_types = self.settings[0].measurements.sensor_params['meas_types']
        n = len(meas_types)
        fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=600, sharex=True)
        res = axs[0].contourf(var_ric_x[:N_max_x, :N_max_y], var_ric_y[:N_max_x, :N_max_y], rms_pos[:N_max_x, :N_max_y])
        col_bar = plt.colorbar(res)
        col_bar.set_label("RMS pos err [m]")

        res = axs[1].contourf(var_ric_x[:N_max_x, :N_max_y], var_ric_y[:N_max_x, :N_max_y], rms_vel[:N_max_x, :N_max_y])
        col_bar = plt.colorbar(res)
        col_bar.set_label("RMS vel err [mm/s]")

        for ax in axs:
            ax.grid(True)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$\sigma_{r,c}^2$ [m$^2$/s$^2$]")
            ax.set_ylabel(r"$\sigma_{i}^2$ [m$^2$/s$^2$]")

        savefig("rms_res_var_2d_ric", *self.fig_dirs)


########################################################################################################################
# Plotting
########################################################################################################################


def plot_pos_errors(times, X_err_xyz, Std_xyz, title, labels_xyz=['R', 'I', 'C'], fig_dirs=[FIG_DIR], n_ignore=0):
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(3, 5), dpi=600)

    x_label, y_label, z_label = labels_xyz

    times_hr = s2hr(times)[n_ignore:]
    X_err_xyz = X_err_xyz[n_ignore:]
    Std_xyz = Std_xyz[n_ignore:]

    axs[0].set_title(title)
    axs[0].plot(times_hr, X_err_xyz[:, 0], '.k')
    axs[0].fill_between(times_hr, -3 * Std_xyz[:, 0], 3 * Std_xyz[:, 0], color='gray', alpha=0.5, label=r'$\pm3\sigma$')
    axs[0].set_ylabel(f"{x_label} error [m]")

    axs[1].plot(times_hr, X_err_xyz[:, 1], '.k')
    axs[1].fill_between(times_hr, -3 * Std_xyz[:, 1], 3 * Std_xyz[:, 1], color='gray', alpha=0.5, label=r'$\pm3\sigma$')
    axs[1].set_ylabel(f"{y_label} error [m]")

    axs[2].plot(times_hr, X_err_xyz[:, 2], '.k')
    axs[2].fill_between(times_hr, -3 * Std_xyz[:, 2], 3 * Std_xyz[:, 2], color='gray', alpha=0.5, label=r'$\pm3\sigma$')
    axs[2].set_ylabel(f"{z_label} error [m]")
    axs[2].set_xlabel("Time [hr]")

    for ax in axs:
        ax.legend()
        ax.grid(True)
        ax.yaxis.set_label_coords(-0.15, 0.5)

    fig.tight_layout()
    savefig(str_to_filename(title), *fig_dirs)
    plt.close(fig)


def plot_residuals(times, Resid, sensor_params, title, fig_dirs=[FIG_DIR], n_ignore=0):
    meas_types = sensor_params['meas_types']
    std_dict = sensor_params['sigma_dict']
    n = len(meas_types)
    times_hr = s2hr(times)
    fig, axs = plt.subplots(n, 1, figsize=(3, 5), dpi=600, sharex=True)
    for i, meas_type in enumerate(meas_types):
        if meas_type in ["el", "az"]:
            y_label = f"RMS {meas_type} resid [deg]"

            def conversion(x):
                return np.rad2deg(x)

        elif meas_type in ["dec", "ra"]:
            y_label = f"RMS {meas_type} resid ['']"

            def conversion(x):
                return rad2arcsec(x)
        else:
            y_label = f"RMS {meas_type} resid [m]"

            def conversion(x):
                return x

        resid = conversion(Resid[:, i])
        avg_resid = np.mean(resid)
        std_resid = np.std(resid)
        axs[i].plot(times_hr[n_ignore:], resid[n_ignore:], '.', color=COLOR(i))
        axs[i].axhline(
            avg_resid, color=COLOR(i), linestyle='--', label=fr"$\overline{{x}}$: {avg_resid:.2f}, $s$: {std_resid:.2f}"
        )

        if meas_type in std_dict:
            std = conversion(std_dict[meas_type])
            axs[i].axhspan(-3 * std, 3 * std, color='gray', alpha=0.5, label=r'$0\pm3\sigma$', zorder=-1)
        axs[i].legend()
        axs[i].set_ylabel(y_label)

    axs[-1].set_xlabel("Time [hr]")

    for ax in axs:
        ax.grid(True)
        ax.yaxis.set_label_coords(-0.15, 0.5)
    axs[0].set_title(title)
    fig.tight_layout()
    savefig(str_to_filename(title), *fig_dirs)
    plt.close(fig)


########################################################################################################################
# single Run
########################################################################################################################

def playground(obs_type='optical',add_q=False, perturb_x0=0, dense=False, worse_integrator=False):
    N_IGNORE = 0
    fig_dirs = [FIG_DIR, SUB_QUESTION, 'playground', obs_type]

    truth = Truth(TRUTH_FILE)
    obs_meas = Measurement(OPTICAL_FILE if obs_type == 'optical' else RADAR_FILE)

    ukf_settings = UkfSettings()
    if add_q:
        fig_dirs += ['add_q']
        ukf_settings.Q_ric = np.diag([1,1,1]) * 1e-3
        # ukf_settings.Qeci = I3 * 1e-4
        # ukf_settings.Q_ric = np.diag([1e-15, 1e-2, 1e-15])
        # ukf_settings.Q_ric = np.diag([1, 1, 1]) * 1e-10
        # ukf_settings.Q_ric = np.diag([1e-13, 1e-5, 1e-15])

        # ukf_settings.Q_ric = np.diag([1, 1, 1]) * 1e-3

        # ukf_settings.alpha = 0.01
        # ukf_settings.tudat_integrator = "rk4"
        # ukf_settings.max_step = 10

    if dense:
        fig_dirs += ['dense']
        # obs_meas.update_meas_from_dense(OPTICAL_FILE_DENSE)
        truth.make_dense(ukf_settings.get_int_params())
        obs_meas.generate_from_truth(truth)

    if perturb_x0:
        fig_dirs += [f'perturb_x0_{perturb_x0}']

    if worse_integrator:
        fig_dirs += ['worse_integrator']
        ukf_settings.tudat_integrator = "rk4"
        ukf_settings.rtol = 1e-9
        ukf_settings.atol = 1e-9

    # ukf_settings.gap_seconds=1e9
    # ukf_settings.alpha=1

    obs_ukf_raw = ukf_settings.run(obs_meas, perturb_x0=perturb_x0, verbose=False)
    obs_ukf = UkfResult(obs_ukf_raw, ukf_settings)

    obs_ukf.update_errors(truth.X_truth, n_ignore_rms=0)

    t0 = obs_ukf.times[0]
    plot_pos_errors(
        obs_ukf.times - t0,
        obs_ukf.X_err_ric,
        obs_ukf.Std_ric,
        f"Position Errors in RIC frame\n(RMS: {obs_ukf.rms_pos:.2f} m, {obs_ukf.rms_vel * 1000:.1f} mm/s)",
        fig_dirs=fig_dirs,
        n_ignore=N_IGNORE
    )
    plot_pos_errors(
        obs_ukf.times - t0,
        obs_ukf.X_err,
        obs_ukf.Std,
        f"Position Errors\n(RMS: {obs_ukf.rms_pos:.2f} m, {obs_ukf.rms_vel * 1000:.1f} mm/s)",
        labels_xyz=['X', 'Y', 'Z'],
        fig_dirs=fig_dirs,
        n_ignore=N_IGNORE
    )

    plot_residuals(
        obs_ukf.times - t0, obs_ukf.Resid, obs_meas.sensor_params, f"Residuals", fig_dirs, n_ignore=N_IGNORE
    )

    pass_indices = obs_ukf.get_indices_start_pass()
    for pass_it in range(len(pass_indices) - 1):
        pass_start = pass_indices[pass_it]
        pass_end = pass_indices[pass_it + 1]

        t_pass = obs_ukf.times[pass_start:pass_end] - t0

        X_err_ric_pass = obs_ukf.X_err_ric[pass_start:pass_end]
        X_err_pass = obs_ukf.X_err[pass_start:pass_end]
        Std_ric_pass = obs_ukf.Std_ric[pass_start:pass_end]
        Std_pass = obs_ukf.Std[pass_start:pass_end]
        Resid_pass = obs_ukf.Resid[pass_start:pass_end]

        plot_pos_errors(
            t_pass,
            X_err_ric_pass,
            Std_ric_pass,
            f"RIC position error - pass {pass_it}",
            ["R", "I", "C"],
            fig_dirs,
            n_ignore=N_IGNORE
        )
        plot_pos_errors(
            t_pass,
            X_err_pass,
            Std_pass,
            f"Position error - pass {pass_it}",
            ['X', 'Y', 'Z'],
            fig_dirs,
            n_ignore=N_IGNORE
        )
        plot_residuals(
            t_pass, Resid_pass, obs_meas.sensor_params, f"Residuals - pass {pass_it}", fig_dirs, n_ignore=N_IGNORE
        )



    print()

    plt.show()
    return


########################################################################################################################
# Main
########################################################################################################################

def run_constant_var_Qric(obs_type='optical', ):
    obs_meas = Measurement(OPTICAL_FILE if obs_type == 'optical' else RADAR_FILE)
    t_truth, X_truth, state_params = read_truth_file(os.path.join(DATA_DIR, TRUTH_FILE))

    ukf_settings = UkfSettings()
    ukf_settings.alpha = 0.01

    sub_dirs = [FIG_DIR, SUB_QUESTION, "constant_var_Qric", obs_type]
    ukfResultsCollection = UkfResultCollection([*sub_dirs])
    sigmas = np.logspace(-9, -5, 20)
    for sigma in sigmas:
        LOG(f"Running with sigma = {sigma}")
        ukf_settings.Q_ric = I3 * sigma
        obs_ukf_raw = ukf_settings.run(obs_meas, perturb_x0=False, verbose=False)
        obs_ukf = UkfResult(obs_ukf_raw, ukf_settings)
        obs_ukf.update_errors(X_truth, n_ignore_rms=0)
        ukfResultsCollection.add_result(obs_ukf)

    save_pkl("ukfResultsCollection.pkl", ukfResultsCollection, DATA_DIR, *sub_dirs)


def run_separate_Qric(obs_type="optical", sigma_min=-9, sigma_max=-5, n_sigma=5):
    obs_meas = Measurement(OPTICAL_FILE if obs_type == 'optical' else RADAR_FILE)
    t_truth, X_truth, state_params = read_truth_file(os.path.join(DATA_DIR, TRUTH_FILE))

    ukf_settings = UkfSettings()
    ukf_settings.alpha = 0.01

    sub_dirs = [FIG_DIR, SUB_QUESTION, "separate_Qric", obs_type]
    ukfResultsCollection = UkfResultCollection([*sub_dirs])
    sigmas = np.logspace(sigma_min, sigma_max, n_sigma)
    for sigma_1 in sigmas:
        for sigma_2 in sigmas:
            LOG(f"Running with sigma = {sigma_1}, {sigma_2}")
            ukf_settings.Q_ric = np.diag([sigma_1, sigma_2, sigma_1])
            # ukf_settings.Q_ric = np.diag([sigma_1, sigma_2, sigma_2])
            obs_ukf_raw = ukf_settings.run(obs_meas, perturb_x0=False, verbose=False)
            obs_ukf = UkfResult(obs_ukf_raw, ukf_settings)
            obs_ukf.update_errors(X_truth, n_ignore_rms=0)
            ukfResultsCollection.add_result(obs_ukf)

    save_pkl("ukfResultsCollection.pkl", ukfResultsCollection, DATA_DIR, *sub_dirs)


def process_constant_var_Qric(obs_type='optical'):
    sub_dirs = [FIG_DIR, SUB_QUESTION, "constant_var_Qric", obs_type]

    ukfResultsCollection = load_pkl("ukfResultsCollection.pkl", *sub_dirs)
    ukfResultsCollection.plot_rms_var_ric()


def process_separate_Qric(obs_type='optical', **kwargs):
    sub_dirs = [FIG_DIR, SUB_QUESTION, "separate_Qric", obs_type]

    ukfResultsCollection = load_pkl("ukfResultsCollection.pkl", *sub_dirs)
    ukfResultsCollection.plot_rms_var_ric2d(**kwargs)


########################################################################################################################
# Simple test
########################################################################################################################

def test():
    X1 = np.array([1, 2, 3, 4, 9, 6])
    X2 = np.array([7, 8, 9, 10, 11, 12])
    Cov = np.eye(6, 6)

    a, b = calc_ric_error(X1, X2, Cov)
    print(a)
    print(b)


if __name__ == "__main__":
    # test()

    # obs_type = 'optical'
    obs_type = 'radar'

    # playground(obs_type)
    # playground(obs_type, add_q=True)
    # playground(obs_type, worse_integrator=True)
    # playground(obs_type, add_q=True, worse_integrator=True)
    playground(obs_type, perturb_x0=3, add_q=True)
    # playground(obs_type, perturb_x0=1)

    #  run_constant_var_Qric(obs_type)
    # process_constant_var_Qric(obs_type)

    # run_separate_Qric(obs_type, -5, -3, 3)
    # run_separate_Qric(obs_type, -8, -4, 6)
    # process_separate_Qric(obs_type, N_max_x=9, N_max_y=5)

    print("\n\nDone.\n")
