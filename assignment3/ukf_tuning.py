import sys

import matplotlib.pyplot as plt

from utils.style import *
from utils.utils import *
import pickle
import os
from EstimationUtilities import *
from assignment2.BreakupUtilities import *
from tudatpy.astro import frame_conversion
from sklearn.utils import Bunch
from dataclasses import dataclass
import time

DATA_DIR = "data/group4"
RADAR_FILE = "group4_q2a_gps_sparse_radar_meas.pkl"
OPTICAL_FILE = "group4_q2a_gps_sparse_optical_meas.pkl"
TRUTH_FILE = "group4_q2a_gps_sparse_truth_grav.pkl"
FIG_DIR = "figures"

I3 = np.eye(3, 3)
I6 = np.eye(6, 6)


def tudat_initialize_bodies():
    # Load spice kernels
    spice_interface.load_standard_kernels()

    # Define string names for bodies to be created from default.
    bodies_to_create = ["Sun", "Earth", "Moon"]

    # Use "Earth"/"J2000" as global frame origin and orientation.
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"

    # Create default body settings, usually from `spice`.
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation)

    # Create system of selected celestial bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    return bodies


def get_ric_error(X_est, X_true, covar_ric=I6):
    Rot = frame_conversion.inertial_to_rsw_rotation_matrix(X_true)
    Cov_ric = Rot @ covar_ric[:3, :3] @ Rot.T

    X_err_ric = Rot @ X_est[:3] - Rot @ X_true[:3]
    return X_err_ric, Cov_ric


def get_ric_errors(X_est, X_true, Cov_eci):
    errs_ric = np.zeros_like(X_est)[:, :3]
    Cov_ric = np.zeros_like(Cov_eci)[:, :3, :3]
    for i in range(len(X_est)):
        err_ric, covar_ric = get_ric_error(X_est[i], X_true[i], Cov_eci[i])
        errs_ric[i] = err_ric
        Cov_ric[i] = covar_ric

    return errs_ric, Cov_ric


class Measurement:
    def __init__(self, filename, directory=DATA_DIR):
        self.filename = os.path.join(directory, filename)
        self.state_params, self.meas_dict, self.sensor_params = read_measurement_file(self.filename)


class UkfSettings:
    def __init__(self):
        self.Qeci = I3 * 0
        self.Qric = I3 * 0
        self.alpha = 0.1
        self.gap_seconds = 600

        self.tudat_integrator = "rkf78"
        self.step = 10.
        self.max_step = 1000.
        self.min_step = 1e-3
        self.rtol = 1e-12
        self.atol = 1e-12

    def get_filter_params(self):
        return {"Qeci": self.Qeci, "Qric": self.Qric, "alpha": self.alpha, "gap_seconds": self.gap_seconds}

    def get_int_params(self):
        return {"tudat_integrator": self.tudat_integrator, "step": self.step, "max_step": self.max_step,
                "min_step": self.min_step,
                "rtol": self.rtol, "atol": self.atol}

    def run(self, measurements: Measurement, **kwargs):
        bodies = tudat_initialize_bodies()
        return ukf(measurements.state_params, measurements.meas_dict, measurements.sensor_params, self.get_int_params(),
                   self.get_filter_params(), bodies, **kwargs)


class UkfResult:
    def __init__(self, ukf_output: dict, filter_settings: UkfSettings):
        self.times = np.array(list(ukf_output.keys()))
        self.X_est = np.array([ukf_output[t]["state"] for t in self.times]).reshape(-1, 6)
        self.Cov = np.array([ukf_output[t]["covar"] for t in self.times])
        n_resids = ukf_output[self.times[0]]["resids"].shape[0]
        self.resids = np.array([ukf_output[t]["resids"] for t in self.times]).reshape(-1, n_resids)
        self.time_step = self.times[1] - self.times[0]
        self.filter_settings = filter_settings
        self.stds_ric = None
        self.X_err = None
        self.rms_pos = None

    def indices_per_pass(self):
        pass_indices = []
        for i in range(1, len(self.times)):
            if self.times[i] - self.times[i - 1] > 100 * self.time_step:
                pass_indices.append(i)
        pass_indices.append(i)

        return pass_indices

    def update_errors(self, X_truth, n_ignore_rms=0):
        self.X_err = self.X_est - X_truth
        self.stds = stds_from_Cov(self.Cov)

        self.X_err_ric, self.Cov_ric = get_ric_errors(self.X_est, X_truth, self.Cov)
        self.stds_ric = stds_from_Cov(self.Cov_ric)

        self.rms_pos = np.sqrt(np.mean(np.linalg.norm(self.X_err[n_ignore_rms:, :3], axis=1) ** 2, axis=0))
        self.rms_vel = np.sqrt(np.mean(np.linalg.norm(self.X_err[n_ignore_rms:, 3:], axis=1) ** 2, axis=0))
        self.rms_resids = np.sqrt(np.mean(self.resids ** 2, axis=0))


def plot_arr_and_3sigma(times, y_array, y_stds, labels=None, fig=None, fmt="none", plot_3sigma=True, *args, **kwargs):
    dimensions = y_array.shape[1]

    if fig is None:
        fig, axs = plt.subplots(dimensions, 1, sharex=True, figsize=(6, 1.5 * dimensions))
    else:
        axs = fig.get_axes()

    for i, ax in enumerate(axs):
        ax.plot(times, y_array[:, i], '.', label=labels[i] if labels is not None else None, *args, **kwargs)
        if plot_3sigma:
            ax.fill_between(times, y_array[:, i] - 3 * y_stds[:, i], y_array[:, i] + 3 * y_stds[:, i], alpha=0.5, *args,
                            **kwargs)
        ax.legend()


def stds_from_Cov(Cov):
    return np.sqrt(Cov.diagonal(axis1=1, axis2=2))


def last_idx(*dirs):
    return len(list_dir(*dirs)) - 1


########################################################################################################################
# Main stuff
########################################################################################################################

def calc_var_eci(obs_typ="radar", VAR_MIN_LOG=-7, VAR_MAX_LOG=-5, N_VAR=10):
    var_eci_list = np.logspace(VAR_MIN_LOG, VAR_MAX_LOG, N_VAR)

    # First try it out for the radar measurements
    if obs_typ == "radar":
        obs_meas = Measurement(RADAR_FILE)
    elif obs_typ == "optical":
        obs_meas = Measurement(OPTICAL_FILE)
    else:
        raise ValueError(f"Unknown observation type: {obs_typ}")

    ukf_settings = UkfSettings()

    for i, var_eci in enumerate(var_eci_list):
        print(f"Running for var_eci = {var_eci:.0e}")

        ukf_settings.Qeci = I3 * var_eci

        obs_ukf_raw = ukf_settings.run(obs_meas, verbose=False)
        obs_ukf = UkfResult(obs_ukf_raw, ukf_settings)

        current_time = time.strftime("%Y%m%d_%H%M%S")
        savepkl(f"ukf_{current_time}.pkl", obs_ukf, "output", obs_typ)


def process_single_run(ukf_res: UkfResult, t_truth, X_truth, state_params, n_ignore_rms=0, plot=True, subdirs=[],
                       obs_typ="radar", savefigs=True):
    times = ukf_res.times - ukf_res.times[0]
    times_hr = times / 3600

    ukf_res.update_errors(X_truth, n_ignore_rms)

    if not plot:
        return

    ####################################################################################################################
    # Plotting stuff for a single ukf run
    ####################################################################################################################

    var_eci = ukf_res.filter_settings.Qeci[0, 0]
    subdirs = [f"var_{var_eci:.0e}"] + subdirs

    plt.figure()
    plot_arr_and_3sigma(times_hr, ukf_res.X_err_ric[:, :3] / 1000, ukf_res.stds_ric[:, :3], plot_3sigma=True,
                        labels=["RA", "DEC", "Range"] if obs_typ == "radar" else ["RA", "DEC"])
    fig = plt.gcf()
    plt.xlabel("Time [hr]")
    fig.supylabel("Position Error [km]")
    if savefigs:
        savefig(f"{obs_typ}_resids", FIG_DIR, "q2", *subdirs)

    # Residuals for all the passes
    pass_indices = ukf_res.indices_per_pass()
    for i in range(len(pass_indices) - 1):
        plt.figure()
        times_hr_pass = times_hr[pass_indices[i]:pass_indices[i + 1]]
        X_err_pass = ukf_res.X_err_ric[pass_indices[i]:pass_indices[i + 1]]
        stds_pass = ukf_res.stds_ric[pass_indices[i]:pass_indices[i + 1]]
        resids_pass = ukf_res.resids[pass_indices[i]:pass_indices[i + 1]]

        plot_arr_and_3sigma(times_hr_pass, X_err_pass / 1000, stds_pass, plot_3sigma=True, fmt=".",
                            labels=["R", "C", "I"])
        fig = plt.gcf()
        plt.xlabel("Time[hr]")
        fig.supylabel("Position error [km]")

        if savefigs:
            savefig(f"{obs_typ}_err_pass_{i}", FIG_DIR, "q2", *subdirs)

    # Plot errors
    plt.figure()
    plot_arr_and_3sigma(times_hr, ukf_res.X_err_ric[:, :3], ukf_res.stds_ric[:, :3], labels=["R", "C", "I"])
    fig = plt.gcf()
    plt.xlabel("Time [hr]")
    fig.supylabel("Position Error [m]")
    if savefigs:
        savefig(f"{obs_typ}_err", FIG_DIR, "q2", *subdirs)


def process_2a(obs_typ="radar"):
    condition = lambda ukf_result: ukf_result.filter_settings.Qeci[0, 0] < 1e-4

    t_truth, X_truth, state_params = read_truth_file(os.path.join(DATA_DIR, TRUTH_FILE))

    plt.figure()
    plt.plot(t_truth, X_truth[:, 0], '.-', label="X")
    plt.plot(t_truth, X_truth[:, 1], '.-', label="Y")
    plt.plot(t_truth, X_truth[:, 2], '.-', label="Z")
    plt.legend()
    savefig("truth.pdf", FIG_DIR, "q2")

    rms_pos_hist = []
    rms_vel_hist = []
    rms_resids_hist = []
    var_eci_list = []

    for i, fname in enumerate(list_dir("output", obs_typ)):
        print(f"Processing for {fname}")
        try:
            obs_ukf = loadpkl(fname, "output", obs_typ)
        except:
            print(f"Skipping {fname}")
            continue

        if condition(obs_ukf):
            process_single_run(obs_ukf, t_truth, X_truth, state_params, n_ignore_rms=20, plot=True, obs_typ=obs_typ)
            var_eci = obs_ukf.filter_settings.Qeci[0, 0]
            rms_pos_hist.append(obs_ukf.rms_pos)
            rms_vel_hist.append(obs_ukf.rms_vel)
            rms_resids_hist.append(obs_ukf.rms_resids)
            var_eci_list.append(var_eci)
        else:
            print(f"Not using {fname}")

    sort = np.argsort(var_eci_list)
    var_eci_list = np.array(var_eci_list)[sort]
    rms_pos_hist = np.array(rms_pos_hist)[sort]
    rms_vel_hist = np.array(rms_vel_hist)[sort]
    rms_resids_hist = np.array(rms_resids_hist)[sort]

    # plt.figure(figsize=(6,2))
    fig, axs = plt.subplots(3, 1, figsize=(6, 3 * 2))
    ax = axs[0]
    ax.plot(var_eci_list, rms_pos_hist, '.k-')
    ax.set_xscale("log")
    ax.set_ylabel("RMS Position Error [m]")

    ax = axs[1]
    ax.plot(var_eci_list, rms_vel_hist, '.k-')
    ax.set_xscale("log")
    ax.set_ylabel("RMS Velocity Error [m/s]")

    ax = axs[2]
    ax.plot(var_eci_list, rms_resids_hist[:, 0], '.-', label="RA")
    ax.plot(var_eci_list, rms_resids_hist[:, 1], '.-', label="DEC")
    if obs_typ == "radar":
        ax.plot(var_eci_list, rms_resids_hist[:, 2], '.-', label="Range")
    ax.set_xscale("log")
    ax.set_xlabel("Process Noise Variance [$m^2/s^4$]")
    ax.set_ylabel("Angle Resid [rad]")
    ax.legend()

    for ax in axs:
        ax.yaxis.set_label_coords(-0.1, 0.5)

    fig.tight_layout()
    savefig(f"rms_resids.pdf", FIG_DIR, "q2", obs_typ)

    return


def optimise_2a(obs_typ="radar"):
    t_truth, X_truth, state_params = read_truth_file(os.path.join(DATA_DIR, TRUTH_FILE))
    if obs_typ == "radar":
        obs_meas = Measurement(RADAR_FILE)
    elif obs_typ == "optical":
        obs_meas = Measurement(OPTICAL_FILE)
    else:
        raise ValueError(f"Unknown observation type: {obs_typ}")

    ukf_settings = UkfSettings()

    std_eci_hist = []
    rms_pos_hist = []
    rms_vel_hist = []

    std_eci_step = 1e-5
    std_eci = 1e-4

    iter = 0
    while True:
        ukf_settings.Qeci = I3 * std_eci
        obs_ukf_raw = ukf_settings.run(obs_meas, verbose=False)
        obs_ukf = UkfResult(obs_ukf_raw, ukf_settings)

        obs_ukf.update_errors(X_truth, n_ignore_rms=20)

        std_eci_hist.append(std_eci)
        rms_pos_hist.append(obs_ukf.rms_pos)
        rms_vel_hist.append(obs_ukf.rms_vel)
        print(f"std_eci: {std_eci:.2e}, RMS Pos: {obs_ukf.rms_pos:.2f}, RMS Vel: {obs_ukf.rms_vel:.4f}")

        if len(rms_pos_hist) > 2:
            deriv = (rms_vel_hist[-1] - rms_vel_hist[-2]) / (std_eci - std_eci_hist[-2])
            newton_corrector = rms_vel_hist[-1] / deriv

        else:
            newton_corrector = std_eci_step

        print(f"Newton corrector: {newton_corrector:.2e}")
        std_eci = std_eci - newton_corrector * 0.01

        iter += 1
        if iter > 100:
            break

    return


def playground():
    obs_meas = Measurement(RADAR_FILE)
    ukf_settings = UkfSettings()
    ukf_settings.Qeci = I3 * 1e-4
    obs_ukf_raw = ukf_settings.run(obs_meas, verbose=False)
    obs_ukf = UkfResult(obs_ukf_raw, ukf_settings)

    t_truth, X_truth, state_params = read_truth_file(os.path.join(DATA_DIR, TRUTH_FILE))
    process_single_run(obs_ukf, t_truth, X_truth, state_params, n_ignore_rms=20, plot=True, obs_typ="radar",
                       savefigs=False)
    plt.show()
    return


if __name__ == "__main__":
    # calc_2a('optical')
    # calc_2a('radar')
    # process_2a('optical')
    # process_2a('radar')
    # optimise_2a('optical')
    playground()

    pass
