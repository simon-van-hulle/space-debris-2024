"""
Q4 as long as Kyle doesn't respond
"""

###############################################################################
# Imports
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys


from datetime import datetime, timedelta

from tudatpy.astro.two_body_dynamics import LambertTargeterIzzo, propagate_kepler_orbit
from tudatpy.interface import spice
from tudatpy import constants
from tudatpy.astro import element_conversion
from tudatpy.numerical_simulation import environment
from ConjunctionUtilities import compute_TCA
from TudatPropagator import propagate_orbit

from sutils.utils import *
from sutils.logging import *
from sutils.style import *

from scipy.stats import gaussian_kde
import seaborn as sns
from iod import RsoFile, get_inertial_to_ric_matrix




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

from ukf_tuning import UkfResult, UkfSettings


###############################################################################
# Dataclasses
###############################################################################





###############################################################################
# RMS
###############################################################################
FIGDIRS = [FIG_DIR, "q4"]
FILENAMES_RSO = [
    "q3_meas_rso_99004.pkl",
    "q4_meas_rso_91104.pkl",
    "q4_meas_rso_91368.pkl",
    "q4_meas_rso_91401.pkl",
    "q4_meas_rso_91438.pkl",   
    "q4_meas_rso_91447.pkl",   
    "q4_meas_rso_91518.pkl",   
    "q4_meas_rso_91714.pkl",   
    "q4_meas_rso_91787.pkl",   
    "q4_meas_rso_91821.pkl",   
    "q4_meas_rso_91861.pkl",   
]


UKF_SETTINGS = UkfSettings()


def rms():
    
    # norad_vals = []
    # rms_vals = []
    
    # alpha_resid_hist = []
    # dec_resid_hist = []
    
    norads = []
    results = []
    
    for fname in FILENAMES_RSO:
        rso  = RsoFile(os.path.join(DATA_DIR, fname))
        
        result = UkfResult(UKF_SETTINGS.run(rso, verbose=False), UKF_SETTINGS)
        resid_rms = np.sqrt(np.mean(result.Resid ** 2, axis=0))
        
        result.update_errors(result.X_est)
                
        NORAD = fname.split("_")[3].split(".")[-2]
        # rms_vals.append(resid_rms)
        # alpha_resid_hist.append(result.Resid[:, 0].flatten())
        # dec_resid_hist.append(result.Resid[:, 1].flatten())
        norads.append(NORAD)
        results.append(result)
        
        
    save_pkl("q4_rms_2.pkl", [norads, results], CACHE_DIR)
    
def process_rms():
    norads, results = load_pkl("q4_rms_2.pkl", CACHE_DIR)
    
    ra_resids = []
    dec_resids = []

    rms_vals = []
    
    
    for res in results:
        ra_resids.append(res.Resid[:, 0].flatten())
        dec_resids.append(res.Resid[:, 1].flatten())
        
    
    abs_alpha_errors_arcsec = [np.abs(np.rad2deg(r)) for r in ra_resids]
    abs_dec_errors_arcsec = [np.abs(np.rad2deg(r)) for r in dec_resids]
    
    
    for i, rms in zip(norads, rms_vals):
        print(f"NORAD {i} RMS: {rms}")
    
    plt.figure(figsize=(3.5, 2.5))
    print(norads, abs_alpha_errors_arcsec)
    sns.boxplot(abs_alpha_errors_arcsec)
    plt.xticks(ticks=range(len(norads)), labels=norads, rotation=45)
    plt.yscale("log")
    plt.grid(True, zorder=-10)
    plt.ylabel(r"Abs. resid. $\alpha$ [deg]")
    plt.xlabel("NORAD ID")
    savefig("q4_resid_alpha_boxplot", *FIGDIRS) 
    
    
    plt.figure(figsize=(3.5, 2.5))
    sns.boxplot(abs_dec_errors_arcsec)
    plt.xticks(ticks=range(len(norads)), labels=norads,  rotation=45)
    plt.yscale("log")
    plt.grid(True, zorder=-10)
    plt.ylabel(r"Abs. resid. $\delta$ [deg]")
    plt.xlabel("NORAD ID")
    savefig("q4_resid_delta_boxplot", *FIGDIRS)
    
    for norad, res in zip(norads,results):
        print(res.times.shape, res.Std_ric.shape)      
        
        fig, axs = plt.subplots(3, 1, figsize=(3.5, 2.5), sharex=True)
    
        time_hr = (res.times - res.times[0]) / 3600
        
        ax = axs[0]
        axs[0].fill_between(time_hr, -res.Std_ric[:, 0]*3, res.Std_ric[:, 0]*3, alpha=0.5)
        axs[0].set_ylabel(r"$3\sigma$ $R$ [m]")
        axs[1].fill_between(time_hr, -res.Std_ric[:, 1]*3, res.Std_ric[:, 1]*3, alpha=0.5, label=r"$I$")
        axs[1].set_ylabel(r"$3\sigma$ $I$ [m]")
        axs[2].fill_between(time_hr, -res.Std_ric[:, 2]*3, res.Std_ric[:, 2]*3, alpha=0.5, label=r"$C$")
        axs[2].set_ylabel(r"$3\sigma$ $C$ [m]")
        # plt.plot(time_hr, res.Std_ric[:, 0]*3, label=r"$R$")
        # plt.plot(time_hr, res.Std_ric[:, 1]*3, label=r"$I$")
        # plt.plot(time_hr, res.Std_ric[:, 2]*3, label=r"$C$")
        axs[0].set_title(f"NORAD ID: {norad}")
        axs[2].set_xlabel("Time [hr]")
        savefig(f"q4_ric_std_{norad}", *FIGDIRS)
    



###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    # rms()
    process_rms()


    plt.show()
