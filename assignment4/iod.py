"""
Initial Orbit Determination
"""


####################################################################################################
# Imports
####################################################################################################
import numpy as np
import os
from datetime import datetime
from sutils.utils import *
from sutils.logging import *
from sutils.style import *

####################################################################################################
# Constants
####################################################################################################

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = f"{FILE_DIR}/data/group4"

IOD_MEAS_FILENAME = "q3_meas_iod_99004.pkl"
IOD_MEAS_PATH = os.path.join(DATA_DIR, IOD_MEAS_FILENAME)

####################################################################################################
# Main
####################################################################################################

if __name__ == "__main__":
    TITLE("Assignment 4: Initial Orbit Determination")

    d = load_pkl(IOD_MEAS_PATH)









