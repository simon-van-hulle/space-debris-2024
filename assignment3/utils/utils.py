import os
import matplotlib.pyplot as plt
import inspect
import numpy as np
from sklearn.utils import Bunch
from tudatpy.kernel.astro import element_conversion
import pickle
import re


########################################################################################################################
# Logging
########################################################################################################################

def NAME(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def LOG(*strings, separator=" "):
    message = separator.join(["[LOG]:", *strings])
    print(message)


def SECTION(name: str, filler="_", length=80):
    print(f"\n{f'{filler * 2} {name} ':{filler}<{length}s}\n")


def TITLE(name: str, filler: str = "="):
    print(f"\n{name}")
    print(filler * len(name))


def END():
    SECTION("Done")


def START():
    SECTION("Simulation Started")


def HLINE(char="_", length=80):
    print(char * length)


def LOGVAR(var, format="10.5f", space=20):
    name = NAME(var)[0]
    extra = "\n" if type(var) == np.ndarray else ""
    try:
        print(f"\t {name:{space}s} : {extra}{var:{format}}")
    except TypeError:
        print(f"\t {name:{space}s} : {extra}{var}")


LV = LOGVAR


def FILE() -> str:
    # python has a native __file__
    return inspect.currentframe().f_back.f_code.co_filename


def LINE() -> int:
    # python has no native __line__, the closest thing I could find was: sys._getframe().f_lineno
    return inspect.currentframe().f_back.f_lineno


########################################################################################################################
# Files
########################################################################################################################

def process_filename(file_path, prepend="", postpend="", default_ext=".pdf"):
    fdir, fname = os.path.split(file_path)
    name, ext = os.path.splitext(fname)

    ext = ext or default_ext
    name = prepend + name + postpend
    processed_file_path = os.path.join(fdir, name + ext)

    return processed_file_path


def str_to_filename(name):
    bracket_pattern = r"\(.*\)"
    name = re.sub(bracket_pattern, "", name)
    name = name.replace("\n", "")
    return name.replace(" ", "_").lower()


def process_dirs(*dirs):
    the_dir = "."
    for current_dir in dirs:
        the_dir = os.path.join(the_dir, current_dir)
        if not os.path.exists(the_dir):
            os.makedirs(the_dir)
    return the_dir


def list_dir(*dirs):
    the_dir = process_dirs(*dirs)
    return sorted(os.listdir(the_dir))


def save_pkl(fname, object, *dirs):
    the_dir = process_dirs(*dirs)
    filename = process_filename(fname, prepend="", postpend="", default_ext=".pkl")
    with open(os.path.join(the_dir, filename), "wb") as f:
        pickle.dump(object, f)
    LOG(f"Saved object to {the_dir}/{filename}")


def load_pkl(file_name, *dirs):
    the_dir = "."
    for current_dir in dirs:
        the_dir = os.path.join(the_dir, current_dir)
    filename = process_filename(file_name, prepend="", postpend="", default_ext=".pkl")
    with open(os.path.join(the_dir, filename), "rb") as f:
        return pickle.load(f)


def load_np_txt(data_path: str, *args):
    """Load all files in a directory"""
    files = {}
    for var in args:
        files[var] = np.loadtxt(f"{data_path}/{var}.txt")

    if len(args) == 1:
        return files[args[0]]

    return Bunch(**files)


def dump_np_txt(data_path: str, array, name):
    np.savetxt(f"{data_path}/{name}.txt", array)
    LOG(f"Dumped {name} to {data_path}/{name}.txt")


########################################################################################################################
# Figures
########################################################################################################################

def savefig(filename, *dirs, fig=None, close=True, append_size=True, tight_layout=True, **kwargs):
    """Save a figure with the given filename and directories"""
    the_dir = process_dirs(*dirs)

    fig = fig or plt.gcf()
    if tight_layout:
        fig.tight_layout()

    if append_size:
        size_x, size_y = fig.get_size_inches()
        size_str = f"_{size_x:.1f}x{size_y:>.1f}"

    filename = process_filename(filename, prepend="", postpend=size_str, default_ext=".png")

    fig.savefig(os.path.join(the_dir, filename), bbox_inches="tight", **kwargs)
    LOG(f"Saved figure to {the_dir}/{filename}")
    if close:
        plt.close(fig)


def fig_note(text, fontsize=3):
    """Add a tiny note to the current figure"""

    if plt.gca().name == "3d":
        plt.gca().text(1, 1, 1, f"Note: {text}", fontsize=fontsize, ha="right", va="bottom",
                       transform=plt.gca().transAxes)
    else:
        plt.text(1, 1, f"Note: {text}", fontsize=fontsize, ha="right", va="bottom", transform=plt.gca().transAxes)


########################################################################################################################
# Math and Data
########################################################################################################################


def s2hr(s: np.ndarray):
    return s / 3600.0


def s2day(s: np.ndarray):
    return s / 86400.0


def s2yr(s: np.ndarray):
    return s / 31557600.0


def rad2arcsec(rad: np.ndarray):
    return rad * 180 / np.pi * 3600.0


def print_matlab(arr):
    string = "[ "
    for a in arr:
        string += str(a) + " "
    string += "]"
    return string


def np2latex_table(data_array: np.ndarray) -> str:
    """Make a latex table from a 2D array"""
    table = ""
    for row in data_array:
        table += " & ".join([str(x) for x in row])
        table += "\\\\ \n"
    return table


########################################################################################################################
# Vector and Matrix
########################################################################################################################


def col_vec(array, rows_required=None):
    if len(array.shape) == 1:
        array = array.reshape(-1, 1)

    if rows_required is not None and array.shape[0] != rows_required:
        raise ValueError(f"Array must have {rows_required} rows")

    return array


###############################################################################
# Aliases for Tudat
###############################################################################

MU_EARTH = 3.986004415e14


def kep2cart(*args, **kwargs):
    return element_conversion.keplerian_to_cartesian(*args, **kwargs, gravitational_parameter=MU_EARTH)


def cart2kep(*args, **kwargs):
    return element_conversion.cartesian_to_keplerian(*args, **kwargs, gravitational_parameter=MU_EARTH)
