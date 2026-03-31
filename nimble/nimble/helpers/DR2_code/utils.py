import numpy as np
from astropy.io import fits
import astropy.coordinates as coord
import astropy.units as u


# basepath = '/global/cfs/cdirs/desi/science/mws/y3kp/'
basepath = "~/data/observations/local_desi/y3kp/"


def get_auridesi_frame(cat_file):
    header = fits.open(cat_file)[0].header

    galcen_v_sun = [header["U_SUN"], header["VLSR"] + header["V_SUN"], header["W_SUN"]]
    zsun = header["SOLAR_H"]
    galcen_dist = -header["SOLAR_R"]

    galcen_frame = coord.Galactocentric(
        galcen_v_sun=galcen_v_sun * u.km / u.s,
        z_sun=zsun * u.Mpc,
        galcen_distance=galcen_dist * u.Mpc,
    )

    return galcen_frame


def load_true_profile(halo):
    profile_path = basepath + "true-profiles/auriga/enclosed-mass-profiles/"
    fname = profile_path + f"halo{halo}-level3-enclosed-mass-profile.csv"
    rvals, mvals = np.loadtxt(fname, delimiter=",").T
    return rvals, mvals


def subsample_ids(id_arr, n):
    if n > len(id_arr):
        raise ValueError("n must be <= the length of id_arr")

    id_arr_sort = np.sort(id_arr)
    max_id = id_arr_sort[n]

    return id_arr < max_id


def get_halo_list(suite):
    suite_list = ["auriga", "symphony", "mwest", "eden", "symphony-hr"]
    assert suite in suite_list, f"Suite must be one of {suite_list}"

    suite_dict = {
        "auriga": [6, 16, 21, 23, 24, 27],
        "symphony": [23, 268, 415, 800, 878, 937, 9749],
        "mwest": [788],
        "eden": [247, 268, 327, 349, 364, 374, 415, 490, 641, 878, 937, 9749],
        "symphony-hr": [23],
    }

    view_list = [30, 120, 210, 300] if suite == "auriga" else [0, 1, 2, 3, 4]

    return suite_dict[suite], view_list
