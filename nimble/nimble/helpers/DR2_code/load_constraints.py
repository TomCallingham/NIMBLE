import numpy as np
import pandas as pd
import zipfile


from .desi_data_loc import base_y3kp_folder

# basepath = '/global/cfs/cdirs/desi/science/mws/y3kp/constraints/'
basepath = f"{base_y3kp_folder}/constraints/"


def load_profile(method, dataset, tracer, halo=6, view=30, rmin=0, rmax=np.inf):
    if method == "empdf":
        r, med, low, high = load_empdf_profile(dataset, tracer, halo, view)
    elif method == "gme":
        r, med, low, high = load_gme_profile(dataset, tracer, halo, view)
    elif method == "nimble":
        r, med, low, high = load_nimble_profile(dataset, tracer, halo, view)
    else:
        raise ValueError(f"method must be one of empdf, gme, nimble; got {method}")

    sel = (r >= rmin) & (r <= rmax)
    return r[sel], med[sel], low[sel], high[sel]


def load_true_profile(halo_num):
    path = "/global/homes/a/ahriley/dr2kp/true-profiles/auriga/enclosed-mass-profiles/"
    fname = path + f"halo{halo_num}-level3-enclosed-mass-profile.csv"
    data = np.loadtxt(fname, delimiter=",")
    r = data[:, 0]
    m = data[:, 1]
    return r, m


def load_empdf_profile(dataset, star_type, halo_num=6, deg=30):
    # path to relevant constraint
    path = basepath + "empdf/"
    if dataset == "auridesi":
        path += "AuriDESI/calibrated/"
    elif dataset == "loa":
        path += "Loa/"
    path += f"sampling_posterior/mass_profile/"

    # file naming convention
    if dataset == "auridesi":
        fname = f"{star_type}_H{halo_num}_{deg}deg_enclosedmass.csv"
    elif dataset == "loa":
        fname = f"{star_type}_tot_enclosedmass.csv"

    sampling_profile = pd.read_csv(path + fname)
    m_50 = sampling_profile["mass_50"].to_numpy() * 1e12
    m_16 = sampling_profile["mass_16"].to_numpy() * 1e12
    m_84 = sampling_profile["mass_84"].to_numpy() * 1e12
    r = sampling_profile["r_kpc"].to_numpy()

    return r, m_50, m_16, m_84


def load_gme_profile(dataset, tracer, halo=6, view=30):
    # path to relevant constraint
    path = basepath + "gme/"
    if dataset == "auridesi":
        zf = zipfile.ZipFile(basepath + "gme/all_CMPs_AuriDESI.zip")
        fname = f"joint_posterior_outputs/CMP_h3p-fixed_halo{halo}-view{view}-{tracer}_obs.csv"
        data = pd.read_csv(zf.open(fname))
    elif dataset == "loa":
        fname = f"CMP_{tracer}_Y3_GME_20260204.csv"
        data = pd.read_csv(path + fname)

    # NOTE: percentiles don't match emPDF exactly
    r = data["r"].to_numpy()
    med = data["q50.0"].to_numpy() * 1e12
    low = data["q12.5"].to_numpy() * 1e12
    high = data["q87.5"].to_numpy() * 1e12

    return r, med, low, high


def load_nimble_profile(dataset, tracer, halo=6, view=30):
    if tracer == "bhb":
        raise NotImplementedError("NIMBLE profiles not available for BHBs yet")

    # path to relevant constraint
    path = basepath + "nimble/"
    if dataset == "auridesi":
        path += f"auridesi/MV_RRL/H{halo:02}/deg{view}/"
    elif dataset == "loa":
        path += f"loaRRL/woSgr2Dwfs/"

    fname = f"Menc_beta_final.csv"
    data = pd.read_csv(path + fname, sep=",")

    # NOTE: not clear what percentiles are
    r = data["# r"].to_numpy()
    med = data[" Menc_med"].to_numpy()
    low = data[" Menc_low"].to_numpy()
    high = data[" Menc_upp"].to_numpy()

    return r, med, low, high
