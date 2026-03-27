import astropy.table as atpy
import numpy as np
import agama
import astropy.units as u
from dataclasses import dataclass
from .sag_select import select_Sgr

# from ..config import Config
# from .utils import load_cols_from_fits
from ..config import Config
import pandas as pd

y3kp_folder = "/home/tcallingh/data/observations/local_desi/y3kp"
# loa_rrl_file = f"{y3kp_folder}/field-stars/loa/loa-rrl-latest.csv"
loa_rrl_file = f"{y3kp_folder}/field-stars/loa/loa-rrl_mass_modeling_20260223.csv"
loa_bhb_file = f"{y3kp_folder}/field-stars/loa/loa-bhb-latest.fits"

extra_y3kp_folder = "/home/tcallingh/data/observations/local_desi/callingham_new2/y3kp/"
rrl_flag_fname = f"{extra_y3kp_folder}/loa/loa-rrl-latest-flags.csv"


solar_params0 = {
    "vlsr": 232.8,
    "R0": 8.20,
    "U": 11.1,
    "V": 12.24,
    "W": 7.25,
    "z_sun": 0.02,
}

sun_vel = np.array(
    [solar_params0["U"], solar_params0["V"] + solar_params0["vlsr"], solar_params0["W"]]
) * (u.km / u.s)
sun_pos = np.array([-solar_params0["R0"], 0, solar_params0["z_sun"]]) * u.kpc

solar_params = {"galcen_vsun": sun_vel, "galcen_pos": sun_pos}

import pandas as pd

y3kp_folder = "/home/tcallingh/data/observations/local_desi/y3kp"
loa_rrl_file = f"{y3kp_folder}/field-stars/loa/loa-rrl_mass_modeling_20260223.csv"

old_loa_rrl_file = f"{y3kp_folder}/field-stars/loa/loa-rrl-latest.csv"


@dataclass(init=False)
class MWLoaData:
    figs_root: str
    # true_path: str

    def __init__(self):
        self.figs_root = "results/mw_loa/"
        self.solar_params = solar_params

    def create_config(self):
        cfg = Config()
        cfg.min_r = 1.0
        cfg.max_r = 100.0
        cfg.num_knots = 5
        cfg.figs_root = self.figs_root
        # cfg.true_path = self.true_path
        # cfg.lsr_info = self.solar_params["lsr_info"]
        return cfg

    def load_rrl_loa(
        self, all_cols=False, filter_sag=True, filter_dwarfs=True, filter_gcs=True
    ):

        if not filter_sag & filter_dwarfs & filter_gcs:
            raise AssertionError("Using new Gustavo RRL, all filters already applied!")

        dfp = pd.read_csv(loa_rrl_file)
        old_dfp = pd.read_csv(old_loa_rrl_file)
        dfp = dfp.join(
            old_dfp.set_index("source_id")["pmra_pmdec_corr"], on="source_id"
        )

        translate = {
            "distance": "dist_FEH",
            "distance_err": "dist_FEH_err",
            "vrad": "VRAD",
            "vrad_err": "VRAD_ERR",
            "pmra_err": "pmra_error",
            "pmdec_err": "pmdec_error",
            "pm_radec_corr": "pmra_pmdec_corr",
        }
        min_keys = [
            "ra",
            "dec",
            "pmra",
            "pmdec",
            "distance",
            "vrad",
            "pmra_err",
            "pmdec_err",
            "distance_err",
            "vrad_err",
            "pm_radec_corr",
        ]

        data_dic = {key: dfp[translate.get(key, key)].to_numpy() for key in min_keys}

        if all_cols:
            cols = dfp.columns
            data_dic = data_dic | {key: dfp[key].to_numpy() for key in cols}
        return data_dic

    def load_bhb_loa(self, all_cols=False, substructure_flags=True):
        RVT, FT, GT = load_BHB_data(substructure_flags=substructure_flags)
        data_dict = (
            {key: np.asarray(val) for key, val in dict(RVT).items()}
            | {key: np.asarray(val) for key, val in dict(FT).items()}
            | {key: np.asarray(val) for key, val in dict(GT).items()}
        )

        filt = data_dict["RVS_WARN"] == 0
        data_dict = {key: val[filt] for key, val in data_dict.items()}

        translate = {
            "distance": "dist",  # "distance_err":"dist_FEH_err",
            "vrad": "VRAD",
            "vrad_err": "VRAD_ERR",
            "pmra_err": "PMRA_ERROR",
            "pmdec_err": "PMDEC_ERROR",
            "pm_radec_corr": "PMRA_PMDEC_CORR",
            "ra": "RA",
            "dec": "DEC",
            "pmra": "PMRA",
            "pmdec": "PMDEC",
        }
        min_keys = [
            "ra",
            "dec",
            "pmra",
            "pmdec",
            "distance",
            "vrad",
            "pmra_err",
            "pmdec_err",
            "pm_radec_corr",
            "vrad_err",
        ]

        bhb_data = {key: data_dict[translate.get(key, key)] for key in min_keys}

        bhb_data["distance_err"] = bhb_data["distance"] * 0.0565

        bhb_data = {key: val.astype(float) for key, val in bhb_data.items()}

        if all_cols:
            bhb_data = data_dict | bhb_data

        nan_filt = (~np.isnan(bhb_data["pmra"])) & (~np.isnan(bhb_data["pmdec"]))
        print(f"Nan Filter:{nan_filt.mean():.3f}, {nan_filt.sum()} Total")
        bhb_data = {key: val[nan_filt] for key, val in bhb_data.items()}

        return bhb_data


def load_BHB_data(substructure_flags=True, main_bright=True, redrock_star=True):
    """This function selects the BHBs from the Loa catalogue that are used for the DR2 KP.
    Some quality flags, like RVS_WARN==0 are already applied. See Bystrom25 for details
    """

    RVT = atpy.Table().read(loa_bhb_file, hdu="RVTAB")
    GT = atpy.Table().read(loa_bhb_file, hdu="GAIA")
    FT = atpy.Table().read(loa_bhb_file, hdu="FIBERMAP")

    survey_idx = (RVT["SURVEY"] == "main") & (RVT["PROGRAM"] == "bright")
    halo_idx = ~(FT["Sgr_flag"] | FT["GC_flag"] | FT["dwarf_flag"])
    primary_idx = RVT["PRIMARY"] == True

    sel = primary_idx
    if substructure_flags:
        sel &= halo_idx
    if main_bright:
        sel &= survey_idx
    if redrock_star:  # this selection has a slightly higher quality, and in my BHB paper, I don't use the Redrock criterion
        sel &= RVT["RR_SPECTYPE"] == "STAR"

    return RVT[sel], FT[sel], GT[sel]
