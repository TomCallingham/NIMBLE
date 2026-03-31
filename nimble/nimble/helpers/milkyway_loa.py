import numpy as np
import agama
import astropy.units as u
from dataclasses import dataclass

from nimble.config import Config


from nimble.helpers.DR2_code.load_catalogues import load_bhbs, load_rrls


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

    def load_rrl_loa(self, apply_cuts=True):
        rrl_tab = load_rrls(get_pobs=False, apply_cuts=apply_cuts)
        rrl_dic = {key: np.asarray(rrl_tab[key]) for key in rrl_tab.columns}

        translate = {
            "distance": "dist_from_FEH",
            "distance_err": "dist_from_FEH_err",
            "vrad": "VRAD",
            "vrad_err": "VRAD_ERR",
            "pmra_err": "pmra_error",
            "pmdec_err": "pmdec_error",
            "pm_radec_corr": "pmra_pmdec_corr",
        }

        for trans_key, key in translate.items():
            rrl_dic[trans_key] = rrl_dic[key]
        return rrl_dic

    def load_bhb_loa(self, apply_cuts=True):
        RVT, FT, GT = load_bhbs(get_pobs=False, apply_cuts=apply_cuts)

        bhb_dic = (
            {key: np.asarray(val) for key, val in dict(RVT).items()}
            | {key: np.asarray(val) for key, val in dict(FT).items()}
            | {key: np.asarray(val) for key, val in dict(GT).items()}
        )

        rvs_filt = bhb_dic["RVS_WARN"] == 0
        nan_filt = (~np.isnan(bhb_dic["PMRA"])) & (~np.isnan(bhb_dic["PMDEC"]))

        print(f"rvs warn Filter:{rvs_filt.mean():.3f}, {rvs_filt.sum()} Total")

        print(f"Nan Filter:{nan_filt.mean():.3f}, {nan_filt.sum()} Total")

        filt = rvs_filt & nan_filt

        bhb_dic = {key: val[filt] for key, val in bhb_dic.items()}

        translate = {
            "distance": "dist",
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
        for trans_key, key in translate.items():
            bhb_dic[trans_key] = bhb_dic[key]

        bhb_dic["distance_err"] = bhb_dic["distance"] * 0.0565

        return bhb_dic
