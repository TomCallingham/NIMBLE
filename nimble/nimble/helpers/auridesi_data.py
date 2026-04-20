from typing import Literal
import numpy as np
import agama
import astropy
import astropy.units as u
from dataclasses import dataclass
from ..config import Config
from .utils import h5py_load, robust_table_to_numpy
from functools import cached_property
from ..jeans import spline_d_log_dens_convert

from .DR2_code.load_catalogues import load_auridesi

y3kp_folder = "/home/tcallingh/data/observations/local_desi/y3kp/"
extra_y3kp_folder = "/home/tcallingh/data/observations/local_desi/callingham_new2/y3kp/"
# extra_y3kp_folder = "/home/tcallingh/data/observations/local_desi/callingham_new2/y3kp/auridesirrl_true_profiles/

au_halos = [6, 16, 21, 23, 24, 27]
au_angles = [30, 120, 210, 300]


@dataclass(init=False)
class AuriDesiData:
    halo_n: int
    angle: int
    figs_root: str
    true_path: str

    def __init__(self, halo_n=6, angle=30):
        assert halo_n in au_halos
        assert angle in au_angles

        self.halo_n = halo_n
        self.angle = angle

        print("Halo:", halo_n, "Angle:", angle)

        self.figs_root = f"results/auridesi/deconv_{halo_n}_{angle}/"
        self.true_path = f"data/auriga/H{halo_n}/Au{halo_n}_true.csv"

    def create_config(self):
        cfg = Config()
        cfg.min_r = 1.0
        cfg.max_r = 100.0
        cfg.num_knots = 5
        cfg.figs_root = self.figs_root
        return cfg

    @cached_property
    def solar_params(self):
        return self.load_solar_params()

    def load_solar_params(self) -> dict:
        halo_n, angle = self.halo_n, self.angle
        rrl_desi_fname = f"{y3kp_folder}/field-stars/mocks/auridesi/latest/halo{halo_n}/halo{halo_n}-view{angle}-rrl.fits"

        with astropy.io.fits.open(rrl_desi_fname) as AllTab:
            header = AllTab[0].header
            solar_params = {
                "galcen_vsun": np.asarray(
                    [
                        header["U_SUN"],
                        (header["VLSR"] + header["V_SUN"]),
                        header["W_SUN"],
                    ]
                )
                * u.km
                / u.s,
                "galcen_pos": np.asarray([header["SOLAR_R"], 0.0, header["SOLAR_H"]])
                * u.Mpc.to(u.kpc)
                * u.kpc,
            }
        solar_params["lsr_info"] = (
            -solar_params["galcen_pos"][0].value,
            solar_params["galcen_vsun"].value,
            solar_params["galcen_pos"][2].value,
        )
        return solar_params

    # def load_true_dens_fits(self):
    #     load_fname = f"{extra_y3kp_folder}/auridesirrl_true_profiles/auridesirrl_true_profiles_angle{self.angle}_{self.halo_n}_splinefit.hdf5"
    #     all_fits = h5py_load(load_fname)
    #     return all_fits
    def _load_true_dens_fits(self):
        load_fname = f"{extra_y3kp_folder}/rematchedAurigaTrueProfiles/Au{self.halo_n}Lv3_AccSplineProfiles.hdf5"
        # auridesirrl_true_profiles/auridesirrl_true_profiles_angle{self.angle}_{self.halo_n}_splinefit.hdf5"
        new_fits = h5py_load(load_fname)
        rrl_log_dens_params = new_fits["slice_rematched"]["rrl"][self.angle][
            "params_logrho"
        ]
        rrl_dens_knots_logr = new_fits["slice_rematched"]["rrl"][self.angle][
            "knots_logr"
        ]
        rrl_func_log_dens = agama.Spline(rrl_dens_knots_logr, rrl_log_dens_params)
        rrl_func_d_log_dens = spline_d_log_dens_convert(rrl_func_log_dens)

        bhb_log_dens_params = new_fits["slice_rematched"]["bhb"][self.angle][
            "params_logrho"
        ]
        bhb_dens_knots_logr = new_fits["slice_rematched"]["bhb"][self.angle][
            "knots_logr"
        ]
        bhb_func_log_dens = agama.Spline(bhb_dens_knots_logr, bhb_log_dens_params)
        bhb_func_d_log_dens = spline_d_log_dens_convert(bhb_func_log_dens)

        self.rrl_log_dens_func = rrl_func_log_dens
        self.rrl_d_log_dens_func = rrl_func_d_log_dens

        self.bhb_log_dens_func = bhb_func_log_dens
        self.bhb_d_log_dens_func = bhb_func_d_log_dens
        return

    def load_true_Menc(self):
        enc_fname = f"{y3kp_folder}/true-profiles/auriga/enclosed-mass-profiles/halo{self.halo_n}-level3-enclosed-mass-profile.csv"
        arr = np.loadtxt(enc_fname, delimiter=",")
        r_true = arr[:, 0]
        Menc_true = arr[:, 1]
        return r_true, Menc_true

    def load_rrl(self):
        return self.load_tracers("rrl")

    def load_bhb(self):
        return self.load_tracers("bhb")

    def load_tracers(self, tracer: Literal["rrl", "bhb"]):
        obs_tab, true_tab, _galcen_frame = load_auridesi(
            self.halo_n, view=self.angle, tracer=tracer
        )
        # print("galcen_frame")
        # print(galcen_frame)
        obs_data = robust_table_to_numpy(obs_tab)
        true_data = robust_table_to_numpy(true_tab)

        if tracer == "rrl":
            au_desi_obs_rename = {"v0": "vrad", "v0_err": "vrad_err"}
            for key, val in au_desi_obs_rename.items():
                obs_data[val] = obs_data[key]
                # true_data[val] = true_data[key]
        return obs_data, true_data
