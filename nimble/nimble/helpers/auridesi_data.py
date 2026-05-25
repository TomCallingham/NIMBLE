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
    # def _load_true_dens_fits(self):
    #     load_fname = f"{extra_y3kp_folder}/rematchedAurigaTrueProfiles/Au{self.halo_n}Lv3_AccSplineProfiles.hdf5"
    #     # auridesirrl_true_profiles/auridesirrl_true_profiles_angle{self.angle}_{self.halo_n}_splinefit.hdf5"
    #     new_fits = h5py_load(load_fname)
    #     rrl_log_dens_params = new_fits["slice_rematched"]["rrl"][self.angle][
    #         "params_logrho"
    #     ]
    #     rrl_dens_knots_logr = new_fits["slice_rematched"]["rrl"][self.angle][
    #         "knots_logr"
    #     ]
    #     rrl_func_log_dens = agama.Spline(rrl_dens_knots_logr, rrl_log_dens_params)
    #     rrl_func_d_log_dens = spline_d_log_dens_convert(rrl_func_log_dens)

    #     bhb_log_dens_params = new_fits["slice_rematched"]["bhb"][self.angle][
    #         "params_logrho"
    #     ]
    #     bhb_dens_knots_logr = new_fits["slice_rematched"]["bhb"][self.angle][
    #         "knots_logr"
    #     ]
    #     bhb_func_log_dens = agama.Spline(bhb_dens_knots_logr, bhb_log_dens_params)
    #     bhb_func_d_log_dens = spline_d_log_dens_convert(bhb_func_log_dens)

    #     self.rrl_log_dens_func = rrl_func_log_dens
    #     self.rrl_d_log_dens_func = rrl_func_d_log_dens

    #     self.bhb_log_dens_func = bhb_func_log_dens
    #     self.bhb_d_log_dens_func = bhb_func_d_log_dens
    #     return

    def _load_true_dens_fits(self):
        print("NEW SPLINE NUMBER DENSITY")

        rrl_spl_fname = f"/home/tcallingh/Projects/DesiDR2_Nimble/1_Data/AuriDesi/FittingNumberDensity/SavedNumDensSpline/Au{self.halo_n}_angle_{self.angle}.npz"
        rrl_SplFit = SplineNumberDensity.load(rrl_spl_fname)
        self.rrl_log_dens_func = rrl_SplFit.log_rho_from_logr
        self.rrl_d_log_dens_func = rrl_SplFit.dlnrho_from_logr

        bhb_spl_fname = f"/home/tcallingh/Projects/DesiDR2_Nimble/1_Data/AuriDesi/FittingNumberDensity/SavedNumDensSpline/Au{self.halo_n}_angle_{self.angle}.npz"
        bhb_SplFit = SplineNumberDensity.load(bhb_spl_fname)
        self.bhb_log_dens_func = bhb_SplFit.log_rho_from_logr
        self.bhb_d_log_dens_func = bhb_SplFit.dlnrho_from_logr

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


class SplineNumberDensity:
    """Cubic-spline log number density as a function of log r."""

    def __init__(self, spl, lnr_min, lnr_max, lnr_knots, log_offset=0.0):
        self._spl = spl
        self.lnr_min = lnr_min
        self.lnr_max = lnr_max
        self.lnr_knots = np.asarray(lnr_knots, float)
        self.log_offset = float(log_offset)  # log N - log(4π)

    # --- evaluation --------------------------------------------------------
    def _clip(self, r):
        return np.clip(np.log(np.asarray(r, float)), self.lnr_min, self.lnr_max)

    def log_rho(self, r):
        lnr = self._clip(r)
        return self._spl(lnr) - 3.0 * lnr + self.log_offset

    def dlnrho(self, r):
        return self._spl(self._clip(r), 1) - 3.0

    def log_rho_from_logr(self, logr):
        return self.log_rho(np.exp(logr))

    def dlnrho_from_logr(self, logr):
        return self.dlnrho(np.exp(logr))

    # --- construction from data -------------------------------------------
    @classmethod
    def from_radii(cls, r, fit_range=None, knots=None, n_knots=20, smooth=0.4):
        ln_r = np.log(np.asarray(r, float))
        ln_r = ln_r[np.isfinite(ln_r)]
        if fit_range is not None:
            ln_r = ln_r[(ln_r >= np.log(fit_range[0])) & (ln_r <= np.log(fit_range[1]))]

        knots = (
            np.linspace(ln_r.min(), ln_r.max(), n_knots)
            if knots is None
            else np.asarray(knots, float)
        )

        spl = agama.splineLogDensity(knots, ln_r, smooth=smooth)
        log_offset = np.log(len(ln_r)) - np.log(4.0 * np.pi)
        return cls(spl, float(ln_r.min()), float(ln_r.max()), knots, log_offset)

    # --- persistence -------------------------------------------------------
    def save(self, path):
        knots = self.lnr_knots
        np.savez(
            path,
            lnr_min=self.lnr_min,
            lnr_max=self.lnr_max,
            knots=knots,
            values=self._spl(knots),
            der=self._spl(knots, 1),
            log_offset=self.log_offset,
        )

    @classmethod
    def load(cls, path):
        d = np.load(path)
        spl = agama.Spline(d["knots"], d["values"], der=d["der"], quintic=False)
        return cls(
            spl,
            float(d["lnr_min"]),
            float(d["lnr_max"]),
            d["knots"],
            float(d["log_offset"]),
        )
