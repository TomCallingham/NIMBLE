import numpy as np
import agama
import astropy
import astropy.units as u
from dataclasses import dataclass
from ..config import Config
from .utils import load_cols_from_fits
from functools import cached_property

y3kp_folder = "~/data/observations/local_desi/y3kp"

au_halos = [6, 16, 21, 23, 24, 27]
au_angles = [30, 120, 210, 300]


@dataclass(init=False)
class AuriDesiParams:
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
        # print(f"\033[1;33m**** RUNNING AURIDESI {halo_n} {angle} ****\033[0m")

        self.figs_root = f"results/auridesi/deconv_{halo_n}_{angle}/"
        self.true_path = f"data/auriga/H{halo_n}/Au{halo_n}_true.csv"

    def create_config(self):
        cfg = Config()
        # print("Why is Gmax needed?")
        # cfg.Gmax = 19.0
        # cfg.Gmin = 16.0
        # cfg.bmin = 30.0
        cfg.min_r = 1.0
        cfg.max_r = 100.0
        cfg.num_knots = 5
        cfg.figs_root = self.figs_root
        cfg.true_path = self.true_path
        cfg.lsr_info = self.solar_params["lsr_info"]
        return cfg

    @cached_property
    def solar_params(self):
        return self.load_solar_params()

    def load_solar_params(self) -> dict:
        halo_n, angle = self.halo_n, self.angle
        rrl_desi_fname = f"{y3kp_folder}/field-stars/mocks/auridesi/halo{halo_n}/halo{halo_n}-view{angle}-rrl.fits"
        rrl_desi_fname = f"{y3kp_folder}/field-stars/mocks/auridesi/v3.0/halo{halo_n}/halo{halo_n}-view{angle}-rrl.fits"

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

    def load_rrl_true(self, loa_filt=True, all_cols=False):
        halo_n, angle = self.halo_n, self.angle
        rrl_desi_fname = f"{y3kp_folder}/field-stars/mocks/auridesi/halo{halo_n}/halo{halo_n}-view{angle}-rrl.fits"

        min_true_keys = ["ra", "dec", "pmra", "pmdec", "distance", "vrad", "pop_id"]
        cols = None if all_cols else min_true_keys

        true_data = load_cols_from_fits(rrl_desi_fname, hdu="TRUE", cols=cols)
        filt = true_data["pop_id"] != 2
        # TODO:More Filter?
        if loa_filt:
            print("in loa not added, need new data!")
        true_data = {key: val[filt] for key, val in true_data.items()}
        # TODO: Include pm errors properly
        true_data["pm_err"] = np.sqrt(
            0.5 * ((true_data["pmra"] ** 2) + (true_data["pmdec"] ** 2))
        )
        return true_data

    def load_rrl_obs(self, loa_filt=True, all_cols=False):
        halo_n, angle = self.halo_n, self.angle
        rrl_desi_fname = f"{y3kp_folder}/field-stars/mocks/auridesi/halo{halo_n}/halo{halo_n}-view{angle}-rrl.fits"

        min_obs_keys = [
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
            "in_loa",
            "id_loa",
        ]
        cols = None if all_cols else min_obs_keys
        obs_data = load_cols_from_fits(rrl_desi_fname, hdu="OBS", cols=cols)
        print("No pop id filt?")
        # TODO:More Filter?
        if loa_filt:
            print("in loa not added, need new data!")
            filt = obs_data["in_loa"]
            obs_data = {key: val[filt] for key, val in obs_data.items()}
        # TODO: Include pm errors properly
        obs_data["pm_err"] = np.sqrt(
            0.5 * ((obs_data["pmra"] ** 2) + (obs_data["pmdec"] ** 2))
        )
        return obs_data
