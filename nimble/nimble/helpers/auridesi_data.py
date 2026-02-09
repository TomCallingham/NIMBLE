import numpy as np
import agama
import astropy
import astropy.units as u
from dataclasses import dataclass
from ..config import Config
from .utils import h5py_load, robust_table_to_numpy
from astropy.table import Table
from functools import cached_property

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

    def load_true_dens_fits(self):
        load_fname = f"{extra_y3kp_folder}/auridesirrl_true_profiles/auridesirrl_true_profiles_angle{self.angle}_{self.halo_n}_splinefit.hdf5"
        all_fits = h5py_load(load_fname)
        return all_fits

    def load_true_Menc(self):
        enc_fname = f"{y3kp_folder}/true-profiles/auriga/enclosed-mass-profiles/halo{self.halo_n}-level3-enclosed-mass-profile.csv"
        arr = np.loadtxt(enc_fname, delimiter=",")
        r_true = arr[:, 0]
        Menc_true = arr[:, 1]
        return r_true, Menc_true

    def load_rrl(self, filt_in_loa=True, filt_sub=True):
        obs_tab, true_tab = load_auridesi_cat(
            self.halo_n,
            view=self.angle,
            tracer="rrl",
            filt_in_loa=filt_in_loa,
            filt_sub=filt_sub,
        )
        obs_data = robust_table_to_numpy(obs_tab)
        true_data = robust_table_to_numpy(true_tab)
        return obs_data, true_data

    def load_bhb(self, filt_in_loa=True, filt_sub=True):
        obs_tab, true_tab = load_auridesi_cat(
            self.halo_n,
            view=self.angle,
            tracer="bhb",
            filt_in_loa=filt_in_loa,
            filt_sub=filt_sub,
        )
        obs_data = robust_table_to_numpy(obs_tab)
        true_data = robust_table_to_numpy(true_tab)
        return obs_data, true_data


# basepath = '/global/cfs/cdirs/desi/science/mws/y3kp/'
# ALEX Load
def load_auridesi_cat(halo, view, tracer="bhb", filt_in_loa=True, filt_sub=True):
    """
    Load AuriDESI mock catalog data for a specified halo, view, and tracer.

    Parameters
    ----------
    halo : int
        The halo identifier number (6, 16, 21, 23, 24, 27).
    view : int
        The viewing angle/orientation number (30, 120, 210, 300).
    tracer : str
        The tracer type to load, 'bhb' or 'rrl'.
    apply_cuts : bool, optional
        If True, apply standard selection cuts. Default is True.
        These cuts remove:
        - Stars that are not observed in Loa (sampling p_obs)
        - Stars that are bound to satellites (pop_id < 2)
        - Stars that are in massive streams (stream_flag_1e8 == False)

    Returns
    -------
    obs : astropy.table.Table
        Table containing error-convolved properties of tracers.
    true : astropy.table.Table
        Table containing true properties of tracers.
    """
    cat_path = y3kp_folder + f"field-stars/mocks/auridesi/latest/halo{halo}/"
    cat_file = cat_path + f"halo{halo}-view{view}-{tracer}.fits"

    obs = Table.read(cat_file, hdu="obs")
    true = Table.read(cat_file, hdu="true")

    sel = np.ones(len(obs["ra"]), dtype=bool)
    if filt_in_loa:
        sel &= obs["in_loa"]  # observed in loa (p_obs sampled)
    if filt_sub:
        sel &= true["pop_id"] < 2  # not bound to satellite
        sel &= ~true["stream_flag_1e8"]  # not in massive stream
    obs = obs[sel]
    true = true[sel]

    return obs, true
