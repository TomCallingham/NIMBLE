import numpy as np
import agama
import astropy
import astropy.units as u
from astropy.table import QTable
from astropy.coordinates import SkyCoord

np.random.seed(42)

d2r = np.pi / 180  # conversion from degrees to radians

y3kp_folder = "~/data/observations/local_desi/y3kp"


def load_rrl_auridesi_obs_data(
    halo_n, angle=30, lims=None, obs_not_true=True, loa_filt=True
):
    rrl_desi_fname = f"{y3kp_folder}/field-stars/mocks/auridesi/halo{halo_n}/halo{halo_n}-view{angle}-rrl.fits"
    true_keys = ["gaia_g", "pop_id"]  # ,"in_loa"
    obs_keys = ["pmra_err", "pmdec_err"]
    trans = {"v0": "vlos", "v0_err": "vlos_err", "gaia_g": "Gapp"}
    if obs_not_true:
        obs_keys += ["ra", "dec", "pmra", "pmdec", "distance", "v0", "v0_err"]
        # true_keys+ = []

    else:
        # obs_keys = []
        true_keys += ["ra", "dec", "pmra", "pmdec", "distance", "vrad"]
    true_tab = QTable.read(rrl_desi_fname, hdu="TRUE", format="fits")
    obs_tab = QTable.read(rrl_desi_fname, hdu="OBS", format="fits")
    obs_data = {trans.get(key, key): true_tab[key] for key in true_keys} | {
        trans.get(key, key): obs_tab[key] for key in obs_keys
    }
    del true_tab, obs_tab

    with astropy.io.fits.open(rrl_desi_fname) as AllTab:
        header = AllTab[0].header
        solar_data = {
            "galcen_vsun": np.asarray(
                [header["U_SUN"], (header["VLSR"] + header["V_SUN"]), header["W_SUN"]]
            )
            * u.km
            / u.s,
            "galcen_pos": np.asarray([header["SOLAR_R"], 0.0, header["SOLAR_H"]])
            * u.Mpc.to(u.kpc)
            * u.kpc,
        }
    solar_data["lsr_info"] = (
        -solar_data["galcen_pos"][0].value,
        solar_data["galcen_vsun"].value,
        solar_data["galcen_pos"][2].value,
    )

    filt = obs_data["pop_id"] != 2
    if loa_filt:
        print("in loa not found, need new data!")
    obs_data = {key: val[filt] for key, val in obs_data.items()}

    c_icrs = SkyCoord(
        ra=obs_data["ra"],
        dec=obs_data["dec"],
        pm_ra_cosdec=obs_data["pmra"],  # NOTE: pm_ra_cosdec, not pm_ra
        pm_dec=obs_data["pmdec"],
        frame="icrs",
    )

    c_gal = c_icrs.transform_to("galactic")
    obs_data["l"], obs_data["b"] = c_gal.l.to(u.degree), c_gal.b.to(u.degree)
    obs_data["pml"], obs_data["pmb"] = c_gal.pm_l_cosb, c_gal.pm_b

    # l_rad, b_rad, pml, pmb = agama.transformCelestialCoords(
    #     agama.fromICRStoGalactic, np.deg2rad(obs_data["ra"].value), np.deg2rad(obs_data["dec"].value),
    #                                                                            obs_data["pmra"].value, obs_data["pmdec"].value
    # )
    # # back to degrees
    # l_deg = l_rad / d2r
    # b_deg = b_rad / d2r

    # print(obs_data["l"].value,l_deg)
    # print(obs_data["b"].value,b_deg)
    # print(obs_data["pml"].value,pml)
    # print(obs_data["pmb"].value,pmb)

    obs_data = {key: val.value for key, val in obs_data.items()}

    if lims is not None:
        filt = (
            (np.abs(obs_data["b"]) >= lims["bmin"])
            * (obs_data["dec"] >= lims["decmin"])
            * (obs_data["Gapp"] > lims["Gmin"])
            * (obs_data["Gapp"] < lims["Gmax"])
        )
        obs_data = {key: val[filt] for key, val in obs_data.items()}
        # print(filt.mean())
    obs_data["pm_err"] = np.sqrt(
        0.5 * ((obs_data["pmra"] ** 2) + (obs_data["pmdec"] ** 2))
    )

    return obs_data, solar_data
