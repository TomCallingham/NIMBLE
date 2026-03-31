import astropy.units as u
from astropy.coordinates import ICRS, Galactic
import numpy as np


def add_r_vel_obs_data(obs_data, solar_params):
    data_shape = obs_data["ra"].shape
    if len(data_shape) > 1:
        for key in ["ra", "dec", "pmdec", "pmra", "vrad", "distance"]:
            obs_data[key] = obs_data[key].ravel()

    u_kms = u.km / u.second
    ra = u.Quantity(obs_data["ra"], unit=u.degree, copy=False)
    dec = u.Quantity(obs_data["dec"], unit=u.degree, copy=False)
    pmra_cosdec = u.Quantity(
        obs_data["pmra"], unit=u.milliarcsecond / u.year, copy=False
    )
    pmdec = u.Quantity(obs_data["pmdec"], unit=u.milliarcsecond / u.year, copy=False)
    vrad = u.Quantity(obs_data["vrad"], unit=u_kms, copy=False)
    dist = u.Quantity(obs_data["distance"], unit=u.kpc, copy=False)

    coords = ICRS(
        ra=ra,
        dec=dec,
        distance=dist,
        pm_ra_cosdec=pmra_cosdec,
        pm_dec=pmdec,
        radial_velocity=vrad,
    ).transform_to(Galactic())

    helio_vel = (
        coords.cartesian.differentials["s"].get_d_xyz().transpose() / u_kms
    ).value
    helio_pos = (coords.cartesian.get_xyz().transpose() / u.kpc).value

    pos = helio_pos + solar_params["galcen_pos"].value
    vel = helio_vel + solar_params["galcen_vsun"].value

    obs_data["r"], obs_data["vr"], obs_data["vphi"], obs_data["vtheta"] = (
        cartesian_to_spherical_vel(pos, vel)
    )

    obs_data["logr"] = np.log(obs_data["r"])

    obs_data["vel_gal"] = np.stack(
        [obs_data["vr"], obs_data["vtheta"], obs_data["vphi"]], axis=-1
    )  # (N, 3)

    obs_data["v"] = np.linalg.norm(obs_data["vel_gal"], axis=-1)

    obs_data["pos"] = pos
    obs_data["vel"] = vel
    obs_data["R"] = np.linalg.norm(obs_data["pos"][..., :2], axis=-1)
    obs_data["Z"] = obs_data["pos"][..., 2]

    if len(data_shape) > 1:
        for key in [
            "ra",
            "dec",
            "pmdec",
            "pmra",
            "vrad",
            "distance",
            "r",
            "vr",
            "vphi",
            "vtheta",
            "logr",
        ]:
            obs_data[key] = obs_data[key].reshape(data_shape)
        obs_data["vel_gal"] = obs_data["vel_gal"].reshape(*data_shape, 3)
        obs_data["pos"] = obs_data["pos"].reshape(*data_shape, 3)
        obs_data["vel"] = obs_data["vel"].reshape(*data_shape, 3)

    return obs_data


def cartesian_to_spherical_vel(pos: np.ndarray, vel: np.ndarray):
    """
    pos: (n, 3) array of [x, y, z]
    vel: (n, 3) array of [vx, vy, vz]

    Returns:
        r      : (n,) radial distance
        vr     : (n,) radial velocity (along e_r)
        vphi   : (n,) azimuthal velocity (along e_phi)
        vtheta : (n,) polar velocity (along e_theta)
    """
    x, y, z = pos.T
    vx, vy, vz = vel.T

    # Radius
    r = np.linalg.norm(pos, axis=1)
    vr = ((x * vx) + (y * vy) + (z * vz)) / r

    # Angles
    phi = np.arctan2(y, x)  # [-pi, pi]
    # Guard against roundoff pushing z/r slightly outside [-1, 1]
    cos_theta = np.clip(z / np.where(r == 0, 1.0, r), -1.0, 1.0)
    theta = np.arccos(cos_theta)

    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # e_theta
    e_th_x = cos_theta * cos_phi
    e_th_y = cos_theta * sin_phi
    e_th_z = -np.sin(theta)

    # e_phi
    e_ph_x = -sin_phi
    e_ph_y = cos_phi

    vtheta = vx * e_th_x + vy * e_th_y + vz * e_th_z
    vphi = vx * e_ph_x + vy * e_ph_y

    return r, vr, vphi, vtheta
