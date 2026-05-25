import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from ..helpers.coords import add_r_vel_obs_data
from scipy.stats import truncnorm


def prepare_obs_data(obs_data, solar_params, func_log_rho_from_logr, n_dist=10):
    obs_sample_data = make_distance_samples(obs_data, n_dist=n_dist)
    obs_sample_data = add_r_vel_obs_data(obs_sample_data, solar_params)
    obs_sample_data = add_distance_weights(obs_sample_data, func_log_rho_from_logr)
    obs_sample_data = calc_err_mat_gal(obs_sample_data, solar_params)
    return obs_sample_data


def calc_err_mat_gal(fit_data: dict, solar_params):
    """
    Precompute:
      y_obs = [vrad, pmra, pmdec]  (N,3)
      x_obs = R y_obs + K         (N,3)
      err_mat_gal   = R err_mat_obs R^T           (N,3,3)

    Notes:
    - This uses y = (vrad [km/s], pmra [mas/yr], pmdec [mas/yr]).
      Therefore err_mat_obs is in these mixed units, and R must be consistent (it will be
      if you built it by differencing in vrad (km/s) and pm (mas/yr)).
    """
    data_shape = fit_data["ra"].shape
    if len(data_shape) > 1:
        for key in [
            "ra",
            "dec",
            "pmdec",
            "pmra",
            "vrad",
            "distance",
            "pmra_err",
            "pmdec_err",
            "vrad_err",
        ]:
            fit_data[key] = fit_data[key].ravel()

    R, K = calc_transformation_constants(fit_data, solar_params)

    N = K.shape[0]

    ev = np.asarray(fit_data["vrad_err"], dtype=float)
    e1 = np.asarray(fit_data["pmra_err"], dtype=float)
    e2 = np.asarray(fit_data["pmdec_err"], dtype=float)

    err_mat_obs = np.zeros((N, 3, 3), dtype=float)
    err_mat_obs[:, 0, 0] = ev * ev
    err_mat_obs[:, 1, 1] = e1 * e1
    err_mat_obs[:, 2, 2] = e2 * e2

    pm_corr_key = "pm_radec_corr"  # optional, correlation coefficient
    if pm_corr_key in fit_data:
        print("Using correlation!")
        rho = np.asarray(fit_data[pm_corr_key], dtype=float)
        if len(data_shape) > 1:
            rho = rho.ravel()

        cov12 = rho * e1 * e2
        err_mat_obs[:, 1, 2] = cov12
        err_mat_obs[:, 2, 1] = cov12
    else:
        print("Not using correlation!")

    err_mat_gal = R @ err_mat_obs @ np.swapaxes(R, -1, -2)  # (N,3,3)

    err_mat_gal = 0.5 * (err_mat_gal + np.swapaxes(err_mat_gal, -1, -2))

    fit_data["err_mat_gal"] = err_mat_gal

    if len(data_shape) > 1:
        for key in [
            "ra",
            "dec",
            "pmdec",
            "pmra",
            "vrad",
            "distance",
            "pmra_err",
            "pmdec_err",
            "vrad_err",
        ]:
            fit_data[key] = fit_data[key].reshape(data_shape)
        fit_data["err_mat_gal"] = fit_data["err_mat_gal"].reshape(*data_shape, 3, 3)

    return fit_data


def calc_transformation_constants(fit_data, solar_params, dvlos_kms=1.0, dpm_masyr=1.0):
    ra = np.asarray(fit_data["ra"], dtype=float)
    dec = np.asarray(fit_data["dec"], dtype=float)
    dist = np.asarray(fit_data["distance"], dtype=float)  # kpc
    n = dist.size

    ra4, dec4, dist4 = (
        np.tile(ra, 4) * u.deg,
        np.tile(dec, 4) * u.deg,
        np.tile(dist, 4) * u.kpc,
    )

    vlos4 = np.zeros(4 * n) * (u.km / u.s)
    pmra4 = np.zeros(4 * n) * (u.mas / u.yr)
    pmde4 = np.zeros(4 * n) * (u.mas / u.yr)

    vlos4[n : 2 * n] = dvlos_kms * (u.km / u.s)
    pmra4[2 * n : 3 * n] = dpm_masyr * (u.mas / u.yr)
    pmde4[3 * n : 4 * n] = dpm_masyr * (u.mas / u.yr)

    c_icrs = coord.SkyCoord(
        ra=ra4,
        dec=dec4,
        distance=dist4,
        pm_ra_cosdec=pmra4,
        pm_dec=pmde4,
        radial_velocity=vlos4,
        frame="icrs",
    )
    galcen_frame = coord.Galactocentric(
        galcen_distance=np.abs(solar_params["galcen_pos"][0]),
        z_sun=solar_params["galcen_pos"][-1],
        galcen_v_sun=solar_params["galcen_vsun"],
    )

    c_galcen = c_icrs.transform_to(galcen_frame)

    # Cartesian pos/vel
    X, Y, Z = (
        c_galcen.cartesian.x.to_value(u.kpc),
        c_galcen.cartesian.y.to_value(u.kpc),
        c_galcen.cartesian.z.to_value(u.kpc),
    )
    VX, VY, VZ = (
        c_galcen.velocity.d_x.to_value(u.km / u.s),
        c_galcen.velocity.d_y.to_value(u.km / u.s),
        c_galcen.velocity.d_z.to_value(u.km / u.s),
    )

    r = np.sqrt(X * X + Y * Y + Z * Z)
    phi = np.arctan2(Y, X)
    ct = np.clip(Z / r, -1.0, 1.0)
    st = np.sqrt(1.0 - ct * ct)  # sin(theta)
    sp, cp = np.sin(phi), np.cos(phi)

    # unit vectors
    erx, ery, erz = st * cp, st * sp, ct
    etx, ety, etz = ct * cp, ct * sp, -st
    epx, epy = -sp, cp

    vr = VX * erx + VY * ery + VZ * erz
    vtheta = VX * etx + VY * ety + VZ * etz
    vphi = VX * epx + VY * epy  # + VZ*0

    x_all = np.stack([vr, vtheta, vphi], axis=-1).reshape(4, n, 3)
    x0 = x_all[0]  # y=(0,0,0)
    K = x0
    R = np.stack(
        [
            (x_all[1] - x0) / dvlos_kms,
            (x_all[2] - x0) / dpm_masyr,
            (x_all[3] - x0) / dpm_masyr,
        ],
        axis=-1,
    )  # (n,3,3)

    return R, K


def make_distance_samples(
    fit_data: dict,
    n_dist: int = 100,
    seed: int | None = 0,
    min_dist: float = 1e-6,
) -> dict:
    """
    Create sample_data with shape (N, n_dist) arrays:
      distance_samp: sampled distances
      ra/dec/vrad/pmra/pmdec: repeated along sample axis
    """
    rng = np.random.default_rng(seed)

    d0 = np.asarray(fit_data["distance"], dtype=float)  # (N,)
    de = np.asarray(fit_data["distance_err"], dtype=float)  # (N,)
    # if "distance_err" in fit_data:
    #     de = np.asarray(fit_data["distance_err"], dtype=float)  # (N,)
    #     lin_errors = True
    # else:
    #     de = np.asarray(fit_data["frac_distance_err"], dtype=float)  # (N,)
    #     lin_errors = False
    N = d0.size

    # Gaussian draws
    d = np.zeros((N, n_dist))

    bad = d <= min_dist
    for _ in range(5):
        if not np.any(bad):
            break
        d[bad] = rng.normal(loc=d0[:, None], scale=de[:, None], size=d.shape)[bad]
        # if lin_errors:
        #     d[bad] = rng.normal(loc=d0[:, None], scale=de[:, None], size=d.shape)[bad]
        # else:
        #     d[bad] = (
        #         d0[:, None] * (1.0 + de[:, None] * rng.standard_normal(size=d.shape))
        #     )[bad]
        bad = d <= min_dist
    if np.any(bad):
        d[bad] = min_dist

    # Repeat other observables along sample axis
    def rep(key):
        return np.asarray(fit_data[key], dtype=float)[:, None].repeat(n_dist, axis=1)

    sample_data = {
        "distance": d,  # (N, n_dist)   <-- overwrite distance with samples
        "ra": rep("ra"),
        "dec": rep("dec"),
        "vrad": rep("vrad"),
        "pmra": rep("pmra"),
        "pmdec": rep("pmdec"),
    }
    if "pmra_err" in fit_data:
        for key in ["pmra_err", "pmdec_err", "vrad_err"]:
            sample_data[key] = rep(key)
    if "pm_radec_corr" in fit_data:
        sample_data["pm_radec_corr"] = rep("pm_radec_corr")

    return sample_data


def add_distance_weights(sample_data: dict, log_rho_calc, lin_dist=True) -> dict:
    """
    Adds sample_data["dist_w"] shape (N, n_dist), normalized per star.

    weights ∝ rho(r) [* d^2 if use_d2]
    """
    r = np.asarray(sample_data["r"], dtype=float)  # (N, n_dist)
    print(r.shape)
    dens = np.exp(log_rho_calc(np.log(r)))
    if lin_dist:
        print("Linear Dist Draw")
        w = dens * (sample_data["distance"] ** 2)
    else:
        print("Log Dist Draw")
        w = dens * (sample_data["distance"] ** 3)

    print("NEW DISTANCE FACTOR!!!")
    w *= sample_data["distance"] ** 2

    esp = 1e-300
    w[w < esp] = esp
    w[np.isnan(w)] = esp
    wsum = w.sum(axis=1, keepdims=True)
    w /= wsum

    sample_data["dist_w"] = w

    sample_data["log_dist_w"] = np.log(w)
    return sample_data
