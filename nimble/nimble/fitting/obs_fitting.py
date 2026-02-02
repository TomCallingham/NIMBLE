import numpy as np
from ..config import SmallConfig
from ..models import FitDispersion3D
import agama

import scipy


def loglikelihood_3d(params_sigma, fit_data, scfg: SmallConfig):
    params_sigmar = params_sigma[: scfg.num_knots]
    l_sigma_r = agama.Spline(scfg.knots_logr, params_sigmar)(fit_data["logr"])
    sigma_r2 = np.exp(2 * l_sigma_r)
    ll_r = -l_sigma_r - 0.5 * ((fit_data["vr"] ** 2) / sigma_r2)

    params_sigma_theta = params_sigma[scfg.num_knots : 2 * scfg.num_knots]
    l_sigma_theta = agama.Spline(scfg.knots_logr, params_sigma_theta)(fit_data["logr"])
    sigma_theta2 = np.exp(2 * l_sigma_theta)
    ll_theta = -l_sigma_theta - 0.5 * ((fit_data["vtheta"] ** 2) / sigma_theta2)

    params_sigma_phi = params_sigma[2 * scfg.num_knots :]
    l_sigma_phi = agama.Spline(scfg.knots_logr, params_sigma_phi)(fit_data["logr"])
    sigma_phi2 = np.exp(2 * l_sigma_phi)
    ll_phi = -l_sigma_phi - 0.5 * ((fit_data["vphi"] ** 2) / sigma_phi2)

    loglikelihood = np.sum(ll_r + ll_theta + ll_phi)
    if not np.isfinite(loglikelihood):
        loglikelihood = -np.inf

    return loglikelihood


def neg_loglikelihood_3d(params_sigma, fit_data, scfg: SmallConfig):
    return -loglikelihood_3d(params_sigma, fit_data, scfg)


def loglikelihood_2d(params_sigma, fit_data, scfg: SmallConfig):
    params_sigmar = params_sigma[: scfg.num_knots]
    l_sigma_r = agama.Spline(scfg.knots_logr, params_sigmar)(fit_data["logr"])
    sigma_r2 = np.exp(2 * l_sigma_r)
    ll_r = -l_sigma_r - 0.5 * ((fit_data["vr"] ** 2) / sigma_r2)

    params_sigma_theta_phi = params_sigma[scfg.num_knots :]
    l_sigma_phitheta = agama.Spline(scfg.knots_logr, params_sigma_theta_phi)(
        fit_data["logr"]
    )
    sigma_phitheta2 = np.exp(2 * l_sigma_phitheta)
    ll_theta = -l_sigma_phitheta - 0.5 * ((fit_data["vtheta"] ** 2) / sigma_phitheta2)
    ll_phi = -l_sigma_phitheta - 0.5 * ((fit_data["vphi"] ** 2) / sigma_phitheta2)

    loglikelihood = np.sum(ll_r + ll_theta + ll_phi)
    if not np.isfinite(loglikelihood):
        loglikelihood = -np.inf
    return loglikelihood


def neg_loglikelihood_2d(params_sigma, fit_data, scfg: SmallConfig):
    return -loglikelihood_2d(params_sigma, fit_data, scfg)


### Scipy Fitting


def fit_scipy_3d_dispersion(true_data, scfg, vdisp_min=20, vdisp_max=400):
    x0 = np.full(3 * scfg.num_knots, 5)
    bounds = [(np.log(vdisp_min), np.log(vdisp_max))] * scfg.num_knots * 3
    res_3d = scipy.optimize.minimize(
        neg_loglikelihood_3d,
        x0,
        args=(true_data, scfg),
        bounds=bounds,
        method="L-BFGS-B",
    )
    disp3d = FitDispersion3D.from_tuple_3d(res_3d.x, scfg.knots_logr)
    return disp3d, res_3d


def fit_scipy_2d_dispersion(true_data, scfg, vdisp_min=20, vdisp_max=400):
    x0 = np.full(2 * scfg.num_knots, 5)

    bounds = [(np.log(vdisp_min), np.log(vdisp_max))] * scfg.num_knots * 2
    res_2d = scipy.optimize.minimize(
        neg_loglikelihood_2d,
        x0,
        args=(true_data, scfg),
        bounds=bounds,
        method="L-BFGS-B",
    )

    disp2d = FitDispersion3D.from_tuple_2d(res_2d.x, scfg.knots_logr)
    return disp2d, res_2d
