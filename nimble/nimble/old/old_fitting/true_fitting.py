import numpy as np
import scipy.optimize
import agama
from ..models import FitDispersion3D

def _loglikelihood_1d(params_sigma, v, logr, knots_logr):
    l_sigma = agama.Spline(knots_logr, params_sigma)(logr)
    sigma2 = np.exp(2.0 * l_sigma)
    ll = -l_sigma - 0.5 * (v * v) / sigma2
    out = np.sum(ll)
    return out if np.isfinite(out) else -np.inf

def _neg_loglikelihood_1d(params_sigma, v, logr, knots_logr):
    return -_loglikelihood_1d(params_sigma, v, logr, knots_logr)

def fit_scipy_3d_dispersion(true_data, scfg, vdisp_min=20, vdisp_max=400):
    K = scfg.num_knots
    bounds = [(np.log(vdisp_min), np.log(vdisp_max))] * K
    x0 = np.full(K, 5.0)

    logr = true_data["logr"]

    res_r = scipy.optimize.minimize(
        _neg_loglikelihood_1d,
        x0,
        args=(true_data["vr"], logr, scfg.knots_logr),
        bounds=bounds,
        method="L-BFGS-B",
    )
    res_t = scipy.optimize.minimize(
        _neg_loglikelihood_1d,
        x0,
        args=(true_data["vtheta"], logr, scfg.knots_logr),
        bounds=bounds,
        method="L-BFGS-B",
    )
    res_p = scipy.optimize.minimize(
        _neg_loglikelihood_1d,
        x0,
        args=(true_data["vphi"], logr, scfg.knots_logr),
        bounds=bounds,
        method="L-BFGS-B",
    )

    x_all = np.concatenate([res_r.x, res_t.x, res_p.x])
    disp3d = FitDispersion3D.from_tuple_3d(x_all, scfg.knots_logr)

    # return a small structured result, but keep the same API shape
    res = dict(r=res_r, theta=res_t, phi=res_p)
    return disp3d, res

def fit_scipy_2d_dispersion(true_data, scfg, vdisp_min=20, vdisp_max=400):
    K = scfg.num_knots
    bounds = [(np.log(vdisp_min), np.log(vdisp_max))] * K
    x0 = np.full(K, 5.0)

    logr = true_data["logr"]

    res_r = scipy.optimize.minimize(
        _neg_loglikelihood_1d,
        x0,
        args=(true_data["vr"], logr, scfg.knots_logr),
        bounds=bounds,
        method="L-BFGS-B",
    )

    # fit shared tangential dispersion using BOTH vtheta and vphi contributions
    def nll_tan(params_sigma):
        l_sigma = agama.Spline(scfg.knots_logr, params_sigma)(logr)
        sigma2 = np.exp(2.0 * l_sigma)
        ll = (-l_sigma - 0.5 * (true_data["vtheta"] ** 2) / sigma2) + \
             (-l_sigma - 0.5 * (true_data["vphi"]   ** 2) / sigma2)
        out = np.sum(ll)
        return -(out if np.isfinite(out) else -np.inf)

    res_tp = scipy.optimize.minimize(
        nll_tan,
        x0,
        bounds=bounds,
        method="L-BFGS-B",
    )

    x_all = np.concatenate([res_r.x, res_tp.x])
    disp2d = FitDispersion3D.from_tuple_2d(x_all, scfg.knots_logr)
    res = dict(r=res_r, tangential=res_tp)
    return disp2d, res

