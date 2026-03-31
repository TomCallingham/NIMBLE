import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import agama
import jax.numpy as jnp
import jax.scipy.special as jsp_special

from ..models import DispersionMeanMultiModel3D
from .fitting_numpyro import (
    build_agama_spline_basis_1d,
    chol3x3_from_sym,
    forward_solve_lower3,
)
from .fitting_mean_numpyro import fit_numpyro_nuts_mean


# ---------------------------------------------------------------------------
# Derivative basis (AGAMA-consistent, spline coordinate = log r)
# ---------------------------------------------------------------------------


def build_agama_spline_basis_deriv_1d(
    x: np.ndarray,
    knots: np.ndarray,
    der: int = 1,
) -> np.ndarray:
    """
    Build first-derivative basis B_der with shape (Np, nk) such that

        agama.Spline(knots, p)(x, der=der)  ==  B_der @ p

    Because the spline coordinate is log r and the spline values are
    log sigma, der=1 gives  d(log sigma) / d(log r)  directly — which
    is exactly the quantity we want to constrain to be <= 0.
    """
    x = np.asarray(x, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)
    nk = knots.size
    B_der = np.empty((x.size, nk), dtype=np.float64)
    for k in range(nk):
        ek = np.zeros(nk, dtype=np.float64)
        ek[k] = 1.0
        sp = agama.Spline(knots, ek)
        B_der[:, k] = sp(x, der=der)
    return B_der


# ---------------------------------------------------------------------------
# Log-likelihood with one-sided monotonicity penalty
# ---------------------------------------------------------------------------


def make_loglike_3d_obs_jax_mean_declining(
    B_jax: jnp.ndarray,  # (NM, nk)  value basis
    B_der_jax: jnp.ndarray,  # (Np, nk)  first-derivative basis at penalty points
    nk: int,
    lam_mono: float = 1e2,
):
    """
    Identical to make_loglike_3d_obs_jax_mean, plus a one-sided penalty that
    discourages positive slopes in the three log-sigma splines:

        penalty = lam_mono * sum_{components} sum_{points} max(0, d ln sigma / d ln r)^2

    The penalty fires only when d ln sigma / d ln r > 0 (rising dispersion).
    Negative (declining) slopes are completely unconstrained.

    Mean splines (rows 3-5 of P) are not penalised.

    Parameters
    ----------
    B_jax      : (NM, nk)  AGAMA value basis on flattened (N*M) log-r samples
    B_der_jax  : (Np, nk)  AGAMA first-derivative basis at Np penalty points
    nk         : number of spline knots
    lam_mono   : penalty strength; should be large relative to N (try 1e3–1e5)
    """

    def loglike(params: jnp.ndarray, data: dict) -> jnp.ndarray:
        x_f = data["x_f"]  # (NM, 3)
        Cx_f = data["Cx_f"]  # (NM, 3, 3)
        logw = data["logw"]  # (N, M)

        N, M = logw.shape  # static at trace time

        # 6 spline coefficient blocks, each length nk
        # layout: [lsig_r | lsig_t | lsig_p | mu_r | mu_t | mu_p]
        P = params.reshape(6, nk)

        vals = B_jax @ P.T  # (NM, 6)

        lsr = vals[:, 0]
        lst = vals[:, 1]
        lsp = vals[:, 2]
        mu_r = vals[:, 3]
        mu_t = vals[:, 4]
        mu_p = vals[:, 5]

        sig_r2 = jnp.exp(2.0 * lsr)
        sig_t2 = jnp.exp(2.0 * lst)
        sig_p2 = jnp.exp(2.0 * lsp)

        # Lambda = Cx + diag(sig^2), symmetric elements
        a = Cx_f[:, 0, 0] + sig_r2
        d = Cx_f[:, 1, 1] + sig_t2
        f = Cx_f[:, 2, 2] + sig_p2
        b = Cx_f[:, 1, 0]
        c = Cx_f[:, 2, 0]
        e = Cx_f[:, 2, 1]

        x1 = x_f[:, 0] - mu_r
        x2 = x_f[:, 1] - mu_t
        x3 = x_f[:, 2] - mu_p

        l11, l21, l31, l22, l32, l33, ok = chol3x3_from_sym(a, b, c, d, e, f)
        y1, y2, y3 = forward_solve_lower3(l11, l21, l31, l22, l32, l33, x1, x2, x3)

        quad = y1 * y1 + y2 * y2 + y3 * y3
        logdet = 2.0 * (jnp.log(l11) + jnp.log(l22) + jnp.log(l33))

        ll_f = -0.5 * (quad + logdet)  # (NM,)
        ll = ll_f.reshape(N, M)  # (N, M)

        logp_star = jsp_special.logsumexp(logw + ll, axis=1)  # (N,)
        total = jnp.sum(logp_star)

        # --- one-sided monotonicity penalty ---
        # B_der_jax @ P[:3].T  has shape (Np, 3)
        # Each column is d(log sigma_i) / d(log r) at the Np penalty points.
        # We only penalise the positive part (rising dispersion).
        derivs = B_der_jax @ P[:3].T  # (Np, 3)
        violations = jnp.maximum(0.0, derivs)  # zero where slope < 0
        mono_pen = lam_mono * jnp.sum(violations**2)

        total = total - mono_pen

        ok_all = jnp.all(ok) & jnp.isfinite(total)
        return jnp.where(ok_all, total, -jnp.inf)

    return loglike


# ---------------------------------------------------------------------------
# Setup + fit  (same signature / return values as setup_and_fit_obs_mean)
# ---------------------------------------------------------------------------


def setup_and_fit_obs_mean_declining(
    sample_data,
    scfg,
    vdisp_min: float = 20.0,
    vdisp_max: float = 400.0,
    lam_mono: float = 1e2,
    r_mono_min: float = 30.0,
    **mcmc_kwargs,
):
    """
    Drop-in replacement for setup_and_fit_obs_mean that additionally enforces
    d ln sigma / d ln r <= 0 for all three velocity dispersion components via
    a one-sided quadratic penalty with strength lam_mono.

    Parameters
    ----------
    sample_data : dict with keys
        logr         (N, M)
        vel_gal      (N, M, 3)
        err_mat_gal  (N, M, 3, 3)
        log_dist_w   (N, M)
    scfg        : spline config with attributes num_knots, knots_logr
    vdisp_min   : lower bound on sigma (km/s)
    vdisp_max   : upper bound on sigma (km/s)
    lam_mono    : monotonicity penalty strength (default 1e4; tune if needed)
    **mcmc_kwargs : forwarded to fit_numpyro_nuts_mean

    Returns
    -------
    mcmc, sigma_samples, mean_samples, multi_disp
        Identical types/shapes to setup_and_fit_obs_mean.
    """
    logr = np.asarray(sample_data["logr"], dtype=np.float64)  # (N, M)
    x = np.asarray(sample_data["vel_gal"], dtype=np.float64)  # (N, M, 3)
    Cx = np.asarray(sample_data["err_mat_gal"], dtype=np.float64)  # (N, M, 3, 3)
    logw = np.asarray(sample_data["log_dist_w"], dtype=np.float64)  # (N, M)

    N, M = logr.shape
    NM = N * M

    logr_f = logr.reshape(NM)
    x_f = x.reshape(NM, 3)
    Cx_f = Cx.reshape(NM, 3, 3)

    nk = int(scfg.num_knots)
    knots = np.asarray(scfg.knots_logr, dtype=np.float64)
    assert knots.shape[0] == nk, f"Expected {nk} knots, got {knots.shape[0]}"

    # --- value basis (same as baseline) ---
    B = build_agama_spline_basis_1d(logr_f, knots)  # (NM, nk)
    B_j = jnp.asarray(B)

    # --- derivative basis at knots + interval midpoints ---
    # Knots + midpoints give 2*nk - 1 penalty points, which is enough to
    # catch within-interval violations that knot-only checks would miss.
    knot_mids = 0.5 * (knots[:-1] + knots[1:])  # nk-1 midpoints in log r
    x_pen_all = np.sort(np.concatenate([knots, knot_mids]))  # (2*nk-1,) in log r
    x_pen = x_pen_all[x_pen_all > np.log(r_mono_min)]

    print("NEW rmono min!")

    B_der = build_agama_spline_basis_deriv_1d(x_pen, knots, der=1)  # (2nk-1, nk)
    B_der_j = jnp.asarray(B_der)

    print(
        f"Monotonicity penalty: {2 * nk - 1} evaluation points, lam_mono={lam_mono:.3g}"
    )

    # --- data dict ---
    data_j = {
        "x_f": jnp.asarray(x_f),  # (NM, 3)
        "Cx_f": jnp.asarray(Cx_f),  # (NM, 3, 3)
        "logw": jnp.asarray(logw),  # (N, M)
    }

    # --- build penalised log-likelihood ---
    loglike_jax = make_loglike_3d_obs_jax_mean_declining(
        B_j, B_der_j, nk, lam_mono=float(lam_mono)
    )

    # --- MCMC ---
    mcmc, raw_samples = fit_numpyro_nuts_mean(
        loglike_jax,
        nk=nk,
        data=data_j,
        vdisp_min=vdisp_min,
        vdisp_max=vdisp_max,
        **mcmc_kwargs,
    )

    sigma_samples = raw_samples["params_sigma"]
    mean_samples = raw_samples["params_mean"]

    multi_disp = DispersionMeanMultiModel3D.from_sampler_3d(
        sigma_samples, mean_samples, scfg.knots_logr
    )
    return mcmc, sigma_samples, mean_samples, multi_disp
