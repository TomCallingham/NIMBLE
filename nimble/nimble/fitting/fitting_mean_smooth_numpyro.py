import jax

jax.config.update("jax_enable_x64", True)

print("local_device_count:", jax.local_device_count())
print("devices:", jax.devices())
import numpy as np
import agama
import jax.numpy as jnp
from ..models import DispersionMeanMultiModel3D
import jax.scipy.special as jsp_special
from .fitting_mean_numpyro import fit_numpyro_nuts_mean

from .fitting_numpyro import (
    build_agama_spline_basis_1d,
    chol3x3_from_sym,
    forward_solve_lower3,
)


def trapz_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoidal integration weights for non-uniform x."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be 1D with at least 2 points")
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w


def build_agama_spline_basis_deriv_1d(
    x: np.ndarray, knots: np.ndarray, der: int = 2
) -> np.ndarray:
    """
    Build derivative basis Bder with shape (N, nk) such that
        agama.Spline(knots, p)(x, der=der) == Bder @ p
    (up to roundoff)
    """
    x = np.asarray(x, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)
    nk = knots.size
    Bder = np.empty((x.size, nk), dtype=np.float64)

    for k in range(nk):
        ek = np.zeros(nk, dtype=np.float64)
        ek[k] = 1.0
        sp = agama.Spline(knots, ek)
        # Agama spline derivative wrt the spline coordinate (here: log r)
        Bder[:, k] = sp(x, der=der)

    return Bder


def build_agama_spline_roughness_matrix(
    knots: np.ndarray,
    der: int = 2,
    x_pen: np.ndarray | None = None,
    ngrid: int = 512,
) -> np.ndarray:
    """
    Build roughness matrix R (nk,nk) such that
        p.T @ R @ p  ~  ∫ [f^(der)(x)]^2 dx
    for f(x) = agama.Spline(knots, p), where x is the spline coordinate.

    If your spline coordinate is log r (as here), der=2 penalizes curvature in log r,
    which enforces a smooth first derivative.
    """
    knots = np.asarray(knots, dtype=np.float64)
    if x_pen is None:
        x_pen = np.linspace(knots[0], knots[-1], int(ngrid), dtype=np.float64)
    else:
        x_pen = np.asarray(x_pen, dtype=np.float64)

    Bder = build_agama_spline_basis_deriv_1d(x_pen, knots, der=der)  # (G,nk)
    w = trapz_weights(x_pen)  # (G,)

    # R = ∫ Bder^T Bder dx  ≈  Bder^T diag(w) Bder
    R = (Bder.T * w) @ Bder
    R = 0.5 * (R + R.T)  # numerical symmetrization

    return R


import jax.numpy as jnp
import jax.scipy.special as jsp_special


def make_loglike_3d_obs_jax_mean_smooth(
    B_jax: jnp.ndarray,
    nk: int,
    R_blocks_jax: jnp.ndarray,  # (6,nk,nk)
    lam_smooth_jax: jnp.ndarray,  # (6,)
):
    """
    All errors:
      - distance marginalisation over M samples
      - velocity measurement covariance Cx (non-diagonal)
      - intrinsic dispersion diag(sig_r^2, sig_t^2, sig_p^2)
      - streaming means mu = (mu_r, mu_t, mu_p) from splines

    Penalized log-likelihood:
      total = data_loglike - 0.5 * sum_i lam_i * p_i^T R_i p_i

    loglike params:
      params: (6*nk,) in physical units, layout
              [logsig_r, logsig_t, logsig_p, mean_r, mean_t, mean_p]
    """

    def loglike(params: jnp.ndarray, data: dict) -> jnp.ndarray:
        x_f = data["x_f"]  # (NM,3)
        Cx_f = data["Cx_f"]  # (NM,3,3)
        logw = data["logw"]  # (N,M)

        N, M = logw.shape  # static at trace time

        # 6 spline coefficient blocks (each length nk)
        P = params.reshape(6, nk)  # [lsig_r, lsig_t, lsig_p, mu_r, mu_t, mu_p]

        # Evaluate all spline fields in one matmul
        vals = B_jax @ P.T  # (NM, 6)

        lsr = vals[:, 0]
        lst = vals[:, 1]
        lsp = vals[:, 2]
        mu_r = vals[:, 3]
        mu_t = vals[:, 4]
        mu_p = vals[:, 5]

        # intrinsic variances
        sig_r2 = jnp.exp(2.0 * lsr)
        sig_t2 = jnp.exp(2.0 * lst)
        sig_p2 = jnp.exp(2.0 * lsp)

        # symmetric elements of Lambda = Cx + diag(sig^2)
        a = Cx_f[:, 0, 0] + sig_r2
        d = Cx_f[:, 1, 1] + sig_t2
        f = Cx_f[:, 2, 2] + sig_p2

        b = Cx_f[:, 1, 0]
        c = Cx_f[:, 2, 0]
        e = Cx_f[:, 2, 1]

        # center data by streaming means
        x1 = x_f[:, 0] - mu_r
        x2 = x_f[:, 1] - mu_t
        x3 = x_f[:, 2] - mu_p

        # explicit cholesky + forward solve
        l11, l21, l31, l22, l32, l33, ok = chol3x3_from_sym(a, b, c, d, e, f)
        y1, y2, y3 = forward_solve_lower3(l11, l21, l31, l22, l32, l33, x1, x2, x3)

        quad = y1 * y1 + y2 * y2 + y3 * y3
        logdet = 2.0 * (jnp.log(l11) + jnp.log(l22) + jnp.log(l33))

        ll_f = -0.5 * (quad + logdet)  # (NM,) up to additive constant
        ll = ll_f.reshape(N, M)  # (N,M)

        # distance marginalisation
        logp_star = jsp_special.logsumexp(logw + ll, axis=1)  # (N,)
        total = jnp.sum(logp_star)

        # Smoothness penalty on spline coefficient blocks
        # Penalizing der=2 of the spline makes the *derivative* smoother.
        q = jnp.einsum("ik,ikl,il->i", P, R_blocks_jax, P)  # (6,)
        smooth_pen = 0.5 * jnp.sum(lam_smooth_jax * q)
        total = total - smooth_pen

        ok_all = jnp.all(ok) & jnp.isfinite(total)
        return jnp.where(ok_all, total, -jnp.inf)

    return loglike


def setup_and_fit_obs_mean_smooth(
    sample_data,
    scfg,
    vdisp_min=20.0,
    vdisp_max=400.0,
    # --- smoothness controls ---
    smooth_lambda_sigma: float = 1.0,  # same λ for 3 log-sigma splines
    smooth_lambda_mean: float = 0.05,  # same λ for 3 mean splines
    smooth_der_order: int = 2,  # der=2 => smooth first derivative
    smooth_grid_size: int = 512,
    smooth_add_diag_eps: float = 0.0,
    **mcmc_kwargs,
):
    """
    sample_data must contain:
      logr:        (N,M)
      vel_gal:     (N,M,3)
      err_mat_gal: (N,M,3,3) symmetric
      log_dist_w:  (N,M)
    """
    logr = np.asarray(sample_data["logr"], dtype=np.float64)  # (N,M)
    x = np.asarray(sample_data["vel_gal"], dtype=np.float64)  # (N,M,3)
    Cx = np.asarray(sample_data["err_mat_gal"], dtype=np.float64)  # (N,M,3,3)
    logw = np.asarray(sample_data["log_dist_w"], dtype=np.float64)  # (N,M)

    N, M = logr.shape
    NM = N * M

    logr_f = logr.reshape(NM)
    x_f = x.reshape(NM, 3)
    Cx_f = Cx.reshape(NM, 3, 3)

    nk = int(scfg.num_knots)
    knots = np.asarray(scfg.knots_logr, dtype=np.float64)
    assert knots.shape[0] == nk

    # basis on flattened logr samples
    B = build_agama_spline_basis_1d(logr_f, knots)  # (NM,nk)

    # to JAX
    B_j = jnp.asarray(B)  # (NM,nk)
    data_j = {
        "x_f": jnp.asarray(x_f),  # (NM,3)
        "Cx_f": jnp.asarray(Cx_f),  # (NM,3,3)
        "logw": jnp.asarray(logw),  # (N,M)
    }

    # --------- smoothness setup (precompute once in numpy/agama, use in JAX) ---------

    lam_blocks = np.array(
        [smooth_lambda_sigma] * 3 + [smooth_lambda_mean] * 3,
        dtype=np.float64,
    )

    # Penalize the derivative smoothness of fit functions:
    # use der=2 so the FIRST derivative is smooth.
    R = build_agama_spline_roughness_matrix(
        knots=knots,
        der=int(smooth_der_order),
        ngrid=int(smooth_grid_size),
        # add_diag_eps=float(smooth_add_diag_eps),
    )  # (nk,nk)

    # same roughness matrix for each spline block; different λ allowed
    R_blocks = np.repeat(R[None, :, :], 6, axis=0)  # (6,nk,nk)

    R_blocks_j = jnp.asarray(R_blocks)
    lam_smooth_j = jnp.asarray(lam_blocks)

    print("Smoothness penalty enabled")
    print("  smooth_der_order:", smooth_der_order)
    print("  lambdas:", lam_blocks)

    # lam_smooth_jax = jnp.ones((6,), dtype=B_jax.dtype)

    loglike_jax = make_loglike_3d_obs_jax_mean_smooth(
        B_j,
        nk,
        R_blocks_jax=R_blocks_j,
        lam_smooth_jax=lam_smooth_j,
    )

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
