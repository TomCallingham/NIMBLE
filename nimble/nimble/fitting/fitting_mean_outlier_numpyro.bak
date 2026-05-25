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

from .fitting_mean_decline import build_agama_spline_basis_deriv_1d


import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


# ---------------------------------------------------------------------------
# Log-likelihood: declining + outlier mixture
# ---------------------------------------------------------------------------


def make_loglike_3d_obs_jax_mean_declining_outlier(
    B_jax: jnp.ndarray,  # (NM, nk)  value basis
    B_der_jax: jnp.ndarray,  # (Np, nk)  first-derivative basis at penalty points
    nk: int,
    lam_mono: float = 1e2,
    sig_outlier: float = 600.0,
):
    """
    Identical to make_loglike_3d_obs_jax_mean_declining, but mixes in a
    fixed broad Gaussian outlier component:

        log p(x_n) = log[ (1 - f_out) * p_main(x_n) + f_out * p_out(x_n) ]

    where p_out is a zero-mean Gaussian with sigma=sig_outlier (in km/s) in
    all three velocity components, on top of per-star measurement errors.

    f_outlier is read from data["f_outlier"] — a JAX scalar sampled by the
    NumPyro model, so it is fully traceable without touching the params vector.

    params layout is unchanged: [lsig_r | lsig_t | lsig_p | mu_r | mu_t | mu_p]
    Total length: 6 * nk  (same as the non-outlier version).

    Note: the variable for the (3,3) symmetric matrix element is renamed `fcc`
    here to avoid collision with `f_out`.
    """
    sig_out2 = sig_outlier**2
    log_2pi_3_over_2 = 1.5 * jnp.log(2.0 * jnp.pi)  # constant term in 3D Gaussian

    def loglike(params: jnp.ndarray, data: dict) -> jnp.ndarray:
        x_f = data["x_f"]  # (NM, 3)
        Cx_f = data["Cx_f"]  # (NM, 3, 3)
        logw = data["logw"]  # (N, M)
        f_out = data["f_outlier"]  # scalar in (0, 1)

        N, M = logw.shape

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

        # ------------------------------------------------------------------
        # Main component  Λ = Cx + diag(σ²),  mean = (μ_r, μ_t, μ_p)
        # ------------------------------------------------------------------
        a = Cx_f[:, 0, 0] + sig_r2
        d = Cx_f[:, 1, 1] + sig_t2
        fcc = Cx_f[:, 2, 2] + sig_p2  # renamed from `f` to avoid name clash
        b = Cx_f[:, 1, 0]
        c = Cx_f[:, 2, 0]
        e = Cx_f[:, 2, 1]

        x1 = x_f[:, 0] - mu_r
        x2 = x_f[:, 1] - mu_t
        x3 = x_f[:, 2] - mu_p

        l11, l21, l31, l22, l32, l33, ok = chol3x3_from_sym(a, b, c, d, e, fcc)
        y1, y2, y3 = forward_solve_lower3(l11, l21, l31, l22, l32, l33, x1, x2, x3)

        quad_main = y1 * y1 + y2 * y2 + y3 * y3
        logdet_main = 2.0 * (jnp.log(l11) + jnp.log(l22) + jnp.log(l33))
        ll_main_f = -0.5 * (quad_main + logdet_main) - log_2pi_3_over_2  # (NM,)

        # ------------------------------------------------------------------
        # Outlier component  Λ_out = Cx + sig_outlier² * I,  mean = 0
        # Off-diagonal measurement-error terms are retained.
        # ------------------------------------------------------------------
        a_o = Cx_f[:, 0, 0] + sig_out2
        d_o = Cx_f[:, 1, 1] + sig_out2
        fcc_o = Cx_f[:, 2, 2] + sig_out2

        # b, c, e (off-diagonals of Cx) are unchanged — reuse from above
        l11_o, l21_o, l31_o, l22_o, l32_o, l33_o, ok_o = chol3x3_from_sym(
            a_o, b, c, d_o, e, fcc_o
        )
        y1_o, y2_o, y3_o = forward_solve_lower3(
            l11_o,
            l21_o,
            l31_o,
            l22_o,
            l32_o,
            l33_o,
            x_f[:, 0],
            x_f[:, 1],
            x_f[:, 2],  # zero mean
        )

        quad_out = y1_o * y1_o + y2_o * y2_o + y3_o * y3_o
        logdet_out = 2.0 * (jnp.log(l11_o) + jnp.log(l22_o) + jnp.log(l33_o))
        ll_out_f = -0.5 * (quad_out + logdet_out) - log_2pi_3_over_2  # (NM,)

        # ------------------------------------------------------------------
        # Marginalise over distance samples, then mix
        # ------------------------------------------------------------------
        ll_main = ll_main_f.reshape(N, M)
        ll_out = ll_out_f.reshape(N, M)

        # log p_main(x_n) and log p_out(x_n) after distance marginalisation
        logp_main = jsp_special.logsumexp(logw + ll_main, axis=1)  # (N,)
        logp_out = jsp_special.logsumexp(logw + ll_out, axis=1)  # (N,)

        # Numerically stable log-mixture via logsumexp over two branches
        log_1mf = jnp.log1p(-f_out)
        log_f = jnp.log(f_out)

        branches = jnp.stack([log_1mf + logp_main, log_f + logp_out], axis=1)  # (N, 2)
        logp_star = jsp_special.logsumexp(branches, axis=1)  # (N,)
        total = jnp.sum(logp_star)

        # ------------------------------------------------------------------
        # One-sided monotonicity penalty (main component only)
        # ------------------------------------------------------------------
        derivs = B_der_jax @ P[:3].T  # (Np, 3)
        violations = jnp.maximum(0.0, derivs)
        mono_pen = lam_mono * jnp.sum(violations**2)
        total = total - mono_pen

        ok_all = jnp.all(ok) & jnp.all(ok_o) & jnp.isfinite(total)
        return jnp.where(ok_all, total, -jnp.inf)

    return loglike


# ---------------------------------------------------------------------------
# NumPyro NUTS fitter — self-contained, adds f_outlier sample site
# ---------------------------------------------------------------------------
import time


def fit_numpyro_nuts_mean_outlier(
    loglike_fn,
    nk: int,
    data: dict,
    vdisp_min: float = 20.0,
    vdisp_max: float = 400.0,
    vmean_max: float = 50.0,
    f_outlier_alpha: float = 1.0,
    f_outlier_beta: float = 500.0,
    num_warmup: int = 400,
    num_samples: int = 1200,
    num_chains: int = 1,
    rng_seed: int = 0,
    target_accept_prob: float = 0.7,
    max_tree_depth: int | None = 7,
    start_x=None,
    start_mu=None,
    progress_bar: bool = True,
):
    print("Fitting with mean + outlier mixture...")
    dt = time.time()

    log_min = float(np.log(vdisp_min))
    log_max = float(np.log(vdisp_max))

    dim_sigma = 3 * int(nk)
    dim_mean = 3 * int(nk)

    def _prep_init(x, dim, default_value, clip_low=None, clip_high=None):
        x = x if x is not None else np.full(dim, default_value, dtype=float)
        x = jnp.asarray(x, dtype=jnp.float64)
        if x.shape == (dim,):
            if num_chains > 1:
                x = jnp.broadcast_to(x, (int(num_chains), dim))
        elif x.shape == (int(num_chains), dim):
            pass
        else:
            raise ValueError(
                f"init must have shape {(dim,)} or {(int(num_chains), dim)}, got {x.shape}"
            )
        if (clip_low is not None) or (clip_high is not None):
            eps = 1e-12
            lo = -jnp.inf if clip_low is None else (clip_low + eps)
            hi = jnp.inf if clip_high is None else (clip_high - eps)
            x = jnp.clip(x, lo, hi)
        return x

    x0_sigma = _prep_init(
        start_x,
        dim_sigma,
        default_value=4.0,
        clip_low=log_min,
        clip_high=log_max,
    )
    x0_mean = _prep_init(
        start_mu,
        dim_mean,
        default_value=0.0,
        clip_low=-vmean_max,
        clip_high=+vmean_max,
    )

    # f_outlier init: prior mean of Beta(alpha, beta)
    f0 = float(f_outlier_alpha) / (float(f_outlier_alpha) + float(f_outlier_beta))
    f0 = np.clip(f0, 1e-6, 1.0 - 1e-6)
    if num_chains > 1:
        f0_arr = jnp.full((int(num_chains),), f0, dtype=jnp.float64)
    else:
        f0_arr = jnp.asarray(f0, dtype=jnp.float64)

    init_params = {
        "params_sigma": x0_sigma,
        "params_mean": x0_mean,
        "f_outlier": f0_arr,
    }

    def model():
        low_sigma = jnp.full((dim_sigma,), log_min, dtype=jnp.float64)
        high_sigma = jnp.full((dim_sigma,), log_max, dtype=jnp.float64)
        low_mean = jnp.full((dim_mean,), -vmean_max, dtype=jnp.float64)
        high_mean = jnp.full((dim_mean,), +vmean_max, dtype=jnp.float64)

        params_sigma = numpyro.sample(
            "params_sigma", dist.Uniform(low_sigma, high_sigma)
        )
        params_mean = numpyro.sample("params_mean", dist.Uniform(low_mean, high_mean))
        f_outlier = numpyro.sample(
            "f_outlier", dist.Beta(f_outlier_alpha, f_outlier_beta)
        )

        params = jnp.concatenate([params_sigma, params_mean])
        data_with_f = {**data, "f_outlier": f_outlier}
        ll = loglike_fn(params, data_with_f)
        numpyro.factor("loglike", ll)

    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )

    print(f"num_chains: {num_chains}")
    chain_method = "vectorized" if int(num_chains) > 1 else "sequential"

    mcmc = MCMC(
        kernel,
        num_warmup=int(num_warmup),
        num_samples=int(num_samples),
        num_chains=int(num_chains),
        progress_bar=progress_bar,
        chain_method=chain_method,
    )

    rng_key = jax.random.PRNGKey(int(rng_seed))
    mcmc.run(rng_key, init_params=init_params)

    raw_samples = mcmc.get_samples(group_by_chain=False)

    print("Fit complete!")
    print(f"Time taken: {(time.time() - dt) / 60:.3f} mins")

    return mcmc, raw_samples


def setup_and_fit_obs_mean_declining_outlier(
    sample_data,
    scfg,
    vdisp_min: float = 20.0,
    vdisp_max: float = 400.0,
    lam_mono: float = 1e2,
    r_mono_min: float = 30.0,
    sig_outlier: float = 600.0,
    f_outlier_alpha: float = 1.0,
    f_outlier_beta: float = 100.0,
    **mcmc_kwargs,
):
    """
    Extends setup_and_fit_obs_mean_declining with a fixed broad-Gaussian
    outlier mixture component.

    Extra parameters
    ----------------
    sig_outlier      : dispersion (km/s) of the outlier Gaussian (default 400)
    f_outlier_alpha  : Beta prior shape α for f_outlier (default 1)
    f_outlier_beta   : Beta prior shape β for f_outlier (default 10)
                       E[f] = α/(α+β); default ≈ 9 %

    Returns
    -------
    mcmc, sigma_samples, mean_samples, f_outlier_samples, multi_disp
    """
    logr = np.asarray(sample_data["logr"], dtype=np.float64)
    x = np.asarray(sample_data["vel_gal"], dtype=np.float64)
    Cx = np.asarray(sample_data["err_mat_gal"], dtype=np.float64)
    logw = np.asarray(sample_data["log_dist_w"], dtype=np.float64)

    N, M = logr.shape
    NM = N * M

    logr_f = logr.reshape(NM)
    x_f = x.reshape(NM, 3)
    Cx_f = Cx.reshape(NM, 3, 3)

    nk = int(scfg.num_knots)
    knots = np.asarray(scfg.knots_logr, dtype=np.float64)
    assert knots.shape[0] == nk, f"Expected {nk} knots, got {knots.shape[0]}"

    # Value basis
    B = build_agama_spline_basis_1d(logr_f, knots)
    B_j = jnp.asarray(B)

    # Derivative basis (knots + midpoints, clipped to r > r_mono_min)
    knot_mids = 0.5 * (knots[:-1] + knots[1:])
    x_pen_all = np.sort(np.concatenate([knots, knot_mids]))
    x_pen = x_pen_all[x_pen_all > np.log(r_mono_min)]
    B_der = build_agama_spline_basis_deriv_1d(x_pen, knots, der=1)
    B_der_j = jnp.asarray(B_der)

    print(
        f"Monotonicity penalty: {x_pen.size} evaluation points, "
        f"lam_mono={lam_mono:.3g}, r_mono_min={r_mono_min} kpc"
    )
    print(
        f"Outlier component: sig_outlier={sig_outlier} km/s, "
        f"Beta({f_outlier_alpha}, {f_outlier_beta}) prior"
    )

    data_j = {
        "x_f": jnp.asarray(x_f),
        "Cx_f": jnp.asarray(Cx_f),
        "logw": jnp.asarray(logw),
        # f_outlier is NOT in data_j here — it is injected by fit_numpyro_nuts_mean_outlier
        # as a sampled JAX scalar before each loglike evaluation.
    }

    loglike_jax = make_loglike_3d_obs_jax_mean_declining_outlier(
        B_j,
        B_der_j,
        nk,
        lam_mono=float(lam_mono),
        sig_outlier=float(sig_outlier),
    )

    mcmc, raw_samples = fit_numpyro_nuts_mean_outlier(
        loglike_jax,
        nk=nk,
        data=data_j,
        vdisp_min=vdisp_min,
        vdisp_max=vdisp_max,
        f_outlier_alpha=f_outlier_alpha,
        f_outlier_beta=f_outlier_beta,
        **mcmc_kwargs,
    )

    sigma_samples = raw_samples["params_sigma"]
    mean_samples = raw_samples["params_mean"]
    f_outlier_samples = raw_samples["f_outlier"]

    multi_disp = DispersionMeanMultiModel3D.from_sampler_3d(
        sigma_samples, mean_samples, scfg.knots_logr
    )
    return mcmc, sigma_samples, mean_samples, f_outlier_samples, multi_disp


# import agama
# import jax.numpy as jnp
# import jax.scipy.special as jsp_special

# from nimble.models import DispersionMeanMultiModel3D
# from nimble.fitting.fitting_numpyro import (
#     build_agama_spline_basis_1d,
#     chol3x3_from_sym,
#     forward_solve_lower3,
# )

# from nimble.fitting.fitting_mean_decline import build_agama_spline_basis_deriv_1d


# import numpyro
# import numpyro.distributions as dist
# from numpyro.infer import MCMC, NUTS


def compute_outlier_probabilities(
    sample_data: dict,
    scfg,
    sigma_samples: np.ndarray,  # (S, 3*nk)
    mean_samples: np.ndarray,  # (S, 3*nk)
    f_outlier_samples: np.ndarray,  # (S,)
    sig_outlier: float = 600.0,
) -> np.ndarray:
    """
    Per-star posterior mean outlier responsibility, computed via vmap over
    posterior samples.

    Returns
    -------
    p_out_star : (N,)  posterior mean outlier probability per star
    """
    logr = np.asarray(sample_data["logr"], dtype=np.float64)
    x = np.asarray(sample_data["vel_gal"], dtype=np.float64)
    Cx = np.asarray(sample_data["err_mat_gal"], dtype=np.float64)
    logw = np.asarray(sample_data["log_dist_w"], dtype=np.float64)

    N, M = logr.shape
    NM = N * M
    nk = int(scfg.num_knots)
    knots = np.asarray(scfg.knots_logr, dtype=np.float64)

    logr_f = logr.reshape(NM)
    x_f = jnp.asarray(x.reshape(NM, 3))
    Cx_f = jnp.asarray(Cx.reshape(NM, 3, 3))
    logw_j = jnp.asarray(logw)  # (N, M)

    B_j = jnp.asarray(build_agama_spline_basis_1d(logr_f, knots))  # (NM, nk)

    # Stack posterior draws — shape (S, 6*nk) and (S,)
    params_all = jnp.asarray(
        np.concatenate([sigma_samples, mean_samples], axis=1)
    )  # (S, 6*nk)
    f_all = jnp.asarray(f_outlier_samples)  # (S,)

    sig_out2 = sig_outlier**2
    log_2pi_32 = 1.5 * jnp.log(2.0 * jnp.pi)

    def responsibility_one_sample(params: jnp.ndarray, f_s: jnp.ndarray):
        """
        Compute per-star outlier responsibility for a single posterior draw.
        Returns shape (N,).
        """
        P = params.reshape(6, nk)
        vals = B_j @ P.T  # (NM, 6)

        lsr, lst, lsp = vals[:, 0], vals[:, 1], vals[:, 2]
        mu_r, mu_t, mu_p = vals[:, 3], vals[:, 4], vals[:, 5]

        sig_r2 = jnp.exp(2.0 * lsr)
        sig_t2 = jnp.exp(2.0 * lst)
        sig_p2 = jnp.exp(2.0 * lsp)

        # --- main component ---
        a = Cx_f[:, 0, 0] + sig_r2
        d = Cx_f[:, 1, 1] + sig_t2
        fcc = Cx_f[:, 2, 2] + sig_p2
        b = Cx_f[:, 1, 0]
        c = Cx_f[:, 2, 0]
        e = Cx_f[:, 2, 1]

        l11, l21, l31, l22, l32, l33, _ = chol3x3_from_sym(a, b, c, d, e, fcc)
        y1, y2, y3 = forward_solve_lower3(
            l11,
            l21,
            l31,
            l22,
            l32,
            l33,
            x_f[:, 0] - mu_r,
            x_f[:, 1] - mu_t,
            x_f[:, 2] - mu_p,
        )
        quad_m = y1 * y1 + y2 * y2 + y3 * y3
        logdet_m = 2.0 * (jnp.log(l11) + jnp.log(l22) + jnp.log(l33))
        ll_main = (-0.5 * (quad_m + logdet_m) - log_2pi_32).reshape(N, M)

        # --- outlier component (zero mean, sig_outlier on diagonal) ---
        a_o = Cx_f[:, 0, 0] + sig_out2
        d_o = Cx_f[:, 1, 1] + sig_out2
        fcc_o = Cx_f[:, 2, 2] + sig_out2

        l11_o, l21_o, l31_o, l22_o, l32_o, l33_o, _ = chol3x3_from_sym(
            a_o, b, c, d_o, e, fcc_o
        )
        y1_o, y2_o, y3_o = forward_solve_lower3(
            l11_o,
            l21_o,
            l31_o,
            l22_o,
            l32_o,
            l33_o,
            x_f[:, 0],
            x_f[:, 1],
            x_f[:, 2],
        )
        quad_o = y1_o * y1_o + y2_o * y2_o + y3_o * y3_o
        logdet_o = 2.0 * (jnp.log(l11_o) + jnp.log(l22_o) + jnp.log(l33_o))
        ll_out = (-0.5 * (quad_o + logdet_o) - log_2pi_32).reshape(N, M)

        # --- marginalise over distance samples ---
        logp_main = jsp_special.logsumexp(logw_j + ll_main, axis=1)  # (N,)
        logp_out = jsp_special.logsumexp(logw_j + ll_out, axis=1)  # (N,)

        # --- responsibility  r = f*p_out / ((1-f)*p_main + f*p_out) ---
        log_num = jnp.log(f_s) + logp_out
        log_den = jnp.logaddexp(
            jnp.log1p(-f_s) + logp_main,
            jnp.log(f_s) + logp_out,
        )
        return jnp.exp(log_num - log_den)  # (N,)

    # vmap over S samples, then average
    responsibilities = jax.vmap(responsibility_one_sample)(params_all, f_all)  # (S, N)
    return np.asarray(responsibilities.mean(axis=0))  # (N,):w
