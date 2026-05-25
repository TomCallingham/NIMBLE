"""
fitting_mean_outlier_jeans_numpyro.py

Mean + outlier-mixture kinematic likelihood with an *M_enc monotonicity*
penalty derived from the spherical Jeans equation.

This is the same likelihood as setup_and_fit_obs_mean_declining_outlier
(broad-Gaussian outlier component, distance marginalisation, full 3-D
measurement covariance), except the σ-decline penalty is replaced by a
one-sided penalty on non-monotonic M(<r).

The penalty is evaluated on a uniform-in-log-r grid between
scfg.min_knot and scfg.max_knot. With guard knots configured so that
knots_logr extends one interval beyond on each side, this is exactly
knots_logr[1] .. knots_logr[-2] — the guard intervals are excluded.

Jeans equation (no streaming-mean term in M_enc itself):

    v²_r       = σ_r² + μ_r²
    v²_θ       = σ_θ² + μ_θ²
    v²_φ       = σ_φ² + μ_φ²
    β          = 1 − (v²_θ + v²_φ) / (2 v²_r)
    dln v²_r   = (2 σ_r² · dlsr + 2 μ_r · dμ_r) / v²_r            (d/dlnr)
    M_enc(r)   = −(r v²_r / G) · (dln ρ + dln v²_r + 2 β)

The monotonicity penalty is the cumulative-max-deficit in log space:

    deficit[i] = max_{j ≤ i} ln M[j]  −  ln M[i]      (≥ 0)
    pen        = lam_mono_M · Σ_i deficit[i]²

with M floored at M_floor (= 1 M⊙) before logging so the expression is
finite for non-physical (≤ 0) samples. A valley of width w and depth Δ
contributes O(w · Δ²) rather than O(Δ²) for a per-pair penalty, so wide
wobbles can't dilute themselves into many small unpenalised steps.
"""

import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jsp_special

from nimble.models import DispersionMeanMultiModel3D
from nimble.fitting.fitting_numpyro import (
    build_agama_spline_basis_1d,
    chol3x3_from_sym,
    forward_solve_lower3,
)
from nimble.fitting.fitting_mean_decline import build_agama_spline_basis_deriv_1d
from nimble.fitting.fitting_mean_outlier_numpyro import fit_numpyro_nuts_mean_outlier


G_KMS_KPC = 4.3e-6  # (kpc km²) / (s² M⊙)


# ---------------------------------------------------------------------------
# Log-likelihood: mean + outlier mixture, with M_enc monotonicity penalty
# ---------------------------------------------------------------------------


def make_loglike_3d_obs_jax_mean_mono_outlier_jeans(
    B_jax: jnp.ndarray,  # (NM, nk)  value basis at data
    B_pen_jax: jnp.ndarray,  # (Np, nk)  value basis at penalty points
    B_pen_der_jax: jnp.ndarray,  # (Np, nk)  d/dlnr basis at penalty points
    r_pen_jax: jnp.ndarray,  # (Np,)     r at penalty points (kpc)
    dln_rho_pen_jax: jnp.ndarray,  # (Np,)     dln ρ / dln r at penalty points
    nk: int,
    lam_mono_M: float = 1e4,
    penalty_power: float = 2.0,
    aggregator: str = "sum_squared",
    sig_outlier: float = 600.0,
    M_floor: float = 1.0,  # M⊙; floor before log to keep gradients alive
):
    """
    Build the JAX log-likelihood. Layout of `params` is unchanged from the
    non-Jeans version:  [lsig_r | lsig_t | lsig_p | mu_r | mu_t | mu_p]
    of total length 6·nk.

    `f_outlier` is read from data["f_outlier"] — a sampled JAX scalar passed
    in by the NumPyro model.
    """
    sig_out2 = sig_outlier**2
    log_2pi_3_over_2 = 1.5 * jnp.log(2.0 * jnp.pi)

    def loglike(params: jnp.ndarray, data: dict) -> jnp.ndarray:
        x_f = data["x_f"]  # (NM, 3)
        Cx_f = data["Cx_f"]  # (NM, 3, 3)
        logw = data["logw"]  # (N, M)
        f_out = data["f_outlier"]

        N, M = logw.shape

        P = params.reshape(6, nk)  # rows: lsr, lst, lsp, mu_r, mu_t, mu_p

        # ------------------------------------------------------------------
        # Spline values at data points
        # ------------------------------------------------------------------
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
        # Main component:  Λ = Cx + diag(σ²),  mean = (μ_r, μ_t, μ_p)
        # ------------------------------------------------------------------
        a = Cx_f[:, 0, 0] + sig_r2
        d = Cx_f[:, 1, 1] + sig_t2
        fcc = Cx_f[:, 2, 2] + sig_p2
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
        ll_main_f = -0.5 * (quad_main + logdet_main) - log_2pi_3_over_2

        # ------------------------------------------------------------------
        # Outlier component:  Λ_out = Cx + sig_outlier² · I,  mean = 0
        # Off-diagonal measurement-error terms are retained.
        # ------------------------------------------------------------------
        a_o = Cx_f[:, 0, 0] + sig_out2
        d_o = Cx_f[:, 1, 1] + sig_out2
        fcc_o = Cx_f[:, 2, 2] + sig_out2

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
            x_f[:, 2],
        )
        quad_out = y1_o * y1_o + y2_o * y2_o + y3_o * y3_o
        logdet_out = 2.0 * (jnp.log(l11_o) + jnp.log(l22_o) + jnp.log(l33_o))
        ll_out_f = -0.5 * (quad_out + logdet_out) - log_2pi_3_over_2

        # ------------------------------------------------------------------
        # Distance marginalisation, then mixture
        # ------------------------------------------------------------------
        ll_main = ll_main_f.reshape(N, M)
        ll_out = ll_out_f.reshape(N, M)

        logp_main = jsp_special.logsumexp(logw + ll_main, axis=1)  # (N,)
        logp_out = jsp_special.logsumexp(logw + ll_out, axis=1)  # (N,)

        log_1mf = jnp.log1p(-f_out)
        log_f = jnp.log(f_out)
        branches = jnp.stack([log_1mf + logp_main, log_f + logp_out], axis=1)  # (N, 2)
        logp_star = jsp_special.logsumexp(branches, axis=1)  # (N,)
        total = jnp.sum(logp_star)

        # ------------------------------------------------------------------
        # M_enc monotonicity penalty (Jeans equation on the penalty grid)
        # ------------------------------------------------------------------
        vals_p = B_pen_jax @ P.T  # (Np, 6)
        lsr_p = vals_p[:, 0]
        lst_p = vals_p[:, 1]
        lsp_p = vals_p[:, 2]
        mu_r_p = vals_p[:, 3]
        mu_t_p = vals_p[:, 4]
        mu_p_p = vals_p[:, 5]

        sig_r2_p = jnp.exp(2.0 * lsr_p)
        sig_t2_p = jnp.exp(2.0 * lst_p)
        sig_p2_p = jnp.exp(2.0 * lsp_p)

        v2_r_p = sig_r2_p + mu_r_p**2
        v2_t_p = sig_t2_p + mu_t_p**2
        v2_p_p = sig_p2_p + mu_p_p**2

        beta_p = 1.0 - (v2_t_p + v2_p_p) / (2.0 * v2_r_p)

        # d/dlnr applied to log-σ and to μ
        derivs_p = B_pen_der_jax @ P.T  # (Np, 6)
        dlsr_p = derivs_p[:, 0]
        dmu_r_p = derivs_p[:, 3]

        # d ln v²_r / d ln r  (chain rule, since v²_r = exp(2 lsr) + μ_r²)
        dln_v2r_p = (2.0 * sig_r2_p * dlsr_p + 2.0 * mu_r_p * dmu_r_p) / v2_r_p

        M_pen = -(r_pen_jax * v2_r_p / G_KMS_KPC) * (
            dln_rho_pen_jax + dln_v2r_p + 2.0 * beta_p
        )

        # Cumulative-max-deficit penalty in log space.
        # Floor M at M_floor before logging (gradients still flow above floor;
        # negative / sub-floor M_enc collapses to log(M_floor) and produces a
        # large deficit against any preceding higher value, so unphysical
        # samples are penalised the same way as ordinary wobbles).
        #
        # For each i:  deficit[i] = max_{j ≤ i} log M[j]  −  log M[i]   ≥ 0
        #
        # Two aggregator options:
        # "sum_squared"  (default):  pen = lam · (Σ_i deficit[i])²
        #   Smooth gradient w.r.t. parameters (no kink at zero deficit beyond
        #   the cummax tie-set, which is measure zero), so NUTS stays fast.
        #   A valley of width w and depth Δ contributes (w·Δ)² — broad
        #   wobbles get squared *after* aggregation, attacking them hard.
        # "sum_pow":  pen = lam · Σ_i deficit[i]**penalty_power
        #   Per-point penalty. With penalty_power=2 this is the standard
        #   monotonicity loss; with penalty_power=1 the kink at zero
        #   deficit makes NUTS very slow (max-tree-depth saturation).
        log_M = jnp.log(jnp.maximum(M_pen, M_floor))
        log_M_running_max = jax.lax.cummax(log_M, axis=0)
        deficits = log_M_running_max - log_M  # ≥ 0 by construction

        if aggregator == "sum_squared":
            mono_pen_M = lam_mono_M * (jnp.sum(deficits)) ** 2
        elif aggregator == "sum_pow":
            mono_pen_M = lam_mono_M * jnp.sum(deficits**penalty_power)
        else:
            raise ValueError(
                f"unknown aggregator {aggregator!r}; "
                f"expected 'sum_squared' or 'sum_pow'"
            )

        total = total - mono_pen_M

        ok_all = jnp.all(ok) & jnp.all(ok_o) & jnp.isfinite(total)
        return jnp.where(ok_all, total, -jnp.inf)

    return loglike


# ---------------------------------------------------------------------------
# Setup + fit driver
# ---------------------------------------------------------------------------


def setup_and_fit_obs_mean_mono_outlier(
    sample_data,
    scfg,
    func_d_log_dens,  # callable: (lr,) -> dln ρ / dln r
    vdisp_min: float = 20.0,
    vdisp_max: float = 400.0,
    lam_mono_M: float = 1e3,
    penalty_power: float = 2.0,
    # aggregator: str = "sum_squared",
    aggregator: str = "sum_pow",
    n_pen_M: int = 50,
    sig_outlier: float = 600.0,
    f_outlier_alpha: float = 1.0,
    f_outlier_beta: float = 500.0,
    M_floor: float = 1.0,
    target_accept_prob=0.90,
    **mcmc_kwargs,
):
    """
    Fit kinematic data with the mean+outlier mixture and an M_enc
    monotonicity penalty derived from the Jeans equation.

    Parameters
    ----------
    sample_data : dict with keys logr, vel_gal, err_mat_gal, log_dist_w
        Same shapes as the existing fitters: (N, M), (N, M, 3),
        (N, M, 3, 3), (N, M).
    scfg : spline config exposing num_knots, knots_logr, min_knot, max_knot.
        With guard knots, knots_logr[0] < log(min_knot) = knots_logr[1] and
        knots_logr[-2] = log(max_knot) < knots_logr[-1]; the penalty grid
        therefore lives entirely inside the trustworthy interior.
    func_d_log_dens : callable taking log-r values and returning
        dln ρ / dln r. Pre-evaluated once on the penalty grid; not traced
        through JAX.
    vdisp_min, vdisp_max : sigma bounds (km/s).
    lam_mono_M : strength of the monotonicity penalty in log space.
        Default 1e3.
    penalty_power : exponent for the per-point aggregator. Only used when
        aggregator="sum_pow". Default 2.0 (quadratic).
    aggregator : how to combine per-point deficits. Default "sum_squared":
        pen = lam · (Σ deficits)². Smooth gradient (no kink at zero
        deficit), broad wobbles get squared *after* aggregation so they
        attack hard. "sum_pow" is the alternative: pen = lam · Σ
        deficits**penalty_power; with power=2 this is the standard
        per-point quadratic, with power=1 it's linear (sharp kink, very
        slow NUTS — not recommended).
    n_pen_M : number of grid points between min_knot and max_knot.
    sig_outlier, f_outlier_alpha, f_outlier_beta : as in the outlier fitter.
    M_floor : M⊙; floor applied before logging M_enc.

    Returns
    -------
    diagnostics, sigma_samples, mean_samples, f_outlier_samples, multi_disp
        diagnostics is the dict returned by the inner NumPyro driver
        (num_divergences, mean_tree_depth, etc.). Same shape as
        setup_and_fit_obs_mean_declining_outlier.
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

    # ---- value basis at data ----
    B = build_agama_spline_basis_1d(logr_f, knots)
    B_j = jnp.asarray(B)

    # ---- M_enc penalty grid: between min_knot and max_knot ----
    # With guard knots these are knots_logr[1] and knots_logr[-2], so the
    # guard intervals at each end are excluded from the penalty.
    log_min = float(np.log(scfg.min_knot))
    log_max = float(np.log(scfg.max_knot))
    if log_min >= log_max:
        raise ValueError(
            f"min_knot ({scfg.min_knot}) must be < max_knot ({scfg.max_knot})"
        )
    x_pen = np.linspace(log_min, log_max, int(n_pen_M))
    r_pen = np.exp(x_pen)

    # consistency check with guard-knot convention (informational only)
    if not (np.isclose(x_pen[0], knots[1]) and np.isclose(x_pen[-1], knots[-2])):
        # not necessarily wrong, but worth flagging
        print(
            "Note: penalty grid endpoints don't align with knots[1]/knots[-2]."
            "  If guard knots are in use this should hold; check scfg setup."
        )

    B_pen = build_agama_spline_basis_1d(x_pen, knots)
    B_pen_der = build_agama_spline_basis_deriv_1d(x_pen, knots, der=1)

    # ---- density log-derivative at penalty points (precomputed, fixed) ----
    dln_rho_pen = np.asarray(func_d_log_dens(x_pen), dtype=np.float64)
    if dln_rho_pen.shape != (int(n_pen_M),):
        raise ValueError(
            f"func_d_log_dens returned shape {dln_rho_pen.shape}, expected ({n_pen_M},)"
        )

    if aggregator == "sum_squared":
        pen_label = "sum-squared"
    elif aggregator == "sum_pow":
        pp_label = {1.0: "linear", 2.0: "quadratic"}.get(
            float(penalty_power), f"power-{penalty_power}"
        )
        pen_label = f"sum-pow ({pp_label})"
    else:
        pen_label = aggregator
    print(
        f"M_enc monotonicity penalty: {n_pen_M} points in r ∈ "
        f"[{scfg.min_knot:.2f}, {scfg.max_knot:.2f}] kpc, "
        f"lam_mono_M={lam_mono_M:.3g} ({pen_label}, log-space)"
    )
    print(
        f"Outlier component: sig_outlier={sig_outlier} km/s, "
        f"Beta({f_outlier_alpha}, {f_outlier_beta}) prior"
    )

    data_j = {
        "x_f": jnp.asarray(x_f),
        "Cx_f": jnp.asarray(Cx_f),
        "logw": jnp.asarray(logw),
        # f_outlier is injected by fit_numpyro_nuts_mean_outlier
    }

    loglike_jax = make_loglike_3d_obs_jax_mean_mono_outlier_jeans(
        B_jax=B_j,
        B_pen_jax=jnp.asarray(B_pen),
        B_pen_der_jax=jnp.asarray(B_pen_der),
        r_pen_jax=jnp.asarray(r_pen),
        dln_rho_pen_jax=jnp.asarray(dln_rho_pen),
        nk=nk,
        lam_mono_M=float(lam_mono_M),
        penalty_power=float(penalty_power),
        aggregator=str(aggregator),
        sig_outlier=float(sig_outlier),
        M_floor=float(M_floor),
    )

    samples, diagnostics = fit_numpyro_nuts_mean_outlier(
        loglike_jax,
        nk=nk,
        data=data_j,
        vdisp_min=vdisp_min,
        vdisp_max=vdisp_max,
        f_outlier_alpha=f_outlier_alpha,
        f_outlier_beta=f_outlier_beta,
        target_accept_prob=target_accept_prob,
        **mcmc_kwargs,
    )

    sigma_samples = samples["params_sigma"]
    mean_samples = samples["params_mean"]
    f_outlier_samples = samples["f_outlier"]

    multi_disp = DispersionMeanMultiModel3D.from_sampler_3d(
        sigma_samples, mean_samples, scfg.knots_logr
    )
    return diagnostics, sigma_samples, mean_samples, f_outlier_samples, multi_disp
