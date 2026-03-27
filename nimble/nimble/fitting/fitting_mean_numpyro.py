import jax

jax.config.update("jax_enable_x64", True)

print("local_device_count:", jax.local_device_count())
print("devices:", jax.devices())
import numpy as np
import agama

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import time

import jax.numpy as jnp

from ..models import DispersionMeanMultiModel3D

import jax.scipy.special as jsp_special


def setup_and_fit_obs_mean(
    sample_data, scfg, vdisp_min=20.0, vdisp_max=400.0, **mcmc_kwargs
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

    loglike_jax = make_loglike_3d_obs_jax_mean(B_j, nk)

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

    # return mcmc, sigma_samples, mean_samples
    multi_disp = DispersionMeanMultiModel3D.from_sampler_3d(
        sigma_samples, mean_samples, scfg.knots_logr
    )
    return mcmc, sigma_samples, mean_samples, multi_disp


import jax

jax.config.update("jax_enable_x64", True)

print("local_device_count:", jax.local_device_count())
print("devices:", jax.devices())
import numpy as np
import agama

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import time

import jax.numpy as jnp

from nimble.models import DispersionMultiModel3D

import jax.scipy.special as jsp_special

from nimble.fitting.fitting_numpyro import (
    build_agama_spline_basis_1d,
    chol3x3_from_sym,
    forward_solve_lower3,
)


def fit_numpyro_nuts_mean(
    loglike_fn,
    nk: int,
    data: dict,
    vdisp_min: float = 20.0,
    vdisp_max: float = 400.0,
    vmean_max: float = 50.0,
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
    print("Fitting with mean...")
    dt = time.time()

    log_min = float(np.log(vdisp_min))
    log_max = float(np.log(vdisp_max))
    dim_sigma = 3 * int(nk)
    dim_mean = 3 * int(nk)
    dim = dim_sigma + dim_mean

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
                f" init must have shape {(dim,)} or {(int(num_chains), dim)}, got {x.shape}"
            )

        if (clip_low is not None) or (clip_high is not None):
            eps = 1e-12
            lo = -jnp.inf if clip_low is None else (clip_low + eps)
            hi = jnp.inf if clip_high is None else (clip_high - eps)
            x = jnp.clip(x, lo, hi)
        return x

    # ---------- physical-space init (same API as before) ----------
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

    x0 = jnp.concatenate([x0_sigma, x0_mean], axis=-1)
    init_params = {"params": x0}

    def model():
        dim_sigma = 3 * nk
        dim_mean = 3 * nk
        dim = dim_sigma + dim_mean

        low = jnp.concatenate(
            [
                jnp.full((dim_sigma,), log_min, dtype=jnp.float64),
                jnp.full((dim_mean,), -vmean_max, dtype=jnp.float64),
            ]
        )
        high = jnp.concatenate(
            [
                jnp.full((dim_sigma,), log_max, dtype=jnp.float64),
                jnp.full((dim_mean,), vmean_max, dtype=jnp.float64),
            ]
        )

        params = numpyro.sample(
            "params",
            dist.Uniform(low=low, high=high),
        )  # shape (5*nk,)

        ll = loglike_fn(params, data)
        numpyro.factor("loglike", ll)

    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )

    print(f"num_chains:{num_chains}")
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

    posterior = mcmc.get_samples(group_by_chain=False)  # {"params": (draws, 5*nk)}
    params = posterior["params"]  # already physical
    R = params.reshape((-1, 6, nk))

    params_sigma = R[:, 0:3, :].reshape((-1, 3 * nk))  # already physical log-sigmas
    params_mean = R[:, 3:6, :].reshape((-1, 3 * nk))  # already physical means

    samples = {
        "params": params,  # physical params, (draws, 5*nk)
        "params_sigma": params_sigma,  # (draws, 3*nk)
        "params_mean": params_mean,  # (draws, 3*nk)
        # optionally keep alias for backwards compatibility:
        # "params_raw": params,
    }

    print("Fit complete!")
    print(f"Time taken: {(time.time() - dt) / 60:.3f} mins")
    return mcmc, samples


def make_loglike_3d_obs_jax_mean(B_jax: jnp.ndarray, nk: int):
    """
    All errors:
      - distance marginalisation over M samples
      - velocity measurement covariance Cx (non-diagonal)
      - intrinsic dispersion diag(sig_r^2, sig_t^2, sig_p^2)
      - streaming means mu = (0, mu_theta(r), mu_phi(r)) from splines

    loglike params:
      params: (5*nk,) in physical units, layout
              [logsig_r, logsig_t, logsig_p, mean_t, mean_p]
    """

    def loglike(params: jnp.ndarray, data: dict) -> jnp.ndarray:
        x_f = data["x_f"]  # (NM,3)
        Cx_f = data["Cx_f"]  # (NM,3,3)
        logw = data["logw"]  # (N,M)

        N, M = logw.shape  # static at trace time

        # Reshape into 5 spline coefficient blocks (each length nk)
        P = params.reshape(6, nk)  # rows: [lsig_r, lsig_t, lsig_p, mean_t, mean_p]

        # ONE matmul instead of 5 separate (B @ p_i)
        vals = B_jax @ P.T  # (NM, 5)

        lsr = vals[:, 0]
        lst = vals[:, 1]
        lsp = vals[:, 2]
        mu_r = vals[:, 3]
        mu_t = vals[:, 4]
        mu_p = vals[:, 5]

        # intrinsic variances
        sig_r2 = jnp.exp(2 * lsr)
        sig_t2 = jnp.exp(2 * lst)
        sig_p2 = jnp.exp(2 * lsp)

        # symmetric elements of Lambda = Cx + diag(sig^2)
        a = Cx_f[:, 0, 0] + sig_r2
        d = Cx_f[:, 1, 1] + sig_t2
        f = Cx_f[:, 2, 2] + sig_p2

        b = Cx_f[:, 1, 0]
        c = Cx_f[:, 2, 0]
        e = Cx_f[:, 2, 1]

        # center data by streaming mean (mu_r fixed to 0)
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

        ok_all = jnp.all(ok) & jnp.isfinite(total)
        return jnp.where(ok_all, total, -jnp.inf)

    return loglike
