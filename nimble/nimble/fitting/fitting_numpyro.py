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

from ..models import DispersionMultiModel3D

import jax.scipy.special as jsp_special


def build_agama_spline_basis_1d(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    Build B with shape (N, nk) such that for any p (nk,),
        agama.Spline(knots, p)(x) == B @ p   (up to roundoff)

    x: (N,) float64
    knots: (nk,) float64
    """
    nk = knots.size
    B = np.empty((x.size, nk), dtype=np.float64)
    for k in range(nk):
        ek = np.zeros(nk, dtype=np.float64)
        ek[k] = 1.0
        B[:, k] = agama.Spline(knots, ek)(x)
    print("Basis Built")
    return B


def fit_numpyro_nuts(
    loglike_fn,
    nk: int,
    data: dict,
    vdisp_min: float = 20.0,
    vdisp_max: float = 400.0,
    num_warmup: int = 800,
    num_samples: int = 1200,
    num_chains: int = 2,
    rng_seed: int = 0,
    target_accept_prob: float = 0.8,
    max_tree_depth: int | None = 7,
    start_x=None,
    progress_bar: bool = True,
):
    """
    Generic NUTS wrapper. loglike_fn must be JAX-traceable:
        loglike_fn(params_sigma, data) -> scalar

    start_x (optional):
      - None: default NumPyro init
      - (3*nk,): same init for all chains
      - (num_chains, 3*nk): per-chain init for chain_method="vectorized"
    """
    print("Fitting...")
    dt = time.time()

    log_min = float(np.log(vdisp_min))
    log_max = float(np.log(vdisp_max))
    dim = 3 * int(nk)

    # ---------- optional init params ----------
    init_params = None
    print("5 startup!")
    start_x = start_x if start_x is not None else np.full(dim, 5.0, dtype=float)
    x0 = jnp.asarray(start_x, dtype=jnp.float64)

    if x0.shape == (dim,):
        x0 = jnp.broadcast_to(x0, (int(num_chains), dim))
    elif x0.shape == (int(num_chains), dim):
        pass
    else:
        raise ValueError(
            f"start_x must have shape {(dim,)} or {(int(num_chains), dim)}, got {x0.shape}"
        )

    # clip to Uniform support to avoid invalid init
    eps = 1e-12
    x0 = jnp.clip(x0, log_min + eps, log_max - eps)
    init_params = {"params_sigma": x0}

    # ---------- model ----------
    def model():
        params_sigma = numpyro.sample(
            "params_sigma",
            dist.Uniform(low=log_min, high=log_max).expand([dim]),
        )
        ll = loglike_fn(params_sigma, data)
        numpyro.factor("loglike", ll)

    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )

    # On a single CPU device, vectorized chains is typically the right choice.
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
    mcmc.run(
        rng_key,
        init_params=init_params,
    )

    samples = mcmc.get_samples(group_by_chain=False)  # {"params_sigma": (draws, dim)}
    print("Fit complete!")
    print(f"Time taken: {(time.time() - dt) / 60:.3f} mins")
    return mcmc, samples


def make_loglike_3d_obs_jax(B_jax: jnp.ndarray, nk: int, jitter: float = 0.0):
    """
    All errors:
      - distance marginalisation over M samples
      - velocity measurement covariance Cx (non-diagonal)
      - intrinsic dispersion diag(sig_r^2, sig_t^2, sig_p^2)

    Requires data dict:
      x_f:   (NM,3)      flattened velocities for each (star, sample)
      Cx_f:  (NM,3,3)    flattened symmetric velocity covariances
      logw:  (N,M)       log distance weights
    """

    def loglike(params_sigma: jnp.ndarray, data: dict) -> jnp.ndarray:
        x_f = data["x_f"]  # (NM,3)
        Cx_f = data["Cx_f"]  # (NM,3,3)
        logw = data["logw"]  # (N,M)

        N, M = logw.shape  # static at trace time

        # split parameters (log sigma at knots)
        p_r = params_sigma[0:nk]
        p_t = params_sigma[nk : 2 * nk]
        p_p = params_sigma[2 * nk : 3 * nk]

        # spline-evaluated log(sig) at each sample via basis
        lsr = B_jax @ p_r  # (NM,)
        lst = B_jax @ p_t
        lsp = B_jax @ p_p

        # intrinsic variances
        sig_r2 = jnp.exp(lsr + lsr)
        sig_t2 = jnp.exp(lst + lst)
        sig_p2 = jnp.exp(lsp + lsp)

        # symmetric elements of Lambda = Cx + diag(sig^2)
        a = Cx_f[:, 0, 0] + sig_r2 + jitter
        d = Cx_f[:, 1, 1] + sig_t2 + jitter
        f = Cx_f[:, 2, 2] + sig_p2 + jitter

        b = Cx_f[:, 1, 0]
        c = Cx_f[:, 2, 0]
        e = Cx_f[:, 2, 1]

        x1 = x_f[:, 0]
        x2 = x_f[:, 1]
        x3 = x_f[:, 2]

        # explicit cholesky + forward solve
        l11, l21, l31, l22, l32, l33, ok = chol3x3_from_sym(a, b, c, d, e, f)
        y1, y2, y3 = forward_solve_lower3(l11, l21, l31, l22, l32, l33, x1, x2, x3)

        quad = y1 * y1 + y2 * y2 + y3 * y3
        logdet = 2.0 * (jnp.log(l11) + jnp.log(l22) + jnp.log(l33))

        ll_f = -0.5 * (quad + logdet)  # (NM,)
        ll = ll_f.reshape(N, M)  # (N,M)

        # distance marginalisation
        logp_star = jsp_special.logsumexp(logw + ll, axis=1)  # (N,)
        total = jnp.sum(logp_star)

        ok_all = jnp.all(ok) & jnp.isfinite(total)
        return jnp.where(ok_all, total, -jnp.inf)

    return loglike


def setup_and_fit_obs(
    sample_data, scfg, vdisp_min=20.0, vdisp_max=400.0, jitter=0.0, **mcmc_kwargs
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

    # optional: enforce symmetry once (cheap safety)
    Cx = 0.5 * (Cx + np.swapaxes(Cx, -1, -2))

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

    loglike_jax = make_loglike_3d_obs_jax(B_j, nk, jitter=float(jitter))

    mcmc, raw_samples = fit_numpyro_nuts(
        loglike_jax,
        nk=nk,
        data=data_j,
        vdisp_min=vdisp_min,
        vdisp_max=vdisp_max,
        **mcmc_kwargs,
    )
    samples = raw_samples["params_sigma"]
    multi_disp = DispersionMultiModel3D.from_sampler_3d(samples, scfg.knots_logr)
    return mcmc, samples, multi_disp


def chol3x3_from_sym(a, b, c, d, e, f):
    """
    Batched Cholesky for symmetric 3x3 SPD matrix with elements:
      [ a  b  c ]
      [ b  d  e ]
      [ c  e  f ]

    Inputs are arrays with the same leading shape (...,).
    Returns l11,l21,l31,l22,l32,l33 and ok mask.
    """
    ok1 = a > 0.0
    a_safe = jnp.where(ok1, a, 1.0)
    l11 = jnp.sqrt(a_safe)
    inv_l11 = 1.0 / l11

    l21 = b * inv_l11
    l31 = c * inv_l11

    t22 = d - l21 * l21
    ok2 = t22 > 0.0
    t22_safe = jnp.where(ok2, t22, 1.0)
    l22 = jnp.sqrt(t22_safe)
    inv_l22 = 1.0 / l22

    l32 = (e - l21 * l31) * inv_l22

    t33 = f - (l31 * l31 + l32 * l32)
    ok3 = t33 > 0.0
    t33_safe = jnp.where(ok3, t33, 1.0)
    l33 = jnp.sqrt(t33_safe)

    ok = ok1 & ok2 & ok3
    return l11, l21, l31, l22, l32, l33, ok


def forward_solve_lower3(l11, l21, l31, l22, l32, l33, x1, x2, x3):
    """
    Solve y = L^{-1} x for lower-triangular L with:
      [l11  0   0 ]
      [l21 l22  0 ]
      [l31 l32 l33]
    Batched over leading dims.
    """
    y1 = x1 / l11
    y2 = (x2 - l21 * y1) / l22
    y3 = (x3 - l31 * y1 - l32 * y2) / l33
    return y1, y2, y3
