import numpy as np
import agama
from ..config import SmallConfig


def _logsumexp(a, axis=-1):
    amax = np.max(a, axis=axis, keepdims=True)
    s = np.sum(np.exp(a - amax), axis=axis, keepdims=True)
    out = amax + np.log(s)
    return np.squeeze(out, axis=axis)


def loglike_3d_galframe(params_sigma, fit_data, scfg: SmallConfig):
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

    loglike = np.sum(ll_r + ll_theta + ll_phi)
    if not np.isfinite(loglike):
        loglike = -np.inf

    return loglike


def neg_loglike_3d_galframe(params_sigma, fit_data, scfg: SmallConfig):
    return -loglike_3d_galframe(params_sigma, fit_data, scfg)


def loglike_3d_vel_err(params_sigma, fit_data, scfg):
    logr = fit_data["logr"]  # (N,)
    x_obs = fit_data["vel_gal"]  # (N,3)
    C_x = fit_data["err_mat_gal"]  # (N,3,3)

    nk = scfg.num_knots
    knots = scfg.knots_logr

    # Lambdas
    p_r = params_sigma[:nk]
    l_sigma_r = agama.Spline(knots, p_r)(logr)
    sigma_r2 = np.exp(2.0 * l_sigma_r)
    p_t = params_sigma[nk : 2 * nk]
    l_sigma_t = agama.Spline(knots, p_t)(logr)
    sigma_t2 = np.exp(2.0 * l_sigma_t)
    p_p = params_sigma[2 * nk : 3 * nk]
    l_sigma_p = agama.Spline(knots, p_p)(logr)
    sigma_p2 = np.exp(2.0 * l_sigma_p)

    # With Error
    Lambda = C_x.copy()
    Lambda[:, 0, 0] += sigma_r2
    Lambda[:, 1, 1] += sigma_t2
    Lambda[:, 2, 2] += sigma_p2

    ### Fancy Way of finding  log (  multivariate)

    # --- Cholesky ---
    try:
        L = np.linalg.cholesky(Lambda)  # (N,3,3), lower
    except np.linalg.LinAlgError:
        return -np.inf

    try:
        z = np.linalg.solve(L, x_obs[..., None])[..., 0]  # (N,3,1) -> (N,3)
    except np.linalg.LinAlgError:
        return -np.inf

    quad = np.einsum("ni,ni->n", z, z)  # (N,) == sum(z*z, axis=1)

    # logdet(Lambda) = 2 * sum(log(diag(L)))
    diagL = np.diagonal(L, axis1=-2, axis2=-1)  # (N,3)
    if np.any(diagL <= 0):
        return -np.inf
    logdet = 2.0 * np.sum(np.log(diagL), axis=1)

    ll = -0.5 * (quad + logdet)
    loglike = float(np.sum(ll))
    return loglike if np.isfinite(loglike) else -np.inf


def neg_loglike_3d_vel_err(params_sigma, fit_data, scfg: SmallConfig):
    return -loglike_3d_vel_err(params_sigma, fit_data, scfg)


def loglike_3d_obs_fast(params_sigma, sample_data, scfg):
    # assume upstream: float64 + C-contig + log_dist_w provided
    logr = sample_data["logr"]  # (N,M)
    x = sample_data["vel_gal"]  # (N,M,3)
    Cx = sample_data["err_mat_gal"]  # (N,M,3,3) symmetric
    logw = sample_data["log_dist_w"]  # (N,M)

    N, M = logr.shape
    nk = scfg.num_knots
    NM = N * M

    logr_f = logr.reshape(NM)
    x_f = x.reshape(NM, 3)
    Cx_f = Cx.reshape(NM, 3, 3)

    # --- spline: log sigma_i(r) at each sample ---
    knots = scfg.knots_logr

    p_r = params_sigma[:nk]
    lsr = agama.Spline(knots, p_r)(logr_f)
    sig_r2 = np.exp(lsr + lsr)  # exp(2*lsr)

    p_t = params_sigma[nk : 2 * nk]
    lst = agama.Spline(knots, p_t)(logr_f)
    sig_t2 = np.exp(lst + lst)

    p_p = params_sigma[2 * nk : 3 * nk]
    lsp = agama.Spline(knots, p_p)(logr_f)
    sig_p2 = np.exp(lsp + lsp)

    # --- pull needed symmetric elements (views) and add dispersions to diagonal ---
    a = Cx_f[:, 0, 0] + sig_r2
    d = Cx_f[:, 1, 1] + sig_t2
    f = Cx_f[:, 2, 2] + sig_p2

    b = Cx_f[:, 1, 0]
    c = Cx_f[:, 2, 0]
    e = Cx_f[:, 2, 1]

    x1 = x_f[:, 0]
    x2 = x_f[:, 1]
    x3 = x_f[:, 2]

    # --- vectorized 3x3 Cholesky for SPD matrix ---
    # L =
    # [ l11  0    0 ]
    # [ l21 l22   0 ]
    # [ l31 l32  l33 ]
    # with:
    # l11 = sqrt(a)
    # l21 = b/l11
    # l31 = c/l11
    # l22 = sqrt(d - l21^2)
    # l32 = (e - l21*l31)/l22
    # l33 = sqrt(f - l31^2 - l32^2)

    # Cheap early fails (avoid doing work if proposal makes non-SPD)
    if np.min(a) <= 0.0:
        return -np.inf

    l11 = np.sqrt(a)
    inv_l11 = 1.0 / l11
    l21 = b * inv_l11
    l31 = c * inv_l11

    t22 = d - l21 * l21
    if np.min(t22) <= 0.0:
        return -np.inf
    l22 = np.sqrt(t22)
    inv_l22 = 1.0 / l22

    l32 = (e - l21 * l31) * inv_l22

    t33 = f - (l31 * l31 + l32 * l32)
    if np.min(t33) <= 0.0:
        return -np.inf
    l33 = np.sqrt(t33)

    # --- forward solve y = L^{-1} x (no need for second solve) ---
    y1 = x1 * inv_l11
    y2 = (x2 - l21 * y1) * inv_l22
    y3 = (x3 - l31 * y1 - l32 * y2) / l33

    quad = y1 * y1 + y2 * y2 + y3 * y3
    logdet = 2.0 * (np.log(l11) + np.log(l22) + np.log(l33))

    ll_f = -0.5 * (quad + logdet)  # (NM,)
    ll = ll_f.reshape(N, M)

    # distance marginalisation (uses your _logsumexp)
    logp_star = _logsumexp(logw + ll, axis=1)
    loglike = float(np.sum(logp_star))

    return loglike if np.isfinite(loglike) else -np.inf


def neg_loglike_3d_obs(params_sigma, fit_data, scfg: SmallConfig):
    return -loglike_3d_obs_fast(params_sigma, fit_data, scfg)
