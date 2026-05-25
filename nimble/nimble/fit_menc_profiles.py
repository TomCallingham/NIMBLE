import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.special import hyp2f1
from scipy.optimize import brentq, least_squares


rho_crit0 = 127.54  # Msun / kpc^3   (h = 0.6777, z = 0)


def M_enc_abg(r, rho0, rs, alpha, beta, gamma):
    """Enclosed mass of an αβγ double power-law density profile."""
    x = np.asarray(r) / rs
    a = (3 - gamma) / alpha
    b = (beta - gamma) / alpha
    return (
        4
        * np.pi
        * rho0
        * rs**3
        * x ** (3 - gamma)
        / (3 - gamma)
        * hyp2f1(a, b, a + 1, -(x**alpha))
    )


def _interp_baryon(r, r_baryon, M_enc_baryon):
    """Interpolate baryon enclosed mass; flat-extrapolate beyond the baryon grid.

    Below the innermost baryon radius we use the first value (flat); above the
    outermost radius we use the last value (total baryon mass is constant beyond
    the edge of the provided profile).
    """
    return np.interp(
        r, r_baryon, M_enc_baryon, left=M_enc_baryon[0], right=M_enc_baryon[-1]
    )


def fit_abg_profile(
    r_space,
    M_enc,
    M_enc_o1s,
    M_enc_u1s,
    r_baryon=None,
    M_enc_baryon=None,
    *,
    alpha_fix=1.0,
    beta_bounds=(2.1, 6.0),
    gamma_bounds=(0.0, 1.9),
    log_rho0_bounds=(-8.0, 13.0),
    alpha_bounds=(0.3, 3.0),
    rho_crit=rho_crit0,
    overdensity=200.0,
    nsteps=6000,
    nwalkers_per_dim=8,
    seed=None,
    progress=True,
    plot=False,
    n_post_draws=500,
):
    """Fit an αβγ density profile to an enclosed-mass curve and propagate to (R200, M200, c200).

    If ``r_baryon`` and ``M_enc_baryon`` are supplied the baryon contribution is
    treated as exact: it is subtracted from the data before fitting (so the αβγ
    model describes the dark-matter halo only), and then added back when solving
    for R200/M200 so that those quantities refer to the *total* (DM + baryon)
    enclosed mass at the overdensity radius.  Beyond the outer edge of the
    baryon profile the baryon mass is held constant at its last tabulated value.

    Priors are uniform within the given bounds (rho0 and rs in log10 space).
    log_rs prior bounds are taken from the radial range of the data:
    [log10(r.min()), log10(r.max())].

    Returns
    -------
    res_chain : dict of 1-D ndarrays
        Posterior samples (post burn-in, thinned, with non-converged R200 dropped).
    res_stats_dic : dict of dicts
        For each key, {'median', 'minus', 'plus'} from 16/50/84 percentiles.
    """

    rng = np.random.default_rng(seed)
    r = np.asarray(r_space, float)
    M_med = np.asarray(M_enc, float)
    M_up = np.asarray(M_enc_o1s, float)
    M_lo = np.asarray(M_enc_u1s, float)

    # ---- baryon subtraction ------------------------------------------------
    # The baryon profile is treated as exact, so its contribution is removed
    # from the median *and* from the 1σ envelopes (which shift rigidly with
    # the median, preserving the DM uncertainty).
    have_baryons = (r_baryon is not None) and (M_enc_baryon is not None)
    if have_baryons:
        r_baryon = np.asarray(r_baryon, float)
        M_enc_baryon = np.asarray(M_enc_baryon, float)
        M_bar_at_r = _interp_baryon(r, r_baryon, M_enc_baryon)
        M_med = M_med - M_bar_at_r
        M_up = M_up - M_bar_at_r
        M_lo = M_lo - M_bar_at_r
        # Guard against negative DM mass at small radii where baryons may
        # dominate.  Clamp to a small positive floor so the sampler stays valid,
        # but warn the user.
        floor = 1e-30
        if np.any(M_med <= 0):
            import warnings

            warnings.warn(
                "Baryon subtraction produced non-positive DM M_enc at some radii; "
                "clamping to a small floor.  Consider restricting r_space to radii "
                "where the DM contribution is non-negligible.",
                RuntimeWarning,
                stacklevel=2,
            )
            M_med = np.maximum(M_med, floor)
            M_up = np.maximum(M_up, floor)
            M_lo = np.maximum(M_lo, floor)
    # ------------------------------------------------------------------------

    # ---- prior bounds: [log10(rho0), log10(rs), beta, gamma] (+ alpha if free) ----
    LB = np.array(
        [log_rho0_bounds[0], np.log10(r.min()), beta_bounds[0], gamma_bounds[0]]
    )
    UB = np.array(
        [log_rho0_bounds[1], np.log10(r.max()), beta_bounds[1], gamma_bounds[1]]
    )
    if alpha_fix is None:
        LB = np.append(LB, alpha_bounds[0])
        UB = np.append(UB, alpha_bounds[1])

    def unpack(theta):
        if alpha_fix is None:
            log_rho0, log_rs, beta, gamma, alpha = theta
        else:
            log_rho0, log_rs, beta, gamma = theta
            alpha = alpha_fix
        return 10**log_rho0, 10**log_rs, alpha, beta, gamma

    def log_post(theta):
        if np.any(theta < LB) or np.any(theta > UB):
            return -np.inf
        rho0, rs, alpha, beta, gamma = unpack(theta)
        if beta <= gamma + 1e-3:
            return -np.inf
        try:
            Mm = M_enc_abg(r, rho0, rs, alpha, beta, gamma)
        except (FloatingPointError, ValueError):
            return -np.inf
        if not np.all(np.isfinite(Mm)) or np.any(Mm <= 0):
            return -np.inf
        sig_hi = M_up - M_med
        sig_lo = M_med - M_lo
        sig = np.where(Mm > M_med, sig_hi, sig_lo)
        sig = np.maximum(sig, 1e-30)
        norm = np.log(2.0 / (sig_lo + sig_hi)) - 0.5 * np.log(2 * np.pi)
        return np.sum(norm - 0.5 * ((Mm - M_med) / sig) ** 2)

    # ---- LSQ initial point ----
    def residuals(theta):
        rho0, rs, alpha, beta, gamma = unpack(theta)
        Mm = M_enc_abg(r, rho0, rs, alpha, beta, gamma)
        sig = np.where(Mm > M_med, M_up - M_med, M_med - M_lo)
        return (Mm - M_med) / np.maximum(sig, 1e-30)

    x0 = [
        np.log10(M_med[-1] / (4 * np.pi * r[-1] ** 3)),
        np.log10(np.median(r)),
        3.0,
        1.0,
    ]
    if alpha_fix is None:
        x0.append(1.0)
    lsq = least_squares(residuals, x0, bounds=(LB.tolist(), UB.tolist()))

    # ---- MCMC ----
    ndim = len(lsq.x)
    nwalkers = nwalkers_per_dim * ndim
    p0 = lsq.x + 1e-3 * rng.standard_normal((nwalkers, ndim)) * np.maximum(
        np.abs(lsq.x), 1e-2
    )
    p0 = np.clip(p0, LB + 1e-6, UB - 1e-6)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post)
    sampler.run_mcmc(p0, nsteps, progress=progress)

    tau = sampler.get_autocorr_time(quiet=True)
    chain = sampler.get_chain(
        discard=int(3 * tau.max()), thin=max(1, int(tau.max() / 2)), flat=True
    )

    # ---- propagate to R200, M200, c200 ------------------------------------
    # R200 is defined by the *total* (DM + baryon) enclosed mass.  When baryons
    # are provided we add back their (flat-extrapolated) profile when evaluating
    # the overdensity condition.
    target = (4 / 3) * np.pi * overdensity * rho_crit

    def M_enc_total(R, rho0, rs, alpha, beta, gamma):
        M_dm = M_enc_abg(R, rho0, rs, alpha, beta, gamma)
        if have_baryons:
            M_dm += _interp_baryon(R, r_baryon, M_enc_baryon)
        return M_dm

    def solve_R200(rho0, rs, alpha, beta, gamma):
        f = lambda R: M_enc_total(R, rho0, rs, alpha, beta, gamma) - target * R**3
        lo, hi = 1.0, 1e3
        while f(lo) > 0 and lo > 1e-4:
            lo /= 10
        while f(hi) < 0 and hi < 1e7:
            hi *= 10
        if f(lo) * f(hi) > 0:
            return np.nan
        return brentq(f, lo, hi)

    R200 = np.array([solve_R200(*unpack(th)) for th in chain])
    ok = np.isfinite(R200)
    chain, R200 = chain[ok], R200[ok]

    rho0_s = 10 ** chain[:, 0]
    rs_s = 10 ** chain[:, 1]
    beta_s = chain[:, 2]
    gamma_s = chain[:, 3]
    alpha_s = chain[:, 4] if alpha_fix is None else np.full(len(chain), alpha_fix)

    # M200 is total mass inside R200
    M200 = np.array([M_enc_total(R, *unpack(th)) for R, th in zip(R200, chain)])
    c200 = R200 / rs_s

    res_chain = {
        "log_rho0": chain[:, 0],
        "log_rs": chain[:, 1],
        "rho0": rho0_s,
        "rs": rs_s,
        "alpha": alpha_s,
        "beta": beta_s,
        "gamma": gamma_s,
        "R200": R200,
        "M200": M200,
        "c200": c200,
    }

    res_stats_dic = {}
    for k, v in res_chain.items():
        p16, p50, p84 = np.percentile(v, [16, 50, 84])
        res_stats_dic[k] = {"median": p50, "minus": p50 - p16, "plus": p84 - p50}

    # ---- plot ----
    if plot:
        r_grid = np.geomspace(r.min(), r.max(), 200)

        n_draws = min(n_post_draws, len(chain))
        idx = rng.choice(len(chain), size=n_draws, replace=False)
        Mm_draws = np.empty((n_draws, r_grid.size))
        for i, j in enumerate(idx):
            Mm_draws[i] = M_enc_abg(r_grid, *unpack(chain[j]))

        Mm_p16, Mm_p50, Mm_p84 = np.percentile(Mm_draws, [16, 50, 84], axis=0)

        fig, ax = plt.subplots(figsize=(7, 5))
        # data (DM only after subtraction)
        yerr = np.vstack([M_med - M_lo, M_up - M_med])
        label = "data (DM only)" if have_baryons else "data"
        ax.errorbar(
            r,
            M_med,
            yerr=yerr,
            fmt="o",
            color="k",
            ms=4,
            capsize=2,
            label=label,
            zorder=3,
        )
        ax.fill_between(
            r_grid,
            Mm_p16,
            Mm_p84,
            alpha=0.3,
            color="C0",
            label=r"posterior 1$\sigma$ (DM)",
        )
        ax.plot(r_grid, Mm_p50, color="C0", lw=1.5, label="posterior median (DM)")

        if have_baryons:
            M_bar_grid = _interp_baryon(r_grid, r_baryon, M_enc_baryon)
            ax.plot(
                r_grid, M_bar_grid, color="C1", lw=1.5, ls="--", label="baryons (input)"
            )
            ax.plot(
                r_grid,
                Mm_p50 + M_bar_grid,
                color="C2",
                lw=1.5,
                label="posterior median (total)",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("r [kpc]")
        ax.set_ylabel(r"$M_{\rm enc}(r)$ [M$_\odot$]")
        ax.legend(frameon=False)
        fig.tight_layout()
        plt.show()

    return res_chain, res_stats_dic


def _nfw_mu(x):
    """μ(x) = ln(1+x) - x/(1+x); appears in NFW enclosed mass."""
    return np.log(1.0 + x) - x / (1.0 + x)


def M_enc_nfw(r, M200, c200, *, rho_crit=rho_crit0, overdensity=200.0):
    """NFW enclosed mass, parameterized by (M200, c200).

    R200 is set by the overdensity condition; rho_s by the c-M relation.
    """
    R200 = (3.0 * M200 / (4.0 * np.pi * overdensity * rho_crit)) ** (1.0 / 3.0)
    rs = R200 / c200
    x = np.asarray(r) / rs
    return M200 * _nfw_mu(x) / _nfw_mu(c200)


def fit_nfw_profile(
    r_space,
    M_enc,
    M_enc_o1s,
    M_enc_u1s,
    r_baryon=None,
    M_enc_baryon=None,
    *,
    M200_bounds=(1e10, 1e13),
    c200_bounds=(1.0, 30.0),
    rho_crit=rho_crit0,
    overdensity=200.0,
    nsteps=6000,
    nwalkers_per_dim=8,
    seed=None,
    progress=True,
):
    """Fit an NFW profile to an enclosed-mass curve in (M200, c200).

    If ``r_baryon`` and ``M_enc_baryon`` are supplied the baryon contribution
    is subtracted before fitting (NFW describes DM only) and added back when
    computing R200/M200.

    Parameters
    ----------
    r_space : array_like
        Radii in kpc.
    M_enc, M_enc_o1s, M_enc_u1s : array_like
        Median enclosed mass and its upper / lower 1σ envelopes, in Msun.
    r_baryon, M_enc_baryon : array_like, optional
        Exact baryon enclosed-mass profile.  Flat-extrapolated beyond its range.
    M200_bounds, c200_bounds : (lo, hi)
        Linear-uniform prior bounds.
    rho_crit, overdensity, nsteps, nwalkers_per_dim, seed, progress
        Same meaning as in fit_abg_profile.

    Returns
    -------
    res_chain : dict of 1-D ndarrays
        Posterior samples (post-burn-in, thinned). Keys: 'M200', 'c200',
        'R200', 'rs', 'rho_s'.
    res_stats_dic : dict of dicts
        For each key, {'median', 'minus', 'plus'} from 16/50/84 percentiles.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(r_space, float)
    M_med = np.asarray(M_enc, float)
    M_up = np.asarray(M_enc_o1s, float)
    M_lo = np.asarray(M_enc_u1s, float)

    # ---- baryon subtraction ------------------------------------------------
    have_baryons = (r_baryon is not None) and (M_enc_baryon is not None)
    if have_baryons:
        r_baryon = np.asarray(r_baryon, float)
        M_enc_baryon = np.asarray(M_enc_baryon, float)
        M_bar_at_r = _interp_baryon(r, r_baryon, M_enc_baryon)
        M_med = M_med - M_bar_at_r
        M_up = M_up - M_bar_at_r
        M_lo = M_lo - M_bar_at_r
        floor = 1e-30
        if np.any(M_med <= 0):
            import warnings

            warnings.warn(
                "Baryon subtraction produced non-positive DM M_enc at some radii; "
                "clamping to a small floor.",
                RuntimeWarning,
                stacklevel=2,
            )
            M_med = np.maximum(M_med, floor)
            M_up = np.maximum(M_up, floor)
            M_lo = np.maximum(M_lo, floor)
    # ------------------------------------------------------------------------

    LB = np.array([M200_bounds[0], c200_bounds[0]])
    UB = np.array([M200_bounds[1], c200_bounds[1]])

    def _Mm_dm(theta):
        M200, c200 = theta
        return M_enc_nfw(r, M200, c200, rho_crit=rho_crit, overdensity=overdensity)

    def log_post(theta):
        if np.any(theta < LB) or np.any(theta > UB):
            return -np.inf
        try:
            Mm = _Mm_dm(theta)
        except (FloatingPointError, ValueError):
            return -np.inf
        if not np.all(np.isfinite(Mm)) or np.any(Mm <= 0):
            return -np.inf
        sig_hi = M_up - M_med
        sig_lo = M_med - M_lo
        sig = np.where(Mm > M_med, sig_hi, sig_lo)
        sig = np.maximum(sig, 1e-30)
        norm = np.log(2.0 / (sig_lo + sig_hi)) - 0.5 * np.log(2 * np.pi)
        return np.sum(norm - 0.5 * ((Mm - M_med) / sig) ** 2)

    # ---- LSQ initial point ----
    def residuals(theta):
        Mm = _Mm_dm(theta)
        sig = np.where(Mm > M_med, M_up - M_med, M_med - M_lo)
        return (Mm - M_med) / np.maximum(sig, 1e-30)

    x0 = [np.clip(M_med[-1], *M200_bounds), 10.0]
    lsq = least_squares(residuals, x0, bounds=(LB.tolist(), UB.tolist()))

    # ---- MCMC ----
    ndim = 2
    nwalkers = nwalkers_per_dim * ndim
    jitter = np.array([1e-3 * lsq.x[0], 1e-3 * max(abs(lsq.x[1]), 1e-2)])
    p0 = lsq.x + rng.standard_normal((nwalkers, ndim)) * jitter
    p0 = np.clip(p0, LB + 1e-6 * (UB - LB), UB - 1e-6 * (UB - LB))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post)
    sampler.run_mcmc(p0, nsteps, progress=progress)

    tau = sampler.get_autocorr_time(quiet=True)
    chain = sampler.get_chain(
        discard=int(3 * tau.max()), thin=max(1, int(tau.max() / 2)), flat=True
    )

    # ---- propagate to R200, M200, c200 ------------------------------------
    # NFW M200 is the DM-only mass enclosed at R200; R200 itself is defined by
    # the *total* mass.  We solve for R200 numerically to be consistent with
    # fit_abg_profile.
    target = (4 / 3) * np.pi * overdensity * rho_crit

    def M_enc_nfw_total(R, M200, c200):
        M_dm = M_enc_nfw(R, M200, c200, rho_crit=rho_crit, overdensity=overdensity)
        if have_baryons:
            M_dm += _interp_baryon(R, r_baryon, M_enc_baryon)
        return M_dm

    def solve_R200(M200, c200):
        f = lambda R: M_enc_nfw_total(R, M200, c200) - target * R**3
        lo, hi = 1.0, 1e3
        while f(lo) > 0 and lo > 1e-4:
            lo /= 10
        while f(hi) < 0 and hi < 1e7:
            hi *= 10
        if f(lo) * f(hi) > 0:
            return np.nan
        return brentq(f, lo, hi)

    M200_s = chain[:, 0]
    c200_s = chain[:, 1]

    if have_baryons:
        R200 = np.array([solve_R200(M200, c200) for M200, c200 in zip(M200_s, c200_s)])
        ok = np.isfinite(R200)
        chain, M200_s, c200_s, R200 = chain[ok], M200_s[ok], c200_s[ok], R200[ok]
        M200_total = np.array(
            [M_enc_nfw_total(R, M, c) for R, M, c in zip(R200, M200_s, c200_s)]
        )
    else:
        # No baryons: closed-form R200 from the NFW definition
        R200 = (3.0 * M200_s / (4.0 * np.pi * overdensity * rho_crit)) ** (1.0 / 3.0)
        M200_total = M200_s

    rs = R200 / c200_s
    rho_s = M200_s / (4.0 * np.pi * rs**3 * _nfw_mu(c200_s))

    res_chain = {
        "M200": M200_total,  # total mass at R200
        "M200_dm": M200_s,  # DM-only NFW mass (the fitted parameter)
        "c200": c200_s,
        "R200": R200,
        "rs": rs,
        "rho_s": rho_s,
    }

    res_stats_dic = {}
    for k, v in res_chain.items():
        p16, p50, p84 = np.percentile(v, [16, 50, 84])
        res_stats_dic[k] = {"median": p50, "minus": p50 - p16, "plus": p84 - p50}

    return res_chain, res_stats_dic
