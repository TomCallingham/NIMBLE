import numpy as np
import matplotlib.pyplot as plt
import corner

# GLOBAL STATE
from .config import Config

from .modelling import modelDensity, modelSigma
from .analytic_density_profile import medina24rrl_rho

d2r = np.pi / 180  # conversion from degrees to radians


def plot_profiles(chain, true_data, cfg: Config, plotname=""):
    scfg = cfg.to_small()
    # TODO:

    # From data
    Gapp = np.array([])
    logr_obs = np.array([])
    vr_obs = np.array([])
    vt_obs = np.array([])

    # From sim
    true_dens_radii = true_data["true_dens_radii"]
    true_sigmar = true_data["true_sigmar"]
    true_sigmat = true_data["true_sigmat"]

    # Density plots
    fig = plt.figure(figsize=(7, 7))
    axs = fig.subplots(
        nrows=2, ncols=1, sharex=True, gridspec_kw=dict(hspace=0, height_ratios=[3, 1])
    )

    r = np.logspace(np.log10(cfg.min_r), np.log10(cfg.max_r), 200)
    lr = np.log(r)
    rhist = np.logspace(np.log10(cfg.min_r), np.log10(cfg.max_r), 81)
    lrhist = np.log(rhist)
    if cfg.true_path is not None:
        num_true_knots = 12
        if cfg.run_type == "auridesi":
            num_true_knots = 4
        knots_logr_true = np.linspace(np.log(1), np.log(200), num_true_knots)
        S = agama.splineLogDensity(
            knots_logr_true,
            x=np.log(true_dens_radii),
            w=np.ones(len(true_dens_radii)),
            infLeft=True,
            infRight=True,
        )
        trueparams_dens = np.log(
            (np.exp(S(knots_logr_true)))
            / (4.0 * np.pi * (np.exp(knots_logr_true) ** 3))
        )
        # set the first element of array to zero and exclude it
        trueparams_dens = trueparams_dens[1:] - trueparams_dens[0]
        truedens = np.exp(modelDensity(trueparams_dens, scfg, truedens=True)(lr))
    else:
        truedens = None

    # main plot
    if (plotname == "converged") or cfg.verbose:
        if cfg.true_path is not None:
            axs[0].plot(r, truedens, "k-", label="True")
            axs[0].set_ylim(min(truedens) * 0.2, max(truedens) * 2)
        if cfg.use_external_density:
            axs[0].plot(r, medina24rrl_rho(r))
        # retrieve density profiles of each model in the chain, and compute median and 16/84 percentiles
        results = np.zeros((len(chain), len(r)))
        for i in range(len(chain)):
            results[i] = np.exp(modelDensity(chain[i, 0 : cfg.num_knots - 1], scfg)(lr))
        dens_low, dens_med, dens_upp = np.percentile(results, [16, 50, 84], axis=0)
        # plot the model profiles with 1sigma confidence intervals
        axs[0].fill_between(r, dens_low, dens_upp, alpha=0.3, lw=0, color="r")
        axs[0].plot(r, dens_med, color="r", label="MCMC Fit")
        count_obs = np.histogram(logr_obs, bins=lrhist)[0]
        rho_obs = count_obs / (
            4
            * np.pi
            * (lrhist[1:] - lrhist[:-1])
            * (rhist[1:] * rhist[:-1]) ** 1.5
            * len(Gapp)
        )
        axs[0].plot(
            np.repeat(rhist, 2)[1:-1],
            np.repeat(rho_obs, 2),
            "r--",
            label="with SFs and Errors",
        )

        axs[0].set_title(cfg.figs_path)
        axs[0].set_ylabel("3d density of tracers")
        axs[0].set_yscale("log")
        axs[0].set_xlim(cfg.min_r, cfg.max_r)
        axs[0].legend(loc="upper right", frameon=False)

        # percent error
        if cfg.true_path is not None:
            percerr = 100 * ((dens_med - truedens) / truedens)
            lowerr = 100 * ((dens_low - truedens) / truedens)
            upperr = 100 * ((dens_upp - truedens) / truedens)
            axs[1].plot(r, percerr, "r")
            axs[1].fill_between(r, lowerr, upperr, alpha=0.3, lw=0, color="r")
        axs[1].axhline(0, c="gray", linestyle="dashed")
        axs[1].set_ylim(-40, 40)

        axs[1].set_xlabel("Galactocentric radius (kpc)")
        axs[1].set_ylabel("percent error")
        axs[1].set_xscale("log")

        plt.tight_layout()
        plt.savefig(f"{cfg.figs_path}{plotname}_dens.pdf", dpi=200, bbox_inches="tight")
        plt.cla()

        # Sigma plots
        fig = plt.figure(figsize=(12, 7))
        axs = fig.subplots(
            nrows=2,
            ncols=2,
            sharex=True,
            gridspec_kw=dict(
                hspace=0, wspace=0, height_ratios=[3, 1], width_ratios=[1, 1]
            ),
        )
        axs[0, 0].text(x=45, y=470, s="Radial", size=18)
        axs[0, 1].text(x=45, y=470, s="Tangential", size=18)

        # collect the model profiles and plot median and 16/84 percentile confidence intervals
        results_r, results_t = np.zeros((2, len(chain), len(r)))
        for i in range(len(chain)):
            results_r[i] = np.exp(
                modelSigma(chain[i, cfg.num_knots - 1 : 2 * cfg.num_knots - 1], scfg)(
                    lr
                )
            )
            results_t[i] = np.exp(
                modelSigma(chain[i, 2 * cfg.num_knots - 1 :], scfg)(lr)
            )
        low_r, med_r, upp_r = np.percentile(results_r, [16, 50, 84], axis=0)
        axs[0, 0].fill_between(r, low_r, upp_r, alpha=0.3, lw=0, color="g")
        axs[0, 0].plot(r, med_r, color="g", label="MCMC Fit $\sigma_\mathrm{rad}$")
        low_t, med_t, upp_t = np.percentile(results_t, [16, 50, 84], axis=0)
        axs[0, 1].fill_between(r, low_t, upp_t, alpha=0.3, lw=0, color="b")
        axs[0, 1].plot(r, med_t, color="b", label="MCMC Fit $\sigma_\mathrm{tan}$")

        if cfg.true_path is not None:
            truesigr = true_sigmar(lr) ** 0.5
            truesigt = true_sigmat(lr) ** 0.5
            axs[0, 0].plot(r, truesigr, "k-", label="True $\sigma_\mathrm{rad}$")
            axs[0, 1].plot(r, truesigt, "k-", label="True $\sigma_\mathrm{tan}$")

            percerr_r = 100 * ((med_r - truesigr) / truesigr)
            lowerr_r = 100 * ((low_r - truesigr) / truesigr)
            upperr_r = 100 * ((upp_r - truesigr) / truesigr)
            axs[1, 0].plot(r, percerr_r, c="g")
            axs[1, 0].axhline(0, c="gray", linestyle="dashed")
            axs[1, 0].fill_between(r, lowerr_r, upperr_r, alpha=0.3, lw=0, color="g")
            percerr_t = 100 * ((med_t - truesigt) / truesigt)
            lowerr_t = 100 * ((low_t - truesigt) / truesigt)
            upperr_t = 100 * ((upp_t - truesigt) / truesigt)
            axs[1, 1].plot(r, percerr_t, c="b")
            axs[1, 1].axhline(0, c="gray", linestyle="dashed")
            axs[1, 1].fill_between(r, lowerr_t, upperr_t, alpha=0.3, lw=0, color="b")

        # plot the observed radial/tangential dispersions, which are affected by distance errors
        # and broadened by PM errors (especially the tangential dispersion)
        sigmar_obs = (
            np.histogram(logr_obs, bins=lrhist, weights=vr_obs**2)[0] / count_obs
        ) ** 0.5
        sigmat_obs = (
            np.histogram(logr_obs, bins=lrhist, weights=vt_obs**2)[0] / count_obs
        ) ** 0.5
        axs[0, 0].plot(
            np.repeat(rhist, 2)[1:-1],
            np.repeat(sigmar_obs, 2),
            "g--",
            label="with SFs and Errors $\sigma_\mathrm{rad}$",
            alpha=0.3,
        )
        axs[0, 1].plot(
            np.repeat(rhist, 2)[1:-1],
            np.repeat(sigmat_obs, 2),
            "b--",
            label="with SFs and Errors $\sigma_\mathrm{tan}$",
            alpha=0.3,
        )

        # upper_bound = max(np.hstack((upp_r, upp_t, sigmar_obs,
        #                              sigmat_obs, truesigr, truesigt)))*1.1
        axs[0, 0].set_ylim(-20, 500)
        axs[0, 1].set_ylim(-20, 500)
        axs[0, 1].set_yticklabels([])
        axs[1, 0].set_ylim(-40, 40)
        axs[1, 1].set_ylim(-40, 40)
        axs[1, 1].set_yticklabels([])

        axs[0, 0].set_xlim(cfg.min_r, cfg.max_r * 1.1)
        axs[0, 0].set_title(cfg.figs_path)
        axs[1, 0].set_xlabel("Galactocentric radius (kpc)")
        axs[1, 1].set_xlabel("Galactocentric radius (kpc)")
        axs[0, 0].set_ylabel("velocity dispersion of tracers")
        axs[1, 0].set_ylabel("percent error (%)")
        axs[0, 0].legend(loc="upper left", frameon=False)
        axs[0, 1].legend(loc="upper left", frameon=False)

        plt.tight_layout()
        plt.savefig(
            cfg.figs_path + plotname + "_sigs.pdf", dpi=200, bbox_inches="tight"
        )
        plt.cla()

    if plotname == "converged":
        # True and MCMC velocity dispersion and density profiles
        if cfg.true_path is not None:
            profiles = np.stack(
                [
                    r,
                    dens_low,
                    dens_med,
                    dens_upp,
                    truedens,
                    low_r,
                    med_r,
                    upp_r,
                    truesigr,
                    low_t,
                    med_t,
                    upp_t,
                    truesigt,
                ],
                axis=1,
            )
            cfg.write_csv(
                profiles,
                "veldisp_dens_profiles.csv",
                "r, dens_low, dens_med, dens_upp, truedens, low_r, med_r, upp_r, truesigr, low_t, med_t, upp_t, truesigt",
            )
        else:
            profiles = np.stack(
                [
                    r,
                    dens_low,
                    dens_med,
                    dens_upp,
                    low_r,
                    med_r,
                    upp_r,
                    low_t,
                    med_t,
                    upp_t,
                ],
                axis=1,
            )
            cfg.write_csv(
                profiles,
                "veldisp_dens_profiles.csv",
                "r, dens_low, dens_med, dens_upp, low_r, med_r, upp_r, low_t, med_t, upp_t",
            )

        # histograms of velocity dispersion and density after obs effects
        histograms = np.stack(
            [
                np.repeat(rhist, 2)[1:-1],
                np.repeat(rho_obs, 2),
                np.repeat(sigmar_obs, 2),
                np.repeat(sigmat_obs, 2),
            ],
            axis=1,
        )
        cfg.write_csv(
            histograms, "veldisp_dens_hists.csv", "rgrid, dens, sigmar, sigmat"
        )


def plot_mass_enclosed(Menc_med, Menc_low, Menc_upp, cfg: Config):
    # Mass enclosed plot
    plt.rcParams.update({"font.size": 18})
    plt.rcParams["agg.path.chunksize"] = (
        10000  # overflow error on line 835 without this
    )
    fig = plt.figure(figsize=(15, 10))
    axs = fig.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        gridspec_kw=dict(hspace=0, height_ratios=[3, 1]),
    )

    description = cfg.run_type + " ".join(cfg.load_params)

    if "m12f" in description:
        color = "#4a0078"
    elif "m12i" in description:
        color = "#157F1F"
    elif "m12m" in description:
        color = "#931621"
    elif "06" in description:
        color = "g"
    elif "21" in description:
        color = "r"
    elif "24" in description:
        color = "b"
    elif "iron" in description:
        color = "mediumblue"
    else:
        color = "gold"

    axs[0].plot(r, Menc_med, c=color, linewidth=2.5, label="Jeans estimate")
    axs[0].fill_between(
        r,
        Menc_low,
        Menc_upp,
        color=color,
        alpha=0.3,
        lw=0,
        label=r"$\pm1\sigma$ interval",
    )

    if cfg.true_path is not None:
        # Read the _true.csv file written by read_latte.ipynb
        # files with '_true.csv' have the following properties:
        # radius in kpc, mass in Msun
        # columns arranged as [r, Menc] where Menc is the true enclosed mass
        # rows are sorted by increasing radius
        r_true, M_true = np.loadtxt(
            fname=cfg.true_path, delimiter=",", skiprows=1, unpack=True
        )

        def frac_error(r_est, r_true, M_est, M_true):
            frac_error = np.zeros(len(r_est))
            for i in range(len(r_est)):
                match_idx = (np.abs(r_true - r_est[i])).argmin()
                frac_error[i] = (M_est[i] - M_true[match_idx]) / M_true[match_idx]
            return frac_error

        axs[0].plot(
            r_true, M_true, c="k", linewidth=1.5, linestyle="dashed", label="True"
        )

        axs[1].plot(r, frac_error(r, r_true, Menc_med, M_true), c=color, linewidth=2.0)
        axs[1].fill_between(
            r,
            frac_error(r, r_true, Menc_low, M_true),
            frac_error(r, r_true, Menc_upp, M_true),
            color=color,
            alpha=0.3,
            lw=0,
        )

    axs[0].set_title(cfg.figs_path)
    axs[0].set_ylim([0.9 * min(Menc_med), 1.1 * Menc_med[-1]])
    axs[1].axhline(0, c="k", linewidth=1)
    axs[1].axhline(0.2, c="k", linewidth=0.5, linestyle="dotted")
    axs[1].axhline(-0.2, c="k", linewidth=0.5, linestyle="dotted")
    axs[1].set_xlim([cfg.min_r, cfg.max_r])
    axs[0].set_ylabel(r"$M(<r) (M_{\odot})$", size=24)
    axs[1].set_xlabel("Galactocentric Radius (kpc)", size=24)
    axs[1].set_ylabel("Fractional Error", size=20)
    axs[0].legend()
    axs[1].set_ylim([-0.40, 0.40])
    axs[1].set_yticks([-0.2, 0, 0.2])

    for ax in axs:
        ax.label_outer()

    plt.tight_layout()
    plt.savefig(f"{cfg.figs_path}mass_enc.pdf", dpi=200, bbox_inches="tight")
    plt.cla()


def plot_anisotropy(cfg: Config):
    plt.figure(figsize=(7, 7))
    plt.plot(r, beta_med, label=r"$\beta$")
    plt.fill_between(r, beta_low, beta_upp, alpha=0.3, label=r"$\pm1\sigma$ interval")
    plt.axhline(0, c="k", label=r"$\beta=0$")
    plt.title(cfg.figs_path)
    plt.xlabel("Galactocentric radius (kpc)")
    plt.ylabel(r"Anisotropy ($\beta$)", fontsize=18)
    plt.ylim([-1.0, 1.0])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.figs_path}anisotropy.pdf", dpi=200, bbox_inches="tight")
    plt.cla()


def plot_survey_footprint(cfg: Config):
    plt.figure()
    plt.scatter(
        l,
        b,
        s=2,
        c=Gapp,
        cmap="hell",
        vmin=Gmin,
        vmax=Gmax + 1,
        edgecolors="none",
        rasterized=True,
    )
    plt.colorbar(label="Gapp")
    if (
        cfg.blow_rad <= -cfg.bmin * d2r
    ):  # selection region in the southern Galactic hemisphere
        bb = np.linspace(cfg.blow_rad, -cfg.bmin * d2r, 100)
        l1 = cfg.lmin_func(bb)
        l2 = 2 * cfg.lsym - l1
        plt.plot(
            np.hstack((l1, l2[::-1], l1[0])) / d2r,
            np.hstack((bb, bb[::-1], bb[0])) / d2r,
            "g",
        )
    if (
        cfg.bupp >= cfg.bmin * d2r
    ):  # selection region in the northern Galactic hemisphere
        bb = np.linspace(cfg.bupp, cfg.bmin * d2r, 100)
        l1 = cfg.lmin_func(bb)
        l2 = 2 * cfg.lsym - l1
        plt.plot(
            np.hstack((l1, l2[::-1], l1[0])) / d2r,
            np.hstack((bb, bb[::-1], bb[0])) / d2r,
            "g",
        )
    plt.title(cfg.figs_path)
    plt.xlabel("galactic longitude l (degrees)")
    plt.ylabel("galactic latitude b (degrees)")
    plt.tight_layout()
    plt.savefig(f"{cfg.figs_path}sel_bounds.pdf", dpi=200, bbox_inches="tight")
    plt.cla()


def plot_walker(params, sampler, paramnames, maxloglike, i_iter, chain, cfg: Config):
    axes = plt.subplots(len(params) + 1, 1, sharex=True, figsize=(10, 10))[1]
    for i in range(len(params)):
        axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.3)
        axes[i].set_xticklabels([])
        axes[i].set_ylabel(paramnames[i])
    axes[0].set_title(cfg.figs_path)
    axes[-1].plot(sampler.lnprobability.T, color="k", alpha=0.3)
    axes[-1].set_ylabel("likelihood")  # bottom panel is the evolution of likelihood
    axes[-1].set_ylim(maxloglike - 3 * len(params), maxloglike)
    plt.tight_layout(h_pad=0)
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(
        f"{cfg.figs_path}param_evol_iter{i_iter}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.cla()
    # 2. corner plot - covariances of all parameters
    corner.corner(
        chain,
        quantiles=[0.16, 0.5, 0.84],
        labels=paramnames,
        show_titles=True,
    )
    plt.title(cfg.figs_path)
    plt.savefig(
        f"{cfg.figs_path}corner_iter{i_iter}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.cla()
