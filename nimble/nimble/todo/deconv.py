from multiprocess import Pool
from datetime import datetime

import scipy.optimize
import numpy as np
import emcee
import agama

# import time
import sys
import os


from .plotting import (
    plot_profiles,
    plot_anisotropy,
    plot_mass_enclosed,
    plot_survey_footprint,
    plot_walker,
)

from .footprint_boundary import getSurveyFootprintBoundary

from .analytic_density_profile import medina24rrl_dlnrho

from .config import Config

np.set_printoptions(linewidth=200, precision=6, suppress=True)
np.random.seed(42)

d2r = np.pi / 180  # conversion from degrees to radians


def main(argv=None):
    if argv is None:
        argv = sys.argv

    cfg = Config.from_args(argv)

    print(cfg.Gmax, cfg.Gmin, cfg.min_r, cfg.max_r, cfg.figs_path)

    # kind, load_fnc, load_params, figs_path, true_path = parse_args(sys.argv)

    (
        l_deg,
        b_deg,
        true_dens_radii,
        Gapp,
        pml,
        pmb,
        vlos,
        PMerr,
        vloserr,
        true_sigmar,
        true_sigmat,
        lsr_info,
    ) = cfg.load_data()

    true_data = {
        "true_dens_radii": true_dens_radii,
        "true_sigmar": true_sigmar,
        "true_sigmat": true_sigmat,
    }

    if cfg.verbose:
        print("Number of particles in survey volume:", len(l_deg))
        print("Gmax:", cfg.Gmax)
        print("Gmin:", cfg.Gmin)
        print("DMerr:", cfg.DMerr)
        print("bmin:", cfg.bmin)
        print("decmin:", cfg.decmin)
        print("min_r:", cfg.min_r)
        print("max_r:", cfg.max_r)
        if cfg.use_external_density:
            print("Using external density profile")

    if not os.path.exists(cfg.figs_path):
        os.makedirs(cfg.figs_path)
        if cfg.verbose:
            print(f"created output directory for figures at {cfg.figs_path}")
    print()

    blow_rad, bupp, lmin, lsym = getSurveyFootprintBoundary(cfg.decmin)
    # cfg holds Survey footprint_boundary

    # diagnostic plot showing the stars in l,b and the selection region boundary
    if cfg.verbose:
        plot_survey_footprint(cfg)

    # this used to be a function, but MCMC parallelization doesn't work unless the likelihood fnc
    # is in the global scope (possibly fixed in the latest EMCEE version - haven't checked)
    # modelDensity
    from .modelling import modelDensity, modelSigma, likelihood

    npoints = len(l_deg)

    # convert l,b,dist.mod. of all stars into log of Galactocentric radius (observed, not true)
    # unit conversion: degrees to radians for l,b,  mas/yr to km/s/kpc for PM
    dist_obs = 10 ** (0.2 * (Gapp - cfg.Grrl) - 2)

    x, y, z, vx, vy, vz = agama.getGalactocentricFromGalactic(
        l_deg * d2r, b_deg * d2r, dist_obs, pml * 4.74, pmb * 4.74, vlos, *lsr_info
    )
    logr_obs = 0.5 * np.log(x**2 + y**2 + z**2)
    vr_obs = (x * vx + y * vy + z * vz) / np.exp(logr_obs)
    vt_obs = (0.5 * (vx**2 + vy**2 + vz**2 - vr_obs**2)) ** 0.5

    # create random samples from distance modulus unc for each star and convert to Galactocentric r
    nsamples = 20  # number of random samples per star

    Gsamp = (
        np.random.normal(size=(npoints, nsamples)) * cfg.DMerr + Gapp[:, None]
    ).reshape(-1)

    dist_samp = 10 ** (0.2 * (Gsamp - cfg.Grrl) - 2)

    def find_Galactic_samples(l_deg, b_deg, dist_samp, pml, pmb):
        x, y, z = agama.getGalactocentricFromGalactic(
            np.repeat(l_deg * d2r, nsamples),
            np.repeat(b_deg * d2r, nsamples),
            dist_samp,
            galcen_distance=lsr_info[0],
            galcen_v_sun=lsr_info[1],
            z_sun=lsr_info[2],
        )
        R = (x**2 + y**2) ** 0.5
        r = (x**2 + y**2 + z**2) ** 0.5  # array of samples for Galactocentric radius
        logr_samp = np.log(r)

        # a rather clumsy way of constructing the matrices describing how intrinsic 3d velocity
        # dispersions are translated to the Vlos and PM dispersions at each data sample: first
        # compute the expected mean values (pml, pmb, vlos) for a star at rest at a given distance,
        # then repeat the exercise 3 times, setting one of velocity components (v_r, v_theta, v_phi)
        # to 1 km/s, and subtract from the zero-velocity projection.
        vel0 = np.array(
            agama.getGalacticFromGalactocentric(
                x, y, z, x * 0, y * 0, z * 0, *lsr_info
            )[3:6]
        )
        velr = (
            np.array(
                agama.getGalacticFromGalactocentric(
                    x, y, z, x / r, y / r, z / r, *lsr_info
                )[3:6]
            )
            - vel0
        )
        velt = (
            np.array(
                agama.getGalacticFromGalactocentric(
                    x, y, z, z / r * x / R, z / r * y / R, -R / r, *lsr_info
                )[3:6]
            )
            - vel0
        )
        velp = (
            np.array(
                agama.getGalacticFromGalactocentric(
                    x, y, z, -y / R, x / R, 0 * r, *lsr_info
                )[3:6]
            )
            - vel0
        )

        # matrix of shape (2, npoints*nsamples) describing how the two intrinsic velocity dispersions
        # in 3d Galactocentric coords translate to the Vlos dispersion at each sample point
        mat_vlos = np.array([velr[2] ** 2, velt[2] ** 2 + velp[2] ** 2])
        # same for the PM dispersions: this is a 2x2 symmetric matrix for each datapoint,
        # characterized by two diagonal and one off-diagonal elements,
        # and each element is computed from the two Galactocentric intrinsic velocity dispersions
        mat_pm = (
            np.array(
                [
                    [velr[0] * velr[0], velt[0] * velt[0] + velp[0] * velp[0]],
                    [velr[1] * velr[1], velt[1] * velt[1] + velp[1] * velp[1]],
                    [velr[0] * velr[1], velt[0] * velt[1] + velp[0] * velp[1]],
                ]
            )
            / 4.74**2
        )
        # difference between the measured PM and Vlos values and expected mean values at each sample
        # (the latter correspond to a zero 3d velocity, translated to the Heliocentric frame)
        pml_samp = np.repeat(pml, nsamples) - vel0[0] / 4.74
        pmb_samp = np.repeat(pmb, nsamples) - vel0[1] / 4.74
        vlos_samp = np.repeat(vlos, nsamples) - vel0[2]
        return mat_vlos, mat_pm, pml_samp, pmb_samp, vlos_samp, logr_samp

    mat_vlos, mat_pm, pml_samp, pmb_samp, vlos_samp, logr_samp = find_Galactic_samples(
        l_deg, b_deg, dist_samp, pml, pmb
    )

    # vectors of PM and Vlos errors for each sample, to be added to the model covariance matrices
    # here pmlerr,pmberr identical, but in general may be different
    fit_data = {}
    fit_data["pmlerr2_samp"] = np.repeat(PMerr, nsamples) ** 2
    fit_data["pmberr2_samp"] = np.repeat(PMerr, nsamples) ** 2
    fit_data["vloserr2_samp"] = np.repeat(vloserr, nsamples) ** 2

    # knots in Galcen radius (minimum is imposed by our cut |b|>30, maximum - by the extent of data)
    knots_logr = np.linspace(np.log(cfg.min_knot), np.log(cfg.max_knot), cfg.num_knots)

    # initial values of parameters
    # log of (un-normalized) density values at radial knots
    params_dens = -(np.linspace(1, 3, cfg.num_knots - 1) ** 2)
    # log of radial velocity dispersion values at the radial knots
    params_sigmar = np.full(cfg.num_knots, 5.0)
    params_sigmat = np.full(cfg.num_knots, 5.0)  # same for tangential dispersion
    params = np.hstack((params_dens, params_sigmar, params_sigmat))
    paramnames = (
        ["logrho(r=%4.1f)" % r for r in np.exp(knots_logr[1:])]
        + ["sigmar(r=%4.1f)" % r for r in np.exp(knots_logr)]
        + ["sigmat(r=%4.1f)" % r for r in np.exp(knots_logr)]
    )
    prevmaxloglike = -np.inf
    prevavgloglike = -np.inf
    # first find the best-fit model by deterministic optimization algorithm,
    # restarting it several times until it seems to arrive at the global minimum
    num_tries = 0
    scfg = cfg.to_small()
    while True:
        if cfg.verbose:
            print("\033[1;37mStarting deterministic search\033[0m")
        # minimization algorithm - so provide a negative likelihood to it
        params = scipy.optimize.minimize(
            lambda x: -likelihood(x, fit_data, scfg),
            params,
            method="Nelder-Mead",
            options=dict(maxfev=500),
        ).x
        maxloglike = likelihood(params, fit_data, scfg)
        if maxloglike - prevmaxloglike < 1.0:
            if cfg.verbose:
                for i in range(len(params)):
                    print("%s = %8.4g" % (paramnames[i], params[i]))
                print("Converged")
            break
        elif cfg.verbose:
            print("Improved log-likelihood by %f" % (maxloglike - prevmaxloglike))
        prevmaxloglike = maxloglike

        num_tries += 1

        if num_tries >= 100:
            print("Too many tries in deterministic search")
            exit()

    # show profiles and wait for the user to marvel at them
    plot_profiles(params.reshape(1, -1), true_data, cfg, "preMCMC")

    # then start a MCMC around the best-fit params
    paramdisp = (
        np.ones(len(params)) * 0.1
    )  # spread of initial walkers around best-fit params
    nwalkers = 2 * len(params)  # minimum possible number of walkers in emcee
    nsteps = 500  # 1000
    walkers = np.empty((nwalkers, len(params)))
    numtries = 0
    for i in range(nwalkers):
        while (
            numtries < 10000
        ):  # ensure that we initialize walkers with feasible values
            walker = params + np.random.randn(len(params)) * paramdisp
            if np.isfinite(likelihood(walker, fit_data, scfg)):
                walkers[i] = walker
                break
            numtries += 1
    if numtries >= 10000:
        raise RuntimeError("cannot initialize MCMC")
    with Pool() as pool:
        # numthreads = nwalkers//2  # parallel threads in emcee
        sampler = emcee.EnsembleSampler(nwalkers, len(params), likelihood, pool=pool)
        if cfg.verbose:
            print("\033[1;37mStarting MCMC search\033[0m")
        converged = False
        i_iter = 0
        while not converged:  # run several passes until log-likelihood converges
            sampler.run_mcmc(walkers, nsteps, progress=True)
            walkers = sampler.chain[:, -1]
            chain = sampler.chain[:, -nsteps:].reshape(-1, len(params))
            maxloglike = np.max(sampler.lnprobability[:, -nsteps:])
            avgloglike = np.mean(sampler.lnprobability[:, -nsteps:])
            walkll = sampler.lnprobability[:, -1]
            if cfg.verbose:
                for i in range(len(params)):
                    print(
                        "%s = %8.4g +- %7.4g"
                        % (paramnames[i], np.mean(chain[:, i]), np.std(chain[:, i]))
                    )
                print(
                    "max loglikelihood: %.2f, average: %.2f" % (maxloglike, avgloglike)
                )
            converged = (
                abs(maxloglike - prevmaxloglike) < 1.0
                and abs(avgloglike - prevavgloglike) < 2.0
            )
            prevmaxloglike = maxloglike
            prevavgloglike = avgloglike
            if converged:
                if cfg.verbose:
                    print("\033[1;37mConverged\033[0m")
                plot_profiles(chain[::20], true_data, cfg, "converged")

            # produce diagnostic plots after each MCMC episode:
            # 1. evolution of parameters along the chain for each walker
            if cfg.verbose:
                plot_walker(params, sampler, paramnames, maxloglike, i_iter, chain, cfg)
            # 3. density and velocity dispersion profiles - same as before
            if not converged:
                plot_profiles(chain[::20], true_data, cfg, f"MCMC_iter{i_iter}")
            i_iter += 1

    # now, plug resulting density and sigma profiles into the jeans equation
    # one mass profile for each trial of MCMC

    r = np.logspace(np.log10(cfg.min_r), np.log10(cfg.max_r), 201)
    lr = np.log(r)
    G = 4.3e-6  # (kpc km2) / (s2 Msun)

    # Thin the chain
    chain_smpl = chain[::20]

    dlnrho, dlnsigr, sigr, sigt = np.zeros((4, len(chain_smpl), len(r)))
    for i in range(len(chain_smpl)):
        if not cfg.use_external_density:
            dlnrho[i] = modelDensity(chain_smpl[i, 0 : cfg.num_knots - 1], scfg)(
                lr, der=1
            )
        else:
            dlnrho[i] = medina24rrl_dlnrho(lr)
        dlnsigr[i] = modelSigma(
            chain_smpl[i, cfg.num_knots - 1 : 2 * cfg.num_knots - 1], scfg
        )(lr, der=1)
        sigr[i] = np.exp(
            modelSigma(chain_smpl[i, cfg.num_knots - 1 : 2 * cfg.num_knots - 1], scfg)(
                lr
            )
        )
        sigt[i] = np.exp(modelSigma(chain_smpl[i, 2 * cfg.num_knots - 1 :], scfg)(lr))

    Mencs, betas = np.zeros((2, len(chain_smpl), len(r)))
    for i in range(len(chain_smpl)):
        betas[i] = 1 - (sigt[i] ** 2 / sigr[i] ** 2)
        Mencs[i] = -(sigr[i] ** 2 * r / G) * (dlnrho[i] + 2 * dlnsigr[i] + 2 * betas[i])

    Menc_low, Menc_med, Menc_upp = np.percentile(Mencs, [16, 50, 84], axis=0)
    beta_low, beta_med, beta_upp = np.percentile(betas, [16, 50, 84], axis=0)
    finals_data = np.stack(
        [r, Menc_low, Menc_med, Menc_upp, beta_low, beta_med, beta_upp], axis=1
    )

    # Anisotropy plot
    if cfg.verbose:
        plot_anisotropy(cfg)

    plot_mass_enclosed(Menc_med, Menc_low, Menc_upp, cfg)

    cfg.write_csv(
        finals_data,
        "Menc_beta_final.csv",
        "r, Menc_low, Menc_med, Menc_upp, beta_low, beta_med, beta_upp",
    )

    print(f"FINISHED AT {cfg.figs_path}\n")


if __name__ == "__main__":
    main(sys.argv)
