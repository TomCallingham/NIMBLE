import numpy as np
import scipy
from .config import SmallConfig

d2r = np.pi / 180  # conversion from degrees to radians


def modelDensity(params, scfg: SmallConfig, truedens=False):
    # params is the array of logs of density at radial knots, which must monotonically decrease
    # note that since the result is invariant w.r.t. the overall amplitude of the density
    # (it is always renormalized to unity), the first element of this array may be fixed to 0
    knots_logdens = np.hstack((0, params))
    if any(knots_logdens[1:] >= knots_logdens[:-1]):
        raise RuntimeError("Density is non-monotonic")
    # represent the spherically symmetric 1d profile log(rho) as a cubic spline in log(r)
    if truedens:
        dens_knots = np.linspace(np.log(1), np.log(200), len(knots_logdens))
    else:
        dens_knots = scfg.knots_logr
    # reg ensures monotonic spline
    logrho = agama.Spline(dens_knots, knots_logdens, reg=True)
    # check that the density profile has a finite total mass
    # (this is not needed for the fit, because normalization is computed over accessible volume,
    # but it generally makes sense to have a physically valid model for the entire space).
    slope_in = logrho(
        dens_knots[0], der=1
    )  # d[log(rho)]/d[log(r)], log-slope at lower radius
    slope_out = logrho(dens_knots[-1], der=1)
    if slope_in <= -3.0 or slope_out >= -3.0:
        raise RuntimeError(
            "Density has invalid asymptotic slope: inner=%.2f, outer=%.2f"
            % (slope_in, slope_out)
        )

    # now the difficult part: normalize the 3d density over the volume of the survey,
    # taking into account fuzzy outer boundary in distance due to obs.errors in abs.magnitude.
    # integration performed in scaled l, sin(b), dm=distance modulus  (l,b expressed in radians)
    # over region lmin(b)<=l<=lmax(b), bmin<=b<=bmax, Gmin-4*DMerr <= dm+Grrl <= Gmax+4*DMerr.
    def integrand(coords):
        # return the density times selection function times jacobian of coord transformation
        scaledl, sinb, dm = coords.T
        b_rad = np.arcsin(sinb)
        # unscale coords inside curved selection region: 0<=scaledl<=1 => lmin(b)<=l<=lmax(b)
        lminb = scfg.lmin_func(b_rad)  # lmin is actually in the data, as a callable!
        lmaxb = 2 * scfg.lsym - lminb
        l_rad = lminb + (lmaxb - lminb) * scaledl
        dist = 10 ** (0.2 * dm - 2)
        x, y, z = agama.getGalactocentricFromGalactic(
            l_rad,
            b_rad,
            dist,
            galcen_distance=scfg.lsr_info[0],
            galcen_v_sun=scfg.lsr_info[1],
            z_sun=scfg.lsr_info[2],
        )
        logr = np.log(x**2 + y**2 + z**2) * 0.5
        jac = dist**3 * np.log(10) / 5 * (lmaxb - lminb)
        # multiplicative factor <= 1 accounting for a gradual decline in selection probability
        # as stars become fainter that the limiting magnitude of the survey
        if scfg.DMerr == 0:
            mult = (
                (dm <= scfg.Gmax - scfg.Grrl) * (dm >= scfg.Gmin - scfg.Grrl)
            ).astype(float)
        else:
            mult = 0.5 * (
                scipy.special.erf((scfg.Gmax - scfg.Grrl - dm) / 2**0.5 / scfg.DMerr)
                - scipy.special.erf((scfg.Gmin - scfg.Grrl - dm) / 2**0.5 / scfg.DMerr)
            )
        return np.exp(logrho(logr)) * mult * jac

    # first compute the integral over the selection region in the northern Galactic hemisphere
    norm = agama.integrateNdim(
        integrand,
        lower=[0, np.sin(max(scfg.bmin * d2r, scfg.blow_rad)), Gmin - 4 * DMerr],
        upper=[1, np.sin(scfg.bupp), Gmax + 4 * DMerr],
        toler=1e-6,
    )[0]
    # then add the contribution from the region in the southern Galactic hemisphere
    if scfg.blow_rad <= -scfg.bmin * d2r:
        norm += agama.integrateNdim(
            integrand,
            lower=[0, np.sin(scfg.blow_rad), scfg.Gmin - 4 * scfg.DMerr],
            upper=[1, np.sin(-scfg.bmin * d2r), scfg.Gmax + 4 * scfg.DMerr],
            toler=1e-6,
        )[0]

    # now renormalize the density profile to have unit integral over the selection volume
    logrho = agama.Spline(dens_knots, knots_logdens - np.log(norm), reg=True)
    return logrho


def modelSigma(params, scfg: SmallConfig):
    # params is array of log(sigma(r)) at radial knots (applicable to both velocity components)
    return agama.Spline(scfg.knots_logr, params)


def likelihood(params, fit_data, scfg: SmallConfig):
    # function to be maximized in the MCMC and deterministic optimization
    params_dens = params[0 : scfg.n_knots]
    params_sigmar = params[scfg.n_knots - 1 : 2 * scfg.n_knots - 1]
    params_sigmat = params[2 * scfg.n_knots - 1 :]
    try:
        # compute the predicted density from the model at each data sample
        logrho = modelDensity(params_dens, scfg)(fit_data["logr_samp"])
        # likelihood of finding each data sample in given density profile (multiplied by r^3)
        like_dens = np.exp(logrho + 3 * fit_data["logr_samp"])
        # construct intrinsic velocity dispersion profiles of the model
        sigmar2 = np.exp(
            2 * modelSigma(scfg, params_sigmar)(fit_data["logr_samp"])
        )  # squared rad vel dispersion
        sigmat2 = np.exp(
            2 * modelSigma(scfg, params_sigmat)(fit_data["logr_samp"])
        )  # squared tangential --"--
        sigmaboth = np.vstack((sigmar2, sigmat2))  # shape: (2, nbody*nsamples)
        # convert profiles to values of line-of-sight velocity dispersion at each data sample
        cov_vlos = np.einsum("kp,kp->p", fit_data["mat_vlos"], sigmaboth)
        # same for PM dispersion profiles - diagonal & off-diagonal elements of PM cov matrix
        cov_pmll, cov_pmbb, cov_pmlb = np.einsum(
            "ikp,kp->ip", fit_data["mat_pm"], sigmaboth
        )
        # add individual observational errors for each data sample
        cov_vlos += fit_data["vloserr2_samp"]
        cov_pmll += fit_data[
            "pmlerr2_samp"
        ]  # here add to diagonal elements of PM cov matrix only,
        cov_pmbb += (
            fit_data[
                "pmberr2_samp"
            ]  # but with actual Gaia data should also use off-diagonal term
        )
        det_pm = cov_pmll * cov_pmbb - cov_pmlb**2  # determinant of the PM cov matrix
        # compute likelihoods of Vlos and PM values of each sample, accounting for obs.errors
        like_vlos = cov_vlos**-0.5 * np.exp(
            -0.5 * fit_data["vlos_samp"] ** 2 / cov_vlos
        )
        like_pm = det_pm**-0.5 * np.exp(
            -0.5
            / det_pm
            * (
                fit_data["pml_samp"] ** 2 * cov_pmbb
                + fit_data["pmb_samp"] ** 2 * cov_pmll
                - 2 * fit_data["pml_samp"] * fit_data["pmb_samp"] * cov_pmlb
            )
        )
        # the overall log-likelihood of the model:
        # first average the likelihoods of all sample points for each star -
        # corresponds to marginalization over distance uncs, also propagated to PM space
        # then sum up marginalized log-likelihoods of all stars.
        # at this stage may also add a prior if necessary
        loglikelihood = np.sum(
            np.log(
                np.mean(
                    (like_dens * like_pm * like_vlos).reshape(npoints, nsamples),
                    axis=1,
                )
            )
        )
        # print("%s => %.2f" % (params, loglikelihood))
        if not np.isfinite(loglikelihood):
            loglikelihood = -np.inf
        return loglikelihood
    except Exception as ex:
        print(ex)
        return -np.inf
