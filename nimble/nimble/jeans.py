import numpy as np
import agama


def spline_d_log_dens_convert(func_log_dens):
    def func_d_log_dens(lr_space):
        return func_log_dens(lr_space, der=1)

    return func_d_log_dens


def jeans_result(DispModel, func_d_log_dens, lr_space):
    G = 4.3e-6  # (kpc km2) / (s2 Msun)
    res = {"lr_space": lr_space}
    r_space = np.exp(lr_space)

    # dln_rho = func_log_dens(lr_space, der=1)
    dln_rho = func_d_log_dens(lr_space)
    sigma_space = DispModel.sigma_space(lr_space)
    sigma_r, sigma_theta, sigma_phi = (
        sigma_space["vr"],
        sigma_space["vtheta"],
        sigma_space["vphi"],
    )
    dln_sigma_r = DispModel.dln_sigma_r(lr_space)

    beta = 1 - ((sigma_theta**2) + (sigma_phi**2)) / (2 * (sigma_r**2))

    M_enc = -(r_space * (sigma_r**2) / G) * (dln_rho + (2 * dln_sigma_r) + 2 * beta)
    term_sigma_r = r_space * (sigma_r**2) / G
    term_rho = dln_rho
    term_d_sigmar = 2 * dln_sigma_r
    term_beta = 2 * beta

    summed_terms = -(term_rho + term_d_sigmar + term_beta)

    res["r_space"] = r_space
    res["dln_rho"] = dln_rho
    res["sigma_space"] = sigma_space
    res["sigma_r"] = sigma_r
    res["sigma_theta"] = sigma_theta
    res["sigma_phi"] = sigma_phi

    res["dln_sigma_r"] = dln_sigma_r

    res["beta"] = beta

    res["term_sigma_r"] = term_sigma_r
    res["term_rho"] = term_rho
    res["term_d_sigmar"] = term_d_sigmar
    res["term_beta"] = term_beta

    res["summed_terms"] = summed_terms
    res["M_enc"] = M_enc

    return res


def multi_jeans_result(MultiDispModel, func_d_log_dens, lr_space):
    G = 4.3e-6  # (kpc km2) / (s2 Msun)
    res = {"lr_space": lr_space}
    r_space = np.exp(lr_space)

    # dln_rho = func_log_dens(lr_space, der=1)
    dln_rho = func_d_log_dens(lr_space)
    sigma_space = MultiDispModel.sigma_space(lr_space)
    sigma_r, sigma_theta, sigma_phi = (
        sigma_space["vr"],
        sigma_space["vtheta"],
        sigma_space["vphi"],
    )
    dln_sigma_r = MultiDispModel.dln_sigma_r(lr_space)

    beta = 1 - ((sigma_theta**2) + (sigma_phi**2)) / (2 * (sigma_r**2))

    M_enc = -(r_space * (sigma_r**2) / G) * (dln_rho + (2 * dln_sigma_r) + 2 * beta)
    term_sigma_r = r_space * (sigma_r**2) / G
    term_rho = dln_rho
    term_d_sigmar = 2 * dln_sigma_r
    term_beta = 2 * beta

    summed_terms = -(term_rho + term_d_sigmar + term_beta)

    res["r_space"] = r_space
    res["dln_rho"] = dln_rho
    # res["sigma_space"] = sigma_space
    res["sigma_r"] = sigma_r
    res["sigma_theta"] = sigma_theta
    res["sigma_phi"] = sigma_phi

    res["dln_sigma_r"] = dln_sigma_r

    res["beta"] = beta

    res["term_sigma_r"] = term_sigma_r
    res["term_rho"] = term_rho
    res["term_d_sigmar"] = term_d_sigmar
    res["term_beta"] = term_beta

    res["summed_terms"] = summed_terms
    res["M_enc"] = M_enc

    res_stats = {}
    for key, val in res.items():
        if len(val.shape) == 1:
            continue
        med, o1s, u1s, o2s, u2s = np.percentile(val, [50, 84, 16, 95, 5], axis=0)
        stats = {"med": med, "o1s": o1s, "u1s": u1s, "o2s": o2s, "u2s": u2s}
        res_stats[key] = stats

    return res, res_stats
