import numpy as np
import agama


def jeans_result(DispModel, func_log_dens, lr_space):
    G = 4.3e-6  # (kpc km2) / (s2 Msun)
    res = {"lr_space": lr_space}
    r_space = np.exp(lr_space)

    dln_rho = func_log_dens(lr_space, der=1)
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
