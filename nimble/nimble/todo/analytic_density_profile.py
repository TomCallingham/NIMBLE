import numpy as np


def medina24rrl_rho(radii):
    # https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.4762M/abstract
    R_break = 18.0
    slope_inner = -2.05
    slope_outer = -4.47
    A1 = 0.67
    A2 = 1.52

    if isinstance(radii, float) or isinstance(radii, int):
        radii = [radii]

    log10dens = np.zeros(len(radii))
    for i, r in enumerate(radii):
        if r < R_break:
            log10dens[i] = A1 + slope_inner * np.log10(r / 8)
        else:
            log10dens[i] = A2 + slope_outer * np.log10(r / 8)

    return 10**log10dens


def medina24rrl_dlnrho(log_radii):
    # https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.4762M/abstract
    R_break = 18.0
    slope_inner = -2.05
    slope_outer = -4.47

    if isinstance(log_radii, float) or isinstance(log_radii, int):
        log_radii = [log_radii]

    dlnrho_dlnr = np.zeros(len(log_radii))
    for i, lr in enumerate(log_radii):
        if np.exp(lr) < R_break:
            dlnrho_dlnr[i] = slope_inner
        else:
            dlnrho_dlnr[i] = slope_outer

    return dlnrho_dlnr
