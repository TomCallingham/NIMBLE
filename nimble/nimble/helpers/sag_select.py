# Credit to Amanda Bystrom
import numpy as np
from scipy.interpolate import interp1d, splrep, BSpline


def compute_Lambda(ra, dec):
    """Computes the Sgr coordinate longitude Lambda"""
    return np.rad2deg(
        np.arctan2(
            -0.93595354 * np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
            - 0.31910658 * np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
            + 0.14886895 * np.sin(np.deg2rad(dec)),
            0.21215555 * np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
            - 0.84846291 * np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
            - 0.48487186 * np.sin(np.deg2rad(dec)),
        )
    )


def compute_Beta(ra, dec):
    """Computes the Sgr coordinate latitude Beta"""
    return np.rad2deg(
        np.arcsin(
            0.28103559 * np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
            - 0.42223415 * np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
            + 0.86182209 * np.sin(np.deg2rad(dec))
        )
    )


def smooth_splines(x, y):
    """Smoothing functions"""
    tck_s = splrep(x, y, s=len(x))
    return BSpline(*tck_s)(x)


def select_Sgr_leadarm(Beta, Lambda, distance):
    """This function creates a boolean array where True means that a star belongs to Sgr leading arm"""

    HernitschekDleadarm = [
        28.83,
        14.3,
        34.14,
        36.65,
        45.94,
        50.5,
        52.59,
        49.19,
        46.22,
        40.59,
        34.8,
        31.19,
        25.9,
        21.34,
        19.66,
        16.2,
    ]
    HernitschekDsigleadarm = [
        1.621,
        2.8,
        2.8,
        4.1,
        3.68,
        3.33,
        4.52,
        3.75,
        4.66,
        3.88,
        6.3,
        3.08,
        5.0,
        2.7,
        2.05,
        3.4,
    ]
    Lambdabinleadarm = [
        5,
        15,
        25,
        35,
        45,
        55,
        65,
        75,
        85,
        95,
        105,
        115,
        125,
        135,
        145,
        155,
    ]

    leadarm_func = interp1d(Lambdabinleadarm, HernitschekDleadarm)
    leadarmsig_func = interp1d(Lambdabinleadarm, HernitschekDsigleadarm)

    x = np.arange(np.min(Lambdabinleadarm), np.max(Lambdabinleadarm) + 0.1, 0.1)
    leadarmsmooth_func = interp1d(x, smooth_splines(x, leadarm_func(x)))
    leadarmsigsmooth_func = interp1d(x, smooth_splines(x, leadarmsig_func(x)))

    # Lambdashift = np.array([(e+360) if (e < 0) else e for e in Lambda])
    Lambdashift = np.mod(Lambda, 360)
    interp_range = (Lambdashift < np.max(x)) & (Lambdashift > np.min(x))
    Sgrleadarmdist_idx = np.array(np.zeros(len(Lambdashift)), dtype=bool)

    d_mid = leadarmsmooth_func(Lambdashift[interp_range])
    d_sigma = leadarmsigsmooth_func(Lambdashift[interp_range])
    Sgrleadarmdist_idx[interp_range] = (
        np.abs(distance[interp_range] - d_mid) < 3 * d_sigma
    )
    Sgrcoord_idx = np.abs(Beta) < 15

    return Sgrleadarmdist_idx & Sgrcoord_idx


def select_Sgr_trailarm(Beta, Lambda, distance):
    """This function creates a boolean array where True means that a star belongs to Sgr trailing arm"""

    HernitschekDtrailarm = [
        55.4,
        62.3,
        57.2,
        66.9,
        81.3,
        83.1,
        89.02,
        92.98,
        86.7,
        60,
        53,
        43.15,
        36.55,
        31.17,
        28.41,
        25.57,
        24.7,
        18,
        20.34,
        21.2,
        20.8,
        21.66,
        22,
        20.1,
        19.7,
        27.605,
    ]
    HernitschekDsigtrailarm = [
        3.2,
        3.5,
        2.3,
        5.8,
        6.1,
        5.2,
        5.13,
        8.99,
        10.5,
        2.8,
        6.78,
        6.65,
        6.28,
        6.16,
        4.66,
        5.14,
        4.86,
        7.7,
        4.44,
        4.7,
        5.17,
        4.84,
        4.41,
        5.3,
        6.43,
        1.245,
    ]
    Lambdabintrailarm = [
        105,
        115,
        125,
        135,
        145,
        155,
        165,
        175,
        185,
        195,
        205,
        215,
        225,
        235,
        245,
        255,
        265,
        275,
        285,
        295,
        305,
        315,
        325,
        335,
        345,
        355,
    ]

    trailarm_func = interp1d(Lambdabintrailarm, HernitschekDtrailarm)
    trailarmsig_func = interp1d(Lambdabintrailarm, HernitschekDsigtrailarm)

    x = np.arange(np.min(Lambdabintrailarm), np.max(Lambdabintrailarm) + 0.1, 0.1)
    trailarmsmooth_func = interp1d(x, smooth_splines(x, trailarm_func(x)))
    trailarmsigsmooth_func = interp1d(x, smooth_splines(x, trailarmsig_func(x)))

    Lambdashift = np.mod(Lambda, 360)
    interp_range = (Lambdashift < np.max(x)) & (Lambdashift > np.min(x))
    Sgrtrailarmdist_idx = np.array(np.zeros(len(Lambdashift)), dtype=bool)

    d_mid = trailarmsmooth_func(Lambdashift[interp_range])
    d_sigma = trailarmsigsmooth_func(Lambdashift[interp_range])
    Sgrtrailarmdist_idx[interp_range] = (
        np.abs(distance[interp_range] - d_mid) < 3 * d_sigma
    )
    Sgrcoord_idx = np.abs(Beta) < 15

    return Sgrtrailarmdist_idx & Sgrcoord_idx


def select_Sgr(ra, dec, distance):
    """This function combines stars in both the leading and the trailing arm of the Sgr stream."""

    Lambda = compute_Lambda(ra, dec)
    Beta = compute_Beta(ra, dec)

    Sgr_leadarm = select_Sgr_leadarm(Beta, Lambda, distance)
    Sgr_trailarm = select_Sgr_trailarm(Beta, Lambda, distance)

    Sgr_flag = Sgr_leadarm | Sgr_trailarm

    return Sgr_flag
