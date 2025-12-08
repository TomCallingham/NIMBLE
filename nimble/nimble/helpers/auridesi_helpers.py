from astropy.table import Table

# from astropy.io import fits
import astropy.coordinates as coord
import astropy.units as u

import numpy as np

# import pandas as pd
import h5py
import agama
import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon

Gmax = 19.0
Gmin = 16.0
Grrl = 0.58
DMerr = 0.24
bmin = 20.0
decmin = -35.0
d2r = np.pi / 180

# adapted from Alex Riley's AuriDESI tutorial and Namitha's how_to_use_mocks.ipynb
# https://github.com/desimilkyway/tutorials/blob/main/ahriley/auridesi-demo-hawaii.ipynb
nersc_path = (
    "/global/cfs/cdirs/desi/users/namitha/Aurigaia/AuriDESI_Mocks_Spectroscopic_Catalog"
)
auridesi_folder = "data/AuriDESI/"
auridesi_folder = "~/data/observations/local_desi/y3kp/field-stars/mocks/auridesi/"


def get_lsr_frame(halonum):
    galcen_distance = 8.000  # * u.kpc
    z_sun = 2e-2  # * u.kpc

    usun = 11.1
    vsun = 12.24
    wsun = 7.25

    if halonum == "06":
        vlsr = 229.2225816451948
    elif halonum == "16":
        vlsr = 213.5378191707707
    elif halonum == "21":
        vlsr = 226.5683036672675
    elif halonum == "23":
        vlsr = 234.6171739867179
    elif halonum == "24":
        vlsr = 218.7874144767017
    elif halonum == "27":
        vlsr = 250.8076194638379
    else:
        print("Could not find halo")
        exit()

    vsun = vsun + vlsr
    galcen_v_sun = [usun, vsun, wsun]  # * u.km / u.s

    return galcen_distance, galcen_v_sun, z_sun


def old_halo_velocity_density_profiles(halonum):
    filename = (
        f"data/auriga/H{halonum}/snapshot_reduced_temprho_halo_{halonum}_063.hdf5"
    )

    with h5py.File(filename, "r") as hf:
        # Extract data from Auriga hdf5 file
        coordinates = (
            np.asarray(hf["PartType4"]["Coordinates"][:]) * 1000
        )  # (Z, Y, X) in kpc
        velocities = np.asarray(
            hf["PartType4"]["Velocities"][:]
        )  # (Vz, Vy, Vx) in km/s
        metallicity = np.asarray(hf["PartType4"]["GFM_Metallicity"][:])
        form_time = np.asarray(hf["PartType4"]["GFM_StellarFormationTime"][:])  # Gyr

    # Set floor for halo metallicity
    metal_mask = metallicity < 1e-7
    metallicity[metal_mask] = 1e-7
    log_metallicity = np.log10(metallicity / 0.0127)

    # Old, low metallicity to select halo - also removes wind particles
    halo = (log_metallicity < -1.5) & (form_time > 8)

    x = coordinates[:, 2][halo]
    y = coordinates[:, 1][halo]
    z = coordinates[:, 0][halo]

    vx = velocities[:, 2][halo]
    vy = velocities[:, 1][halo]
    vz = velocities[:, 0][halo]

    radii = np.sqrt((x**2) + (y**2) + (z**2))
    rvel, tvel, pvel = cartesian_to_spherical(x, y, z, vx, vy, vz)

    truesig_knots = np.logspace(0, np.log10(100), 5)

    # velocity squared as function of log r
    true_sigmar = agama.splineApprox(np.log(truesig_knots), np.log(radii), rvel**2)
    true_sigmat = agama.splineApprox(
        np.log(truesig_knots), np.log(radii), (tvel**2 + pvel**2) / 2
    )

    return true_sigmar, true_sigmat, radii


true_vel_folder = f"~/data/observations/local_desi/y3kp/true-profiles/auriga/velocity-profiles-4monica/"


import pickle
from scipy.interpolate import UnivariateSpline


def halo_velocity_density_profiles(halonum):
    halofile = f"{true_vel_folder}halo{halonum}-level3-profiles-4monica.pickle"
    # print(halofile)
    with open(halofile, "rb") as handle:
        rets = pickle.load(handle)

    edges = rets["edges"]  # kpc, bin edges
    counts = rets["counts"]  # Nstars
    vmeans = rets["means"]  # km/s
    v2s = rets["v2s"]  # km^2/s^2
    sigmas = rets["sigmas"]  # km/s
    density = rets["density"]  # Msun/kpc^3

    # Ignoring mean velocities for now. Need to modify the deconv routine to use it
    radii = edges[1:]
    true_sigmar = UnivariateSpline(radii, sigmas[0, :], s=50)
    true_sigmat = UnivariateSpline(
        radii, np.sqrt(sigmas[1, :] ** 2 + sigmas[2, :] ** 2), s=50
    )
    true_tracedens = UnivariateSpline(radii, density, s=50)

    return true_sigmar, true_sigmat, true_tracedens, radii


def load(halonum, lsrdeg, SUBSAMPLE, VERBOSE):
    # fname = f"{auridesi_folder}/{lsrdeg}_deg/H{halonum}_{lsrdeg}deg_rrl.fits"

    fname = (
        f"{auridesi_folder}halo{int(halonum)}/halo{int(halonum)}-view{lsrdeg}-rrl.fits"
    )

    rvtab = Table.read(fname, hdu="rvtab")
    fibermap = Table.read(fname, hdu="fibermap")
    gaia = Table.read(fname, hdu="gaia")
    # true =     Table.read(fname, hdu='true_value')

    print("Number of initial particles:", len(rvtab))

    lsr_info = get_lsr_frame(halonum)

    ra = np.asarray(gaia["RA"])  # deg
    dec = np.asarray(gaia["DEC"])  # deg
    ra *= d2r  # ra and dec to rad
    dec *= d2r

    pmra = np.asarray(gaia["PMRA"])  # mas/yr
    pmdec = np.asarray(gaia["PMDEC"])  # mas/yr
    pmra_error = np.asarray(gaia["PMRA_ERROR"])  # mas/yr
    pmdec_error = np.asarray(gaia["PMDEC_ERROR"])  # mas/yr

    # TODO: fix
    # BUG: PM_ERR should be quadrature? Or is this the average error? Could have correlated
    PMerr = (pmra_error + pmdec_error) / 2  # mas/yr

    # Use fibermap['dist'] instead?
    Gapp = np.asarray(fibermap["GAIA_PHOT_G_MEAN_MAG"])  # mag
    # dist = 10**((Grrl - Gapp - 5)/-5)  # pc

    vlos = np.asarray(fibermap["v_sys_Braga"])  # km/s
    vloserr = np.asarray(fibermap["v_sys_Braga_err"])  # km/s

    l_rad, b_rad, pml, pmb = agama.transformCelestialCoords(
        agama.fromICRStoGalactic, ra, dec, pmra, pmdec
    )

    # back to degrees
    l_deg = l_rad / d2r
    b_deg = b_rad / d2r
    dec /= d2r

    filt = (abs(b_deg) >= bmin) * (dec >= decmin) * (Gapp > Gmin) * (Gapp < Gmax)

    true_sigmar, true_sigmat, true_dens_radii = halo_velocity_density_profiles(halonum)

    return (
        l_deg[filt],
        b_deg[filt],
        true_dens_radii,
        Gapp[filt],
        pml[filt],
        pmb[filt],
        vlos[filt],
        PMerr[filt],
        vloserr[filt],
        true_sigmar,
        true_sigmat,
        lsr_info,
    )


def load_BHB(halonum, lsrdeg, SUBSAMPLE, VERBOSE):
    fname = f"data/AuriDESI/{lsrdeg}_deg/H{halonum}_{lsrdeg}deg_rrl.fits"

    rvtab = Table.read(fname, hdu="rvtab")
    fibermap = Table.read(fname, hdu="fibermap")
    gaia = Table.read(fname, hdu="gaia")
    # true =     Table.read(fname, hdu='true_value')

    print("Number of initial particles:", len(rvtab))

    lsr_info = get_lsr_frame(halonum)

    ra = np.asarray(gaia["RA"])  # deg
    dec = np.asarray(gaia["DEC"])  # deg
    ra *= d2r  # ra and dec to rad
    dec *= d2r

    pmra = np.asarray(gaia["PMRA"])  # mas/yr
    pmdec = np.asarray(gaia["PMDEC"])  # mas/yr
    pmra_error = np.asarray(gaia["PMRA_ERROR"])  # mas/yr
    pmdec_error = np.asarray(gaia["PMDEC_ERROR"])  # mas/yr

    # TODO: fix
    # BUG: should be quadrature
    PMerr = (pmra_error + pmdec_error) / 2  # mas/yr

    # Use fibermap['dist'] instead?
    Gapp = np.asarray(fibermap["GAIA_PHOT_G_MEAN_MAG"])  # mag
    # dist = 10**((Grrl - Gapp - 5)/-5)  # pc

    vlos = np.asarray(fibermap["v_sys_Braga"])  # km/s
    vloserr = np.asarray(fibermap["v_sys_Braga_err"])  # km/s

    l, b, pml, pmb = agama.transformCelestialCoords(
        agama.fromICRStoGalactic, ra, dec, pmra, pmdec
    )

    # back to degrees
    l /= d2r
    b /= d2r
    dec /= d2r

    filt = (abs(b) >= bmin) * (dec >= decmin) * (Gapp > Gmin) * (Gapp < Gmax)

    true_sigmar, true_sigmat, true_dens_radii = halo_velocity_density_profiles(halonum)

    return (
        l[filt],
        b[filt],
        true_dens_radii,
        Gapp[filt],
        pml[filt],
        pmb[filt],
        vlos[filt],
        PMerr[filt],
        vloserr[filt],
        true_sigmar,
        true_sigmat,
        lsr_info,
    )


def cartesian_to_spherical(xpos, ypos, zpos, xVel, yVel, zVel):
    numParticles = len(xpos)

    sinTheta = np.sqrt(xpos**2 + ypos**2) / np.sqrt(xpos**2 + ypos**2 + zpos**2)
    cosTheta = zpos / np.sqrt(xpos**2 + ypos**2 + zpos**2)
    sinPhi = ypos / np.sqrt(xpos**2 + ypos**2)
    cosPhi = xpos / np.sqrt(xpos**2 + ypos**2)

    for i in range(0, numParticles):
        conversionMatrix = [
            [sinTheta[i] * cosPhi[i], sinTheta[i] * sinPhi[i], cosTheta[i]],
            [cosTheta[i] * cosPhi[i], cosTheta[i] * sinPhi[i], -sinTheta[i]],
            [-sinPhi[i], cosPhi[i], 0],
        ]

    velMatrix = [xVel, yVel, zVel]

    sphereVels = np.matmul(conversionMatrix, velMatrix)

    rVel = sphereVels[0]
    tVel = sphereVels[1]
    pVel = sphereVels[2]

    return rVel, tVel, pVel


def rv_to_gsr(c, v_sun=None):
    """
    From Adrian Price-Whelan (Astropy)
    Transform a barycentric radial velocity to the Galactic Standard of Rest
    (GSR).
    The input radial velocity must be passed in as a
    Parameters
    ----------
    c : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
        The radial velocity, associated with a sky coordinates, to be
        transformed.
    v_sun : `~astropy.units.Quantity`, optional
        The 3D velocity of the solar system barycenter in the GSR frame.
        Defaults to the same solar motion as in the
        `~astropy.coordinates.Galactocentric` frame.
    Returns
    -------
    v_gsr : `~astropy.units.Quantity`
        The input radial velocity transformed to a GSR frame.
    """
    if v_sun is None:
        v_sun = coord.Galactocentric().galcen_v_sun.to_cartesian()

    gal = c.transform_to(coord.Galactic)

    cart_data = gal.data.to_cartesian()

    unit_vector = cart_data / cart_data.norm()
    v_proj = v_sun.dot(unit_vector)

    return c.radial_velocity + v_proj
