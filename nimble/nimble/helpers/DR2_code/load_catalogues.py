from .selection_functions import get_p_obs
from astropy.table import Table
import astropy.coordinates as coord
from .utils import get_auridesi_frame, subsample_ids
import warnings

from .desi_data_loc import base_y3kp_folder


basepath = base_y3kp_folder

selfunc_kwargs = {"desi_release": "loa", "gaia_release": "dr2"}


def load_auridesi(halo, view, tracer, apply_cuts=True):
    """
    Load AuriDESI DR2 KP mock catalogue data for a specified halo, view, and tracer.

    Parameters
    ----------
    halo : int
        The halo number (6, 16, 21, 23, 24, 27).
    view : int
        The viewing angle (30, 120, 210, 300).
    tracer : str
        The tracer type to load ('bhb', 'rrl').
    apply_cuts : bool, optional
        If True, apply standard selection cuts. Default is True.
        These cuts remove:
        - Stars that are not observed in Loa
        - Stars that are bound to satellites (pop_id == 2)
        - Stars that are in massive streams (stream_flag_1e8 == True)
        - Further subsample to approximately match Loa numbers (7000 RRL, 5500 BHB)

    Returns
    -------
    obs : astropy.table.Table
        Table containing error-convolved properties of tracers.
    true : astropy.table.Table
        Table containing true properties of tracers.
    galcen_frame : astropy.coordinates.Galactocentric
        The Astropy frame used to convert from ICRS to Galactocentric coordinates.
    """
    cat_path = basepath + f"field-stars/mocks/auridesi/latest/halo{halo}/"
    cat_file = cat_path + f"halo{halo}-view{view}-{tracer}.fits"

    obs = Table.read(cat_file, hdu="obs")
    true = Table.read(cat_file, hdu="true")

    if apply_cuts:
        sel = obs["in_loa"] == True  # observed in loa (p_obs sampled)
        sel &= true["pop_id"] < 2  # not bound to satellite
        sel &= ~true["stream_flag_1e8"]  # not in massive stream

        obs = obs[sel]
        true = true[sel]

        # further subsample to approximately match Loa numbers (7000 RRL, 5500 BHB)
        n_tracers = 7000 if (tracer == "rrl") else 5500
        sel = subsample_ids(obs["id_loa"], n_tracers)
        obs = obs[sel]
        true = true[sel]

    galcen_frame = get_auridesi_frame(cat_file)

    return obs, true, galcen_frame


def load_bhbs(apply_cuts=True, get_pobs=True):
    """
    Load BHB stars from the Loa catalogue for DR2 KP analysis.

    Parameters
    ----------
    apply_cuts : bool, optional
        If True, applies standard selection cuts (see Notes). Default is True.
    get_pobs : bool, optional
        If True, computes and adds observation probability ('p_obs') column to the
        Fibermap table based on target coordinates. Default is True.

    Returns
    -------
    tuple of astropy.table.Table
        A tuple containing three filtered tables:
        - RVT : Table
            Radial velocity and spectroscopic data for selected BHBs.
        - FT : Table
            Fiber mapping data for selected BHBs. Includes 'p_obs' column if
            get_pobs=True.
        - GT : Table
            Gaia astrometric and photometric data for selected BHBs.

    Notes
    -----
    Selection criteria (what apply_cuts=True applies):
        - 'main' survey with 'bright' program
        - Not member of known globular cluster, dwarf galaxy, or Sagittarius stream
        - Primary target (PRIMARY == True)
        - RR_SPECTYPE == 'STAR' (not applied in Bystrom+2025)
    """
    catpath = basepath + "field-stars/loa/"
    print(catpath)
    catfile = catpath + "loa-bhb-latest.fits"  # symbolic link to latest version

    RVT = Table.read(catfile, hdu="RVTAB")
    FT = Table.read(catfile, hdu="FIBERMAP")
    GT = Table.read(catfile, hdu="GAIA")

    if apply_cuts:
        sel = (RVT["SURVEY"] == "main") & (RVT["PROGRAM"] == "bright")
        sel &= ~(FT["Sgr_flag"] | FT["GC_flag"] | FT["dwarf_flag"])
        sel &= RVT["PRIMARY"] == True
        sel &= RVT["RR_SPECTYPE"] == "STAR"

        RVT = RVT[sel]
        FT = FT[sel]
        GT = GT[sel]

    if get_pobs:
        # gets p_obs from Sergey's selection function (same as used in mocks)
        ra = FT["TARGET_RA"]
        dec = FT["TARGET_DEC"]
        p_obs = get_p_obs(ra=ra, dec=dec, tracer="bhb", **selfunc_kwargs)
        FT["p_obs"] = p_obs

    return RVT, FT, GT


def load_rrls(apply_cuts=True, get_pobs=True):
    """
    Load RRL stars from the Loa catalogue for DR2 KP analysis.

    Parameters
    ----------
    apply_cuts : bool, optional
        If True, removes associations with known substructures. Default is True.
    get_pobs : bool, optional
        If True, computes and adds observation probability ('p_obs') column to the
        table based on target coordinates. Default is True.

    Returns
    -------
    obs : astropy.table.Table
        Table containing DESI, Gaia, and variability data.

    Notes
    -----
    Selection criteria (applied to full catalogue, prior to apply_cuts):
        - 'main' survey with 'bright' program
        - Kiel space: 5000 < TEFF < 8500 and 0 < LOGG < 3.8 (removes outliers)
    """
    catpath = basepath + "field-stars/loa/"
    catfile = catpath + "loa-rrl_mass_modeling_wSubstructures_20260223.csv"

    tab = Table.read(catfile)

    if apply_cuts:
        sel = tab["system_name"].mask  # masks known substructures
        tab = tab[sel]

    if get_pobs:
        # gets p_obs from Gustavo's selection function (same as used in mocks)
        ra = tab["ra"]
        dec = tab["dec"]
        mag = tab["phot_g_mean_mag"]
        p_obs = get_p_obs(ra=ra, dec=dec, tracer="rrl", mag=mag, **selfunc_kwargs)
        tab["p_obs"] = p_obs

    return tab


def load_symphony(suite, halo, view, tracer, apply_cuts=True):
    """
    Load Symphony DR2 KP mock catalogue data for a specified suite, halo, view, and tracer.

    Parameters
    ----------
    suite : str
        The simulation suite ('symphony', 'mwest', 'eden', 'symphony-hr').
    halo : int
        The halo number. See utils.get_halo_list() for valid halo numbers for each suite.
    view : int
        The viewing index (0, 1, 2, 3, 4).
    tracer : str
        The tracer type to load ('bhb', 'rrl').
    apply_cuts : bool, optional
        If True, apply standard selection cuts. Default is True.
        These cuts remove:
        - Stars that are not observed in Loa
        - Stars that are too faint or too bright (16 < decam_r < 20)
        - Stars that are bound to satellites (mask_bound == False)

    Returns
    -------
    obs : astropy.table.Table
        Table containing error-convolved properties of tracers.
    true : astropy.table.Table
        Table containing true properties of tracers.
    galcen_frame : astropy.coordinates.Galactocentric
        The Astropy frame used to convert from ICRS to Galactocentric coordinates.
    """
    cat_path = basepath + f"field-stars/mocks/symphony/{suite}/halo{halo:03}/"
    cat_file = cat_path + f"halo{halo:03}-view{view}-{tracer}.fits"

    obs = Table.read(cat_file, hdu="obs")
    true = Table.read(cat_file, hdu="true")

    if apply_cuts:
        sel = obs["in_loa"] == True  # observed in loa (p_obs sampled)
        sel &= obs["decam_r"] < 20  # impose main/bright mag limits
        sel &= obs["decam_r"] > 16
        sel &= obs["mask_bound"] == True  # not bound to satellite

        obs = obs[sel]
        true = true[sel]

    with coord.galactocentric_frame_defaults.set("v4.0"):
        galcen_frame = coord.Galactocentric()

    return obs, true, galcen_frame
