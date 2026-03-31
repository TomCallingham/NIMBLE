import sys
import numpy as np
import pandas as pd
import healpy as hp
from astropy import table
from astropy.coordinates import SkyCoord
from astropy import units as u

from .desi_data_loc import base_y3kp_folder

# basepath = '/global/cfs/cdirs/desi/science/mws/y3kp/selection-functions/'
basepath = f"{base_y3kp_folder}/selection-functions/"
mateu_path = basepath + "mateu-rrl-completeness/"
desi_path = basepath + "rrl/20260211/"

sys.path.append(mateu_path)
# import completeness_utils

# NOTE: this is Alex's version of Gustavo's code that looks up completeness for a given
# ra/dec/etc from pre-computed tables. It is more efficient for arrays of ra and dec
# than Gustavo's version. You should probably be looking at code/selection_functions.py
# instead of this file, unless you know what you're doing...


def query_2d_map_alex(C, coo):
    # modify completeness_utils.query_2d_map to work better with arrays

    # Healpix indices
    npix = C.hpix.size
    nside = hp.npix2nside(npix)

    # convert to theta, phi
    theta = np.radians(90.0 - coo.galactic.b.degree)
    phi = np.radians(coo.galactic.l.degree)

    # convert to HEALPix indices
    indices = hp.ang2pix(nside, theta, phi)

    assert np.all(np.arange(C.hpix.size) == C.hpix.values)
    return C.loc[indices]


def ObsComp(
    ra_in,
    dec_in,
    primsec,
    surv,
    prog,
    subset,
    Gaia=False,
    GaiaDR="",
    typ="",
    mag=-99,
    wdist=False,
    dist=-99,
):
    import completeness_utils
    # ra_in: R.A. to compute completeness
    # dec_in: Dec. to compute completeness
    # primsec: either 'prim' or 'sec', for primary or secondary targets
    # surv: e.g., 'main'
    # prog: e.g., 'bright' or 'dark'
    # subset: 'iron' (Y1) or 'jura' (Y3)
    # Gaia: should we also consider Gaia's completeness?
    # GaiaDR: 'DR2' or 'DR3'
    # typ: 'ab' or 'c'
    # mag: for the magnitude ranges provided by Mateu+20
    # wdist: should we consider the completeness as a function of distance?
    # dist: what distance is it? (in kpc)

    # AHR: version 2 is newer
    compcat = pd.read_csv(desi_path + f"tiles_wComp-rrl_{surv}_{prog}_{subset}2.csv")

    # single value for ra and dec (i.e., 1 point)
    if isinstance(ra_in, (int, float, complex, str)):
        # check distances
        c1 = SkyCoord(ra_in, dec_in, unit="deg", frame="icrs")  # your coords
        c2 = SkyCoord(
            compcat["TILERA"], compcat["TILEDEC"], unit="deg", frame="icrs"
        )  # objects in table

        # select the index of the minimum distance
        sep = c1.separation(c2)
        min_idx = np.argmin(sep.deg)

        if (
            Gaia == False
        ):  # we will return compcat*compGaia, so in this case we consider compGaia=1
            compGaia = 1.0
        elif Gaia == True:
            # transform c1 into Galactic coordinates first, because that's how Mateu's code work
            c1_l, c1_b = c1.galactic.l.deg, c1.galactic.b.deg
            coo = SkyCoord(l=c1_l * u.deg, b=c1_b * u.deg, frame="galactic")

            if GaiaDR == "DR2":
                if wdist == False:
                    C2D = pd.read_csv(
                        mateu_path + "maps/completeness2d.faint.rr%s.csv" % typ,
                        dtype=dict(hpix=np.int),
                    )

                    losC = completeness_utils.query_2d_map(C2D, coo)
                    if mag == -99:
                        compGaia = losC["Gaia_full"].iloc[0]
                    elif mag < 16:
                        compGaia = losC["Gaia[13,16]"].iloc[0]
                    elif mag >= 16 and mag < 18:
                        compGaia = losC["Gaia[16,18]"].iloc[0]
                    elif mag >= 18:
                        compGaia = losC["Gaia[18,22]"].iloc[0]

                elif wdist == True:
                    # Read 3D maps in the faint end (G>13), including full mag range completeness maps
                    gaia3D = pd.read_csv(
                        mateu_path + "maps/completeness3d.gaiadr2.vcsos.rr%s.csv" % typ,
                        dtype=dict(hpix=np.int),
                    )
                    Ci = completeness_utils.query_3d_map(
                        gaia3D, coo, D=dist * u.kpc, verbose=False, force_nearest=True
                    )
                    compGaia = Ci["C"].iloc[0]

            elif GaiaDR == "DR3":
                # transform c1 into Galactic coordinates first, because that's how Mateu's code work
                c1_l, c1_b = c1.galactic.l.deg, c1.galactic.b.deg
                coo = SkyCoord(l=c1_l * u.deg, b=c1_b * u.deg, frame="galactic")

                if wdist == False:
                    # for the 3d there is no specific catalog. The catalog doesnt make the distinction
                    C2D = pd.read_csv(
                        mateu_path + "maps/completeness2d.faint.rr%s.csv" % typ,
                        dtype=dict(hpix=np.int),
                    )

                    losC = completeness_utils.query_2d_map(C2D, coo)
                    if mag == -99:
                        compGaia = losC["Gaia_full"].iloc[0]
                    elif mag < 16:
                        compGaia = losC["Gaia[13,16]"].iloc[0]
                    elif mag >= 16 and mag < 18:
                        compGaia = losC["Gaia[16,18]"].iloc[0]
                    elif mag >= 18:
                        compGaia = losC["Gaia[18,22]"].iloc[0]

                elif wdist == True:
                    # Read 3D maps in the faint end (G>13), including full mag range completeness maps
                    gaia3D = pd.read_csv(
                        mateu_path
                        + "maps/completeness3d.gaiadr3.sos.filt_All.rr%s.csv" % typ,
                        dtype=dict(hpix=np.int),
                    )
                    Ci = completeness_utils.query_3d_map(
                        gaia3D, coo, D=dist * u.kpc, verbose=False, force_nearest=True
                    )
                    print(Ci["C"])
                    compGaia = Ci["C"].iloc[0]

        # return the completeness of that tile
        if (sep.deg)[min_idx] < 1.6:
            if "prim" in primsec:
                return compcat["obsfrac_prim"][min_idx] * compGaia
            elif "sec" in primsec:
                return compcat["obsfrac_sec"][min_idx] * compGaia
            else:
                # print('Error A')
                return np.nan

        # if minimum distance is larger than 1.6 deg, return np.nan (not observed) or zero
        else:
            # print('Error B')
            return np.nan

    # multiple values for ra and dec (i.e., arrays)
    elif isinstance(ra_in, np.ndarray):
        # to work with astropy tables instead of arrays or dataframes
        dfradec = pd.DataFrame()
        dfradec["ra"] = ra_in
        dfradec["dec"] = dec_in
        tradec = table.Table.from_pandas(dfradec)

        tcompcat = table.Table.from_pandas(compcat)

        # check distances
        c1 = SkyCoord(ra_in, dec_in, unit="deg", frame="icrs")  # your coords
        c2 = SkyCoord(
            tcompcat["TILERA"], tcompcat["TILEDEC"], unit="deg", frame="icrs"
        )  # objects in table

        #
        match, d2d, d3d = c1.match_to_catalog_sky(c2)
        compcat_intab = tcompcat[match]  # [d2d < 0.5*u.degree]
        tab_incompcat = tradec  # [d2d < 0.5*u.degree]

        tab_comb = table.hstack((tab_incompcat, compcat_intab))
        tab_comb["d2d"] = d2d
        # go back to pandas for this part
        dfradec = tab_comb.to_pandas()

        dfradec["obsfrac_return0"] = dfradec["obsfrac_%s" % primsec].copy()

        # AHR: started cleaning code here...
        if Gaia:
            # transform c1 into Galactic coordinates first, because that's how Mateu's code work
            c1_l, c1_b = c1.galactic.l.deg, c1.galactic.b.deg
            coo = SkyCoord(l=c1_l * u.deg, b=c1_b * u.deg, frame="galactic")

            if wdist:
                # Read 3D maps in the faint end (G>13), including full mag range completeness maps
                catstring = {"DR2": "gaiadr2.vcsos", "DR3": "gaiadr3.sos.filt_All"}[
                    GaiaDR
                ]
                gaia3D = pd.read_csv(
                    mateu_path + "maps/completeness3d.%s.rr%s.csv" % (catstring, typ),
                    dtype=dict(hpix=np.int),
                )
                Ci = completeness_utils.query_3d_map(
                    gaia3D, coo, D=dist * u.kpc, verbose=False, force_nearest=True
                )
                compGaia = Ci["C"]
            else:
                # catalog doesn't make distinction between DR2 and DR3 (AHR: really?)
                C2D = pd.read_csv(
                    mateu_path + "maps/completeness2d.faint.rr%s.csv" % typ,
                    dtype=dict(hpix=np.int),
                )

                # AHR: need to be more clever than the query here
                # losC = completeness_utils.query_2d_map(C2D,coo)
                losC = query_2d_map_alex(C2D, coo)

                if isinstance(mag, np.ndarray):
                    assert len(mag) == len(coo), "Length of mag and coords must match"
                    mag_bins = [(-np.inf, 16), (16, 18), (18, np.inf)]

                    compGaia = -np.ones(len(mag), dtype=float)
                    for bright_lim, faint_lim in mag_bins:
                        # silly magnitudes, tricks are for kids
                        mask = (mag >= bright_lim) & (mag < faint_lim)

                        if np.any(mask):
                            blim = 13 if bright_lim == -np.inf else bright_lim
                            flim = 22 if faint_lim == np.inf else faint_lim
                            key = f"Gaia[{blim},{flim}]"
                            compGaia[mask] = losC[key][mask].values

                    assert np.all(compGaia != -1), "Still have some compGaia < 0"
                else:
                    # Gustavo old code
                    if mag == -99:
                        compGaia = losC["Gaia_full"]
                    elif mag < 16:
                        compGaia = losC["Gaia[13,16]"]
                    elif mag >= 16 and mag < 18:
                        compGaia = losC["Gaia[16,18]"]
                    elif mag >= 18:
                        compGaia = losC["Gaia[18,22]"]
        else:
            # if Gaia is False, we will return compcat*compGaia, so in this case we consider compGaia=1
            compGaia = 1.0

        # change the coverage completeness if d2d>=1.6 (i.e., not the Gaia one)
        dfradec.loc[dfradec.d2d >= 1.6, "obsfrac_return0"] = np.nan

        return np.array(dfradec["obsfrac_return0"]) * compGaia
