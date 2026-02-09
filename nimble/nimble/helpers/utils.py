from astropy.table import QTable
import numpy as np
import h5py
from scipy.spatial import KDTree


def load_cols_from_fits(fname, hdu, cols=None):
    tab = QTable.read(fname, hdu=hdu, format="fits")
    keys = cols if cols else list(tab.keys())
    data = {}
    for key in keys:
        x = tab[key].value
        dtype = str(x.dtype)
        if "f" in dtype:
            x = x.astype(float)
        elif "i" in dtype:
            x = x.astype(int)
        data[key] = x
    del tab
    return data


def robust_table_to_numpy(tab, cols=None):
    keys = cols if cols else list(tab.keys())
    data = {}
    for key in keys:
        x = tab[key].value
        dtype = str(x.dtype)
        if "f" in dtype:
            x = x.astype(float)
        elif "i" in dtype:
            x = x.astype(int)
        data[key] = x
    return data


def h5py_load(fname, subgroup=None):
    """Loads hdf5 files structures into Nested Dictionaries"""
    max_nest = 10

    def load(hf, n_nest=0):
        if n_nest > max_nest:
            print("Error, max nesting exceeded!")
            raise SystemError
        n_nest += 1
        data = {}
        for key in hf:
            try:
                dic_key = int(key)
            except Exception:
                dic_key = key
            try:
                if isinstance(hf[key], h5py._hl.group.Group):
                    data[dic_key] = load(hf[key], n_nest)
                else:
                    data[dic_key] = np.asarray(hf[key])
                    if "S" in str(data[dic_key].dtype):
                        data[dic_key] = data[dic_key].astype("<U20")
            except AttributeError:
                print(f"Unidentified structure {key}")
                # raise SystemError'Attribute errors'
        return data

    with h5py.File(fname, "r") as hf:
        if subgroup is None:
            data = load(hf)
        else:
            data = load(hf[subgroup])
    return data


def _radec_to_unitvec(ra_deg, dec_deg):
    """RA/Dec (deg) -> unit vectors on the sphere (x,y,z)."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    cosd = np.cos(dec)
    x = cosd * np.cos(ra)
    y = cosd * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack((x, y, z))


def crossmatch_radec_ckdtree(
    ra_a_deg,
    dec_a_deg,
    ra_b_deg,
    dec_b_deg,
    *,
    max_sep_arcsec=None,
    k=1,
):
    """
    Crossmatch catalog A -> catalog B with KDTree on 3D unit sphere.

    Returns
    -------
    idx_b : (Na,) int
        Index in B of nearest neighbor (or -1 if beyond max_sep_arcsec).
    d_sky_arcsec : (Na,) float
        On-sky separation in arcsec (np.inf for unmatched if max_sep_arcsec is set).
    """
    a_xyz = _radec_to_unitvec(ra_a_deg, dec_a_deg)
    b_xyz = _radec_to_unitvec(ra_b_deg, dec_b_deg)

    tree = KDTree(b_xyz)

    # Query Euclidean chord distance on the unit sphere.
    # If a max angular separation is given, translate to chord distance:
    # chord = 2 * sin(theta/2)
    if max_sep_arcsec is None:
        dist_chord, idx = tree.query(a_xyz, k=k)
    else:
        theta = np.deg2rad(max_sep_arcsec / 3600.0)
        r = 2.0 * np.sin(theta / 2.0)
        dist_chord, idx = tree.query(a_xyz, k=k, distance_upper_bound=r)

    # If k>1, you'd get arrays (Na,k). This keeps k=1 output simple.
    if k != 1:
        raise NotImplementedError(
            "This simple helper returns k=1 only. Ask if you want k>1."
        )

    # Convert chord distance to angular separation:
    # For unit sphere: chord = 2*sin(theta/2)  => theta = 2*asin(chord/2)
    # Clamp for numeric safety.
    dist_chord = np.asarray(dist_chord, dtype=float)
    half = np.clip(dist_chord / 2.0, 0.0, 1.0)
    theta_rad = 2.0 * np.arcsin(half)
    d_sky_arcsec = np.rad2deg(theta_rad) * 3600.0

    idx = np.asarray(idx)
    if max_sep_arcsec is not None:
        # cKDTree uses idx == n for "not found within upper bound"
        not_found = idx >= len(ra_b_deg)
        idx = idx.astype(np.int64)
        idx[not_found] = -1
        d_sky_arcsec = d_sky_arcsec.astype(float)
        d_sky_arcsec[not_found] = np.inf

    return idx, d_sky_arcsec
