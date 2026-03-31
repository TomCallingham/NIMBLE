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


def calc_binned_dispersions(stars, r_edges):
    data = {}
    data["r_edges"] = r_edges
    data["r_cents"] = np.sqrt(r_edges[1:] * r_edges[:-1])

    hist = np.histogram(stars["r"], r_edges)[0]
    dens_data = {}
    dens_data["hist"] = hist
    dens_data["bin_width"] = np.diff(r_edges)
    dens_data["density"] = dens_data["hist"] / (
        (4 * np.pi / 3) * ((r_edges[1:] ** 3) - (r_edges[:-1] ** 3))
    )
    data["dens"] = dens_data
    for vkey in ["vr", "vphi", "vtheta"]:
        print(vkey)
        v_data = {}
        v_data["mean_v"] = (
            np.histogram(stars["r"], r_edges, weights=stars[vkey])[0] / hist
        )
        v_data["mean_v2"] = (
            np.histogram(stars["r"], r_edges, weights=(stars[vkey] ** 2))[0] / hist
        )
        v_data["std_v"] = np.sqrt(v_data["mean_v2"] - (v_data["mean_v"] ** 2))
        data[vkey] = v_data
    return data
