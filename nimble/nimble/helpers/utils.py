from astropy.table import QTable
import numpy as np
import h5py


def load_cols_from_fits(fname, hdu, cols=None):
    tab = QTable.read(fname, hdu=hdu, format="fits")
    keys = cols if cols else list(tab.keys())

    data = {key: tab[key].value for key in keys}
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
