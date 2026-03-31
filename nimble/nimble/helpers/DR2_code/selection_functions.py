from . import bhb_completeness
from . import compfunc_rrl


def get_p_obs(ra, dec, tracer, desi_release, gaia_release="dr2", mag=None):
    if tracer == "bhb":
        # this is funky but jura == loa for these purposes
        release = "jura" if desi_release == "loa" else desi_release
        path = "/global/cfs/cdirs/desi/science/mws/y3kp/selection-functions/bhb/"
        table_name = path + f"bhb_completeness_table_{release}_main_bright.fits"
        bhbfunc = bhb_completeness.get_completness_function(table_name)
        p_obs = bhbfunc(ra, dec)
    elif tracer == "rrl":
        kwargs = {
            "primsec": "prim",
            "surv": "main",
            "prog": "bright",
            "subset": desi_release,
            "Gaia": True,
            "GaiaDR": gaia_release.upper(),
            "typ": "ab",
            "wdist": False,
            "dist": -99,
        }

        p_obs = compfunc_rrl.ObsComp(ra_in=ra, dec_in=dec, mag=mag, **kwargs)
    else:
        raise KeyError("bhb or rrl")

    return p_obs
