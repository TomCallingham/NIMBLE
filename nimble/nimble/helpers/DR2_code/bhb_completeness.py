import astropy.io.fits as pyfits
import healpy
import numpy as np


class Completeness:

    def __init__(self, fname):
        self.fname = fname
        fp = pyfits.open(fname)
        self.nside = fp[0].header['NSIDE']
        self.data_hpx = fp[1].data['healpix'][:]
        self.data_completeness = fp[1].data['completeness'][:]
        fp.close()
        del fp

    def __call__(self, ra, dec):
        ra, dec = np.asarray(ra), np.asarray(dec)
        hpx = healpy.ang2pix(self.nside, ra, dec, lonlat=True, nest=True)
        data_hpx = self.data_hpx
        data_completeness = self.data_completeness
        xind = np.searchsorted(data_hpx, hpx)
        xind = np.minimum(xind, len(data_hpx))
        ret = data_completeness[xind] * (data_hpx[xind] == hpx)
        return ret


def get_completness_function(fname):
    return Completeness(fname)
