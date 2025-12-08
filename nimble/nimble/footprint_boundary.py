import agama
import numpy as np
import scipy.optimize
import scipy.special


d2r = np.pi / 180  # conversion from degrees to radians


def getSurveyFootprintBoundary(decmin):
    # determine the range of l as a function of b, resulting from a cut in declination dec>=decmin
    # (note: the approach is tuned specifically to a constant boundary in declination,
    # but should work for any value of decmin from -90 to +90)
    if decmin == -90:  # no restrictions
        return -np.pi / 2, np.pi / 2, lambda b: b * 0, np.pi

    # 1. obtain the l,b coords of the two poles (south and north) in ICRS
    lp, bp = agama.transformCelestialCoords(
        agama.fromICRStoGalactic, [0, 0], [-np.pi / 2, np.pi / 2]
    )
    # if decmin < decp[0], there is only one loop around the south ICRS pole,
    # likewise if decmin > decp[1], there is only one loop around the north ICRS pole,
    # otherwise there is a boundary spanning the entire range of l

    # 2. find latitude at which boundary crosses the meridional lines in l,b containing the poles
    def rootfinder(b, l0):
        return (
            agama.transformCelestialCoords(agama.fromGalactictoICRS, l0, b)[1]
            - decmin * d2r
        )

    if decmin * d2r < bp[0]:
        b1 = float(scipy.optimize.brentq(rootfinder, bp[0], -np.pi / 2, args=(lp[0],)))
    else:
        b1 = float(scipy.optimize.brentq(rootfinder, bp[1], -np.pi / 2, args=(lp[1],)))
    if decmin * d2r > bp[1]:
        b2 = float(scipy.optimize.brentq(rootfinder, bp[1], +np.pi / 2, args=(lp[1],)))
    else:
        b2 = float(scipy.optimize.brentq(rootfinder, bp[0], +np.pi / 2, args=(lp[0],)))

    # 3. construct a boundary - minimum l for each b between b1 and b2
    npoints = 201
    bb = np.linspace(0, 1, npoints)
    bb = bb * bb * (3 - 2 * bb) * (b2 - b1) + b1
    ll = np.zeros(npoints)
    ll[0] = lp[0] if decmin * d2r < bp[0] else lp[1]
    ll[-1] = lp[1] if decmin * d2r > bp[1] else lp[0]
    for i in range(1, npoints - 1):
        ll[i] = float(
            scipy.optimize.brentq(
                lambda lat: agama.transformCelestialCoords(
                    agama.fromGalactictoICRS, lat, bb[i]
                )[1]
                - decmin * d2r,
                lp[0],
                lp[1],
            )
        )
    curve = agama.CubicSpline(bb, ll)

    # 4. return a tuple of four elements:
    #   lower and upper limit on b,
    #   a function evaluating lmin for the given b,
    #   and lsym such that lmax(b) = 2*lsym - lmin(b).
    # note all coords here in radians, and the range of l is from lsym-pi to lsym+pi (or smaller),
    # not from 0 to 2pi or -pi to pi; thus selection boundary doesn't enclose the actual coordinates
    # of points unless these are shifted to the same angular range, but this doesn't matter for
    # computing the normalization factor (integral of density over the selection region)
    if decmin * d2r < bp[0]:
        # excluded region inside a closed loop around South pole; b anywhere between -pi/2 and pi/2
        return -np.pi / 2, np.pi / 2, lambda b: curve(b, ext=lp[0]), lp[1]
    elif decmin * d2r <= bp[1]:
        # excluded a region below a curve spanning the entire range of l; b must be >= b1
        return b1, np.pi / 2, lambda b: curve(b, ext=lp[0]), lp[1]
    else:  # excluded a region outside a closed loop around North pole; b must be between b1 and b2
        return b1, b2, lambda b: curve(b, ext=lp[1]), lp[1]
