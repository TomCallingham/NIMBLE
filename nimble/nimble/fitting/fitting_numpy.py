import numpy as np
from scipy import optimize
from ..models import DispersionModel


def fit_scipy_3d_dispersion(true_data, scfg, neg_loglike, vdisp_min=20, vdisp_max=400):
    x0 = np.full(3 * scfg.num_knots, 5)
    bounds = [(np.log(vdisp_min), np.log(vdisp_max))] * scfg.num_knots * 3
    res_3d = optimize.minimize(
        neg_loglike,
        x0,
        args=(true_data, scfg),
        bounds=bounds,
        method="L-BFGS-B",
    )
    disp3d = DispersionModel.from_tuple_3d(res_3d.x, scfg.knots_logr)
    return disp3d, res_3d
