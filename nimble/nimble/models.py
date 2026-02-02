import numpy as np
import agama


class DispersionModel:
    def __init__(
        self, params_sigma_r, params_sigma_theta, params_sigma_phi, knots_logr
    ):
        self.params_sigma_r = params_sigma_r
        self.params_sigma_theta = params_sigma_theta
        self.params_sigma_phi = params_sigma_phi
        self.knots_logr = knots_logr
        self.num_knots = len(self.knots_logr)

    @classmethod
    def from_tuple_3d(cls, params_tuple, knots_logr):
        num_knots = len(knots_logr)
        params_sigma_r = params_tuple[:num_knots]
        params_sigma_theta = params_tuple[num_knots : 2 * num_knots]
        params_sigma_phi = params_tuple[2 * num_knots :]
        return cls(params_sigma_r, params_sigma_theta, params_sigma_phi, knots_logr)

    @classmethod
    def from_tuple_2d(cls, params_tuple, knots_logr):
        num_knots = len(knots_logr)
        params_sigma_r = params_tuple[:num_knots]
        params_sigma_thetaphi = params_tuple[num_knots:]
        return cls(
            params_sigma_r, params_sigma_thetaphi, params_sigma_thetaphi, knots_logr
        )

    def sigma_r(self, log_r):
        return np.exp(agama.Spline(self.knots_logr, self.params_sigma_r)(log_r))

    def sigma_theta(self, log_r):
        return np.exp(agama.Spline(self.knots_logr, self.params_sigma_theta)(log_r))

    def sigma_phi(self, log_r):
        return np.exp(agama.Spline(self.knots_logr, self.params_sigma_phi)(log_r))

    def sigma_space(self, log_r):
        sigma_dict = {
            "vr": self.sigma_r(log_r),
            "vtheta": self.sigma_theta(log_r),
            "vphi": self.sigma_phi(log_r),
        }
        return sigma_dict

    def dln_sigma_r(self, log_r):
        return agama.Spline(self.knots_logr, self.params_sigma_r)(log_r, der=1)
