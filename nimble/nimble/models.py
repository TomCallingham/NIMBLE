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


class DispersionMultiModel3D:  # For sampler
    def __init__(
        self, params_sigma_r, params_sigma_theta, params_sigma_phi, knots_logr
    ):
        self.params_sigma_r = params_sigma_r
        self.params_sigma_theta = params_sigma_theta
        self.params_sigma_phi = params_sigma_phi
        self.knots_logr = knots_logr
        self.num_knots = len(self.knots_logr)
        self.n_sample = params_sigma_r.shape[0]

    @classmethod
    def from_sampler_3d(cls, params_sample, knots_logr):
        num_knots = len(knots_logr)
        params_sigma_r = params_sample[:, :num_knots]
        params_sigma_theta = params_sample[:, num_knots : 2 * num_knots]
        params_sigma_phi = params_sample[:, 2 * num_knots :]
        return cls(params_sigma_r, params_sigma_theta, params_sigma_phi, knots_logr)

    @classmethod
    def from_sampler_2d(cls, params_sample, knots_logr):
        num_knots = len(knots_logr)
        params_sigma_r = params_sample[:, :num_knots]
        params_sigma_thetaphi = params_sample[:, num_knots:]
        return cls(
            params_sigma_r, params_sigma_thetaphi, params_sigma_thetaphi, knots_logr
        )

    def sigma_r(self, log_r):
        sigma_r = np.empty((self.n_sample, len(log_r)))
        for i in range(self.n_sample):
            sigma_r[i, :] = np.exp(
                agama.Spline(self.knots_logr, self.params_sigma_r[i, :])(log_r)
            )
        return sigma_r

    def sigma_theta(self, log_r):
        sigma_theta = np.empty((self.n_sample, len(log_r)))
        for i in range(self.n_sample):
            sigma_theta[i, :] = np.exp(
                agama.Spline(self.knots_logr, self.params_sigma_theta[i, :])(log_r)
            )
        return sigma_theta

    def sigma_phi(self, log_r):
        assert len(log_r.shape) == 1
        sigma_phi = np.empty((self.n_sample, len(log_r)))
        for i in range(self.n_sample):
            sigma_phi[i, :] = np.exp(
                agama.Spline(self.knots_logr, self.params_sigma_phi[i, :])(log_r)
            )
        return sigma_phi

    def sigma_space(self, log_r):
        sigma_dict = {
            "vr": self.sigma_r(log_r),
            "vtheta": self.sigma_theta(log_r),
            "vphi": self.sigma_phi(log_r),
        }
        return sigma_dict

    def sigma_space_stats(self, log_r):
        sigma_dict = self.sigma_space(log_r)
        sigma_stats = {}
        for key, x in sigma_dict.items():
            med, o1s, u1s, o2s, u2s = np.percentile(x, [50, 84, 16, 95, 5], axis=0)
            stats = {"med": med, "o1s": o1s, "u1s": u1s, "o2s": o2s, "u2s": u2s}
            sigma_stats[key] = stats

        return sigma_stats

    def dln_sigma_r(self, log_r):
        dln_sigma = np.empty((self.n_sample, len(log_r)))
        for i in range(self.n_sample):
            dln_sigma[i, :] = agama.Spline(self.knots_logr, self.params_sigma_r[i, :])(
                log_r, der=1
            )
        return dln_sigma
