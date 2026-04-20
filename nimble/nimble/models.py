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


class DispersionMeanModel:
    def __init__(
        self,
        params_sigma_r,
        params_sigma_theta,
        params_sigma_phi,
        params_mean_r,
        params_mean_theta,
        params_mean_phi,
        knots_logr,
    ):
        self.params_sigma_r = params_sigma_r
        self.params_sigma_theta = params_sigma_theta
        self.params_sigma_phi = params_sigma_phi
        self.params_mean_r = params_mean_r
        self.params_mean_theta = params_mean_theta
        self.params_mean_phi = params_mean_phi

        self.knots_logr = knots_logr
        self.num_knots = len(self.knots_logr)

    @classmethod
    def from_tuple_3d(cls, params_tuple, knots_logr):
        num_knots = len(knots_logr)
        params_sigma_r = params_tuple[:num_knots]
        params_sigma_theta = params_tuple[num_knots : 2 * num_knots]
        params_sigma_phi = params_tuple[2 * num_knots : 3 * num_knots]
        params_mean_r = params_tuple[3 * num_knots : 4 * num_knots]
        params_mean_theta = params_tuple[4 * num_knots : 5 * num_knots]
        params_mean_phi = params_tuple[5 * num_knots :]
        return cls(
            params_sigma_r,
            params_sigma_theta,
            params_sigma_phi,
            params_mean_r,
            params_mean_theta,
            params_mean_phi,
            knots_logr,
        )

    def sigma_r(self, log_r):
        return np.exp(agama.Spline(self.knots_logr, self.params_sigma_r)(log_r))

    def sigma_theta(self, log_r):
        return np.exp(agama.Spline(self.knots_logr, self.params_sigma_theta)(log_r))

    def sigma_phi(self, log_r):
        return np.exp(agama.Spline(self.knots_logr, self.params_sigma_phi)(log_r))

    def mean_r(self, log_r):
        return agama.Spline(self.knots_logr, self.params_mean_r)(log_r)

    def mean_theta(self, log_r):
        return agama.Spline(self.knots_logr, self.params_mean_theta)(log_r)

    def mean_phi(self, log_r):
        return agama.Spline(self.knots_logr, self.params_mean_phi)(log_r)

    def sigma_space(self, log_r):
        sigma_dict = {
            "vr": self.sigma_r(log_r),
            "vtheta": self.sigma_theta(log_r),
            "vphi": self.sigma_phi(log_r),
        }
        return sigma_dict

    def mean_space(self, log_r):
        mean_dict = {
            "vr": self.mean_r(log_r),
            "vtheta": self.mean_theta(log_r),
            "vphi": self.mean_phi(log_r),
        }
        return mean_dict

    def dln_sigma_r(self, log_r):
        return agama.Spline(self.knots_logr, self.params_sigma_r)(log_r, der=1)


class DispersionMeanMultiModel3D:
    def __init__(
        self,
        params_sigma_r,
        params_sigma_theta,
        params_sigma_phi,
        params_mean_r,
        params_mean_theta,
        params_mean_phi,
        knots_logr,
    ):
        self.params_sigma_r = params_sigma_r
        self.params_sigma_theta = params_sigma_theta
        self.params_sigma_phi = params_sigma_phi
        self.params_mean_r = params_mean_r
        self.params_mean_theta = params_mean_theta
        self.params_mean_phi = params_mean_phi
        self.knots_logr = knots_logr
        self.num_knots = len(self.knots_logr)
        self.n_sample = params_sigma_r.shape[0]

    @classmethod
    def from_sampler_3d(cls, sigma_samples, mean_samples, knots_logr):
        num_knots = len(knots_logr)
        params_sigma_r = sigma_samples[:, :num_knots]
        params_sigma_theta = sigma_samples[:, num_knots : 2 * num_knots]
        params_sigma_phi = sigma_samples[:, 2 * num_knots :]
        params_mean_r = mean_samples[:, :num_knots]
        params_mean_theta = mean_samples[:, num_knots : 2 * num_knots]
        params_mean_phi = mean_samples[:, 2 * num_knots :]
        return cls(
            params_sigma_r,
            params_sigma_theta,
            params_sigma_phi,
            params_mean_r,
            params_mean_theta,
            params_mean_phi,
            knots_logr,
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

    def mean_r(self, log_r):
        mean_theta = np.empty((self.n_sample, len(log_r)))
        for i in range(self.n_sample):
            mean_theta[i, :] = agama.Spline(self.knots_logr, self.params_mean_r[i, :])(
                log_r
            )
        return mean_theta

    def mean_theta(self, log_r):
        mean_theta = np.empty((self.n_sample, len(log_r)))
        for i in range(self.n_sample):
            mean_theta[i, :] = agama.Spline(
                self.knots_logr, self.params_mean_theta[i, :]
            )(log_r)
        return mean_theta

    def mean_phi(self, log_r):
        assert len(log_r.shape) == 1
        mean_phi = np.empty((self.n_sample, len(log_r)))
        for i in range(self.n_sample):
            mean_phi[i, :] = agama.Spline(self.knots_logr, self.params_mean_phi[i, :])(
                log_r
            )
        return mean_phi

    def sigma_mean_space(self, log_r):
        sigma_mean_dict = {
            "s_vr": self.sigma_r(log_r),
            "s_vtheta": self.sigma_theta(log_r),
            "s_vphi": self.sigma_phi(log_r),
            "m_vr": self.mean_r(log_r),
            "m_vtheta": self.mean_theta(log_r),
            "m_vphi": self.mean_phi(log_r),
        }
        return sigma_mean_dict

    def sigma_mean_space_stats(self, log_r):
        sigma_mean_dict = self.sigma_mean_space(log_r)
        sigma_mean_stats = {}
        for key, x in sigma_mean_dict.items():
            med, o1s, u1s, o2s, u2s = np.percentile(x, [50, 84, 16, 95, 5], axis=0)
            stats = {"med": med, "o1s": o1s, "u1s": u1s, "o2s": o2s, "u2s": u2s}
            sigma_mean_stats[key] = stats

        return sigma_mean_stats

        # "s_vtheta": self.sigma_theta(log_r),
        # "s_vphi": self.sigma_phi(log_r),
        # "m_vtheta": self.mean_theta(log_r),
        # "m_vphi": self.mean_phi(log_r),

    def v2_r(self, log_r):
        return (self.sigma_r(log_r) ** 2) + (self.mean_r(log_r) ** 2)

    def v2_theta(self, log_r):
        return (self.sigma_theta(log_r) ** 2) + (self.mean_theta(log_r) ** 2)

    def v2_phi(self, log_r):
        return (self.sigma_phi(log_r) ** 2) + (self.mean_phi(log_r) ** 2)

    def v2_space(self, log_r):
        v2_dict = {
            "vr": self.v2_r(log_r),
            "vtheta": self.v2_theta(log_r),
            "vphi": self.v2_phi(log_r),
        }
        return v2_dict

    def v2_space_stats(self, log_r):
        v2_dict = self.v2_space(log_r)
        v2_stats = {}
        for key, x in v2_dict.items():
            med, o1s, u1s, o2s, u2s = np.percentile(x, [50, 84, 16, 95, 5], axis=0)
            stats = {"med": med, "o1s": o1s, "u1s": u1s, "o2s": o2s, "u2s": u2s}
            v2_stats[key] = stats

        return v2_stats

    def dln_sigma_r(self, log_r):
        dln_sigma = np.empty((self.n_sample, len(log_r)))
        for i in range(self.n_sample):
            dln_sigma[i, :] = agama.Spline(self.knots_logr, self.params_sigma_r[i, :])(
                log_r, der=1
            )
        return dln_sigma

    def dln_v2_r(self, log_r):
        """
        d ln v2_r / d ln r  where  v2_r = sigma_r^2 + mean_r^2

        Uses chain rule:
            d ln v2_r / d ln r = (2 sigma_r^2 * dln_sigma_r + 2 mean_r * dmean_r_dlnr)
                                / (sigma_r^2 + mean_r^2)
        """
        dln_v2 = np.empty((self.n_sample, len(log_r)))
        for i in range(self.n_sample):
            sp_sigma = agama.Spline(self.knots_logr, self.params_sigma_r[i, :])
            sp_mean = agama.Spline(self.knots_logr, self.params_mean_r[i, :])

            ln_sigma_r = sp_sigma(log_r)  # log sigma_r
            sigma_r = np.exp(ln_sigma_r)
            dln_sigma_r = sp_sigma(log_r, der=1)  # d ln sigma_r / d ln r

            mean_r = sp_mean(log_r)  # mean_r (not log)
            dmean_r_dlnr = sp_mean(log_r, der=1)  # d mean_r / d ln r

            v2_r = sigma_r**2 + mean_r**2

            dln_v2[i, :] = (
                2.0 * sigma_r**2 * dln_sigma_r + 2.0 * mean_r * dmean_r_dlnr
            ) / v2_r

        return dln_v2
