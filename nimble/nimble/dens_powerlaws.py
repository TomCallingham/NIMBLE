import numpy as np
import math
from scipy.special import gammaln


class BrokenPowerLaw:
    """
    Spherical broken power-law spatial density rho(r), normalised so that:
        ∫_0^∞ 4π r^2 rho(r) dr = 1

    Parameterisation:
        rho(r) = K * (r/Rb)^{s_in}   for r < Rb
               = K * (r/Rb)^{s_out}  for r >= Rb
    (continuous at r=Rb by construction)

    Constraints for convergence:
        s_in  > -3   (integrable at r -> 0)
        s_out < -3   (integrable at r -> ∞)
    """

    R_break: float
    slope_inner: float  # s_in
    slope_outer: float  # s_out

    def __init__(self, R_break: float, slope_inner: float, slope_outer: float):
        self.R_break = float(R_break)
        self.slope_inner = float(slope_inner)
        self.slope_outer = float(slope_outer)

        if not (self.R_break > 0):
            raise ValueError("R_break must be > 0.")
        if not (self.slope_outer < -3.0):
            raise ValueError(
                "Need slope_outer < -3 for normalisation to converge at infinity."
            )
        if not (self.slope_inner > -3.0):
            raise ValueError(
                "Need slope_inner > -3 for normalisation to converge at r->0."
            )
        self.lnK = self.calc_lnK()

    def calc_lnK(self) -> float:
        """Natural log of the normalisation constant K."""
        Rb = float(self.R_break)
        s_in = float(self.slope_inner)
        s_out = float(self.slope_outer)

        term = (1.0 / (s_in + 3.0)) + (1.0 / (-s_out - 3.0))
        # K = 1 / [4π Rb^3 term]
        return -np.log(4.0 * np.pi) - 3.0 * np.log(Rb) - np.log(term)

    def log_rho(self, r: np.ndarray) -> np.ndarray:
        """ln rho(r) for the hard-broken, unit-normalised profile."""
        out = np.empty_like(r)
        m = r < self.R_break
        out[m] = self.lnK + self.slope_inner * np.log(r[m] / self.R_break)
        out[~m] = self.lnK + self.slope_outer * np.log(r[~m] / self.R_break)
        return out

    def dlnrho(self, r: np.ndarray) -> np.ndarray:
        """d ln rho / d ln r (piecewise constant)."""
        out = np.empty_like(r)
        m = r < self.R_break
        out[m] = self.slope_inner
        out[~m] = self.slope_outer
        return out

    def log_rho_from_logr(self, logr: np.ndarray) -> np.ndarray:
        return self.log_rho(np.exp(logr))

    def dlnrho_from_logr(self, logr: np.ndarray) -> np.ndarray:
        return self.dlnrho(np.exp(logr))


class SmoothDoublePowerLaw:
    """
    Spherical *smoothed* broken power-law spatial density rho(r), normalised so that:
        ∫_0^∞ 4π r^2 rho(r) dr = 1

    Parameterisation (x = r/Rb):
        rho(r) = K * x^{s_in} * (1 + x^{1/delta})^{delta*(s_out - s_in)}

    Asymptotes:
        r << Rb: rho ~ K * (r/Rb)^{s_in}
        r >> Rb: rho ~ K * (r/Rb)^{s_out}

    Constraints for convergence:
        s_in  > -3   (integrable at r -> 0)
        s_out < -3   (integrable at r -> ∞)
        delta > 0
    """

    def __init__(
        self, R_break: float, slope_inner: float, slope_outer: float, delta: float
    ):
        self.R_break = float(R_break)
        self.slope_inner = float(slope_inner)
        self.slope_outer = float(slope_outer)
        self.delta = float(delta)

        if not (self.R_break > 0):
            raise ValueError("R_break must be > 0.")
        if not (self.delta > 0):
            raise ValueError("delta must be > 0.")
        if not (self.slope_outer < -3.0):
            raise ValueError(
                "Need slope_outer < -3 for normalisation to converge at infinity."
            )
        if not (self.slope_inner > -3.0):
            raise ValueError(
                "Need slope_inner > -3 for normalisation to converge at r->0."
            )
        self.lnK = self.calc_lnK()

    def calc_lnK(self) -> float:
        """
        Natural log of the normalisation constant K.

        K = 1 / [4π Rb^3 δ * Beta( δ(s_in+3), δ(-s_out-3) )]
        """
        Rb = self.R_break
        s_in = self.slope_inner
        s_out = self.slope_outer
        d = self.delta

        a = d * (s_in + 3.0)
        b = d * (-s_out - 3.0)
        # ln Beta(a,b) = ln Γ(a) + ln Γ(b) - ln Γ(a+b)
        ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

        return -math.log(4.0 * math.pi) - 3.0 * math.log(Rb) - math.log(d) - ln_beta

    def log_rho(self, r: np.ndarray) -> np.ndarray:
        """ln rho(r) for the smoothed, unit-normalised profile."""
        r = np.asarray(r, dtype=float)
        if np.any(r <= 0):
            raise ValueError("All radii must be > 0 (log undefined at r<=0).")

        x = r / self.R_break
        u = x ** (1.0 / self.delta)

        # ln rho = lnK + s_in ln x + delta (s_out - s_in) ln(1 + u)
        return (
            self.lnK
            + self.slope_inner * np.log(x)
            + self.delta * (self.slope_outer - self.slope_inner) * np.log1p(u)
        )

    def dlnrho(self, r: np.ndarray) -> np.ndarray:
        """d ln rho / d ln r for the smoothed profile."""
        r = np.asarray(r, dtype=float)
        if np.any(r <= 0):
            raise ValueError("All radii must be > 0.")

        x = r / self.R_break
        u = x ** (1.0 / self.delta)

        # slope = s_in + (s_out - s_in) * u/(1+u)
        return self.slope_inner + (self.slope_outer - self.slope_inner) * (
            u / (1.0 + u)
        )

    def log_rho_from_logr(self, logr: np.ndarray) -> np.ndarray:
        return self.log_rho(np.exp(logr))

    def dlnrho_from_logr(self, logr: np.ndarray) -> np.ndarray:
        return self.dlnrho(np.exp(logr))


def _split_normal(rng, mu, sigma_minus, sigma_plus, size):
    """Two-piece (split) normal."""
    z = rng.standard_normal(size=size)
    sigma = np.where(z < 0.0, sigma_minus, sigma_plus)
    return mu + sigma * z


def _broadcast_param(p, r_ndim):
    """(n_sample,) -> (n_sample, 1, 1, ...) for broadcasting over r."""
    return p[(slice(None),) + (None,) * r_ndim]


class MultiBrokenPowerLaw:
    """
    Random (MC) hard-broken power-law spatial density rho(r), normalised so that:
        ∫_0^∞ 4π r^2 rho(r) dr = 1

    Parameterisation (x=r/Rb):
        rho(r) = K * x^{s_in}   for r < Rb
               = K * x^{s_out}  for r >= Rb

    Draws are created once at init (split-normal with asymmetric errors).
    """

    def __init__(
        self,
        *,
        n_sample: int,
        seed: int = 42,
        R_break: float,
        R_break_err: tuple[float, float],
        slope_inner: float,
        slope_inner_err: tuple[float, float],
        slope_outer: float,
        slope_outer_err: tuple[float, float],
    ):
        self.n_sample = int(n_sample)

        rng = np.random.default_rng(int(seed))

        Rb = _split_normal(
            rng, R_break, R_break_err[0], R_break_err[1], size=self.n_sample
        )
        si = _split_normal(
            rng, slope_inner, slope_inner_err[0], slope_inner_err[1], size=self.n_sample
        )
        so = _split_normal(
            rng, slope_outer, slope_outer_err[0], slope_outer_err[1], size=self.n_sample
        )

        Rb_min = 1
        Rb_max = 100
        Rb = np.clip(Rb, Rb_min, Rb_max)
        so[so >= -3] = -3.001
        si[si > -1] = -1
        si[si < so] = so[si < so]

        self.R_break = Rb
        self.slope_inner = si
        self.slope_outer = so

        self.lnK = self.calc_lnK()

    def calc_lnK(self):
        # Vectorised analytic lnK per draw:
        # K = 1 / [4π Rb^3 * ( 1/(s_in+3) + 1/(-s_out-3) )]
        term = (1.0 / (self.slope_inner + 3.0)) + (1.0 / (-self.slope_outer - 3.0))
        return -np.log(4.0 * np.pi) - 3.0 * np.log(self.R_break) - np.log(term)

    def log_rho(self, r: np.ndarray) -> np.ndarray:
        """
        ln rho(r) for each draw.
        Output shape: (n_sample, *r.shape)
        """
        r_b = r[None, ...]
        Rb = _broadcast_param(self.R_break, r.ndim)
        lnK = _broadcast_param(self.lnK, r.ndim)
        si = _broadcast_param(self.slope_inner, r.ndim)
        so = _broadcast_param(self.slope_outer, r.ndim)

        x = r_b / Rb
        ln_in = lnK + si * np.log(x)
        ln_out = lnK + so * np.log(x)
        return np.where(r_b < Rb, ln_in, ln_out)

    def dlnrho(self, r: np.ndarray) -> np.ndarray:
        """
        d ln rho / d ln r for each draw (piecewise constant).
        Output shape: (n_sample, *r.shape)
        """
        r_b = r[None, ...]
        Rb = _broadcast_param(self.R_break, r.ndim)
        si = _broadcast_param(self.slope_inner, r.ndim)
        so = _broadcast_param(self.slope_outer, r.ndim)
        return np.where(r_b < Rb, si, so)

    def stats_log_rho(self, r: np.ndarray):
        return calc_stats(self.log_rho(r), axis=0)

    def stats_dlnrho(self, r: np.ndarray):
        return calc_stats(self.dlnrho(r), axis=0)

    def log_rho_from_logr(self, logr: np.ndarray) -> np.ndarray:
        return self.log_rho(np.exp(logr))

    def dlnrho_from_logr(self, logr: np.ndarray) -> np.ndarray:
        return self.dlnrho(np.exp(logr))


def calc_stats(X, axis=0):
    med, o1s, u1s, o2s, u2s = np.percentile(X, [50, 84, 16, 95, 5], axis=axis)
    stats = {"med": med, "o1s": o1s, "u1s": u1s, "o2s": o2s, "u2s": u2s}
    return stats


class MultiSmoothDoublePowerLaw:
    """
    Random (MC) hard-broken power-law spatial density rho(r), normalised so that:
        ∫_0^∞ 4π r^2 rho(r) dr = 1

    Parameterisation (x=r/Rb):
        rho(r) = K * x^{s_in}   for r < Rb
               = K * x^{s_out}  for r >= Rb

    Draws are created once at init (split-normal with asymmetric errors).
    """

    def __init__(
        self,
        *,
        n_sample: int,
        seed: int = 42,
        R_break: float,
        R_break_err: tuple[float, float],
        slope_inner: float,
        slope_inner_err: tuple[float, float],
        slope_outer: float,
        slope_outer_err: tuple[float, float],
        delta=0.1,
    ):
        self.n_sample = int(n_sample)
        self.delta = delta

        rng = np.random.default_rng(int(seed))

        Rb = _split_normal(
            rng, R_break, R_break_err[0], R_break_err[1], size=self.n_sample
        )
        si = _split_normal(
            rng, slope_inner, slope_inner_err[0], slope_inner_err[1], size=self.n_sample
        )
        so = _split_normal(
            rng, slope_outer, slope_outer_err[0], slope_outer_err[1], size=self.n_sample
        )

        Rb_min = 1
        Rb_max = 100
        Rb = np.clip(Rb, Rb_min, Rb_max)
        so[so >= -3] = -3.001
        si[si > -1] = -1
        si[si < so] = so[si < so]

        self.R_break = Rb
        self.slope_inner = si
        self.slope_outer = so

        self.lnK = self.calc_lnK()

    def calc_lnK(self) -> np.ndarray:
        """
        Natural log of the normalisation constant K.

        K = 1 / [4π Rb^3 δ * Beta( δ(s_in+3), δ(-s_out-3) )]
        """
        Rb = self.R_break
        s_in = self.slope_inner
        s_out = self.slope_outer
        d = self.delta

        a = d * (s_in + 3.0)
        b = d * (-s_out - 3.0)
        # ln Beta(a,b) = ln Γ(a) + ln Γ(b) - ln Γ(a+b)
        ln_beta = gammaln(a) + gammaln(b) - gammaln(a + b)

        return -math.log(4.0 * math.pi) - 3.0 * np.log(Rb) - np.log(d) - ln_beta

    def log_rho(self, r: np.ndarray) -> np.ndarray:
        """ln rho(r) for the smoothed, unit-normalised profile."""
        r = np.asarray(r, dtype=float)
        if np.any(r <= 0):
            raise ValueError("All radii must be > 0 (log undefined at r<=0).")

        x = r[None, :] / self.R_break[:, None]
        u = x ** (1.0 / self.delta)

        # ln rho = lnK + s_in ln x + delta (s_out - s_in) ln(1 + u)
        return (
            self.lnK[:, None]
            + self.slope_inner[:, None] * np.log(x)
            + self.delta * (self.slope_outer - self.slope_inner)[:, None] * np.log1p(u)
        )

    def dlnrho(self, r: np.ndarray) -> np.ndarray:
        """d ln rho / d ln r for the smoothed profile."""
        r = np.asarray(r, dtype=float)
        if np.any(r <= 0):
            raise ValueError("All radii must be > 0.")

        x = r[None, :] / self.R_break[:, None]
        u = x ** (1.0 / self.delta)

        # slope = s_in + (s_out - s_in) * u/(1+u)
        return self.slope_inner[:, None] + (self.slope_outer - self.slope_inner)[
            :, None
        ] * (u / (1.0 + u))

    def log_rho_from_logr(self, logr: np.ndarray) -> np.ndarray:
        return self.log_rho(np.exp(logr))

    def dlnrho_from_logr(self, logr: np.ndarray) -> np.ndarray:
        return self.dlnrho(np.exp(logr))

    def stats_log_rho(self, r: np.ndarray):
        return calc_stats(self.log_rho(r), axis=0)

    def stats_dlnrho(self, r: np.ndarray):
        return calc_stats(self.dlnrho(r), axis=0)


M24_R_break = 18.1
M24_slope_inner = -2.05
M24_slope_outer = -4.47
M24_delta = 0.1
M24_R_break_err = (1.1, 2.1)
M24_slope_inner_err = (0.15, 0.13)
M24_slope_outer_err = (0.18, 0.11)


def create_Medina24RRL_SmoothPowerLaw():
    smooth_Medina24RRL = SmoothDoublePowerLaw(
        R_break=M24_R_break,
        slope_inner=M24_slope_inner,
        slope_outer=M24_slope_outer,
        delta=M24_delta,
    )
    return smooth_Medina24RRL


def create_Medina24RRL_MultiSmoothPowerLaw(n_sample=1200):
    multismooth_Medina24RRL = MultiSmoothDoublePowerLaw(
        R_break=M24_R_break,
        slope_inner=M24_slope_inner,
        slope_outer=M24_slope_outer,
        delta=M24_delta,
        n_sample=n_sample,
        R_break_err=M24_R_break_err,
        slope_inner_err=M24_slope_inner_err,
        slope_outer_err=M24_slope_outer_err,
    )
    return multismooth_Medina24RRL


A24_R_break = 19.15
# A24_slope_inner=-2.45
A24_slope_inner = -2.9
A24_slope_outer = -4.55
A24_delta = 0.1
A24_R_break_err = (1.8, 1.7)
A24_slope_inner_err = (0.14, 0.17)
A24_slope_outer_err = (0.1, 0.11)


def create_Amarante24BHB_SmoothPowerLaw():
    smooth_Amarante24BHB = SmoothDoublePowerLaw(
        R_break=A24_R_break,
        slope_inner=A24_slope_inner,
        slope_outer=A24_slope_outer,
        delta=A24_delta,
    )
    return smooth_Amarante24BHB


def create_Amarante24BHB_MultiSmoothPowerLaw(n_sample=1200):
    multismooth_Amarante24BHB = MultiSmoothDoublePowerLaw(
        R_break=A24_R_break,
        slope_inner=A24_slope_inner,
        slope_outer=A24_slope_outer,
        delta=A24_delta,
        n_sample=n_sample,
        R_break_err=A24_R_break_err,
        slope_inner_err=A24_slope_inner_err,
        slope_outer_err=A24_slope_outer_err,
    )
    return multismooth_Amarante24BHB
