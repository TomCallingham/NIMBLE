from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass()
class Config:
    # # --- selection / flags ---
    # subsample: bool = False
    # verbose: bool = True

    # Spline/Knot. Will be fit by the data
    min_knot: float = 5.0
    max_knot: float = 80.0
    num_knots: int = 4
    # min_r: float = 10.0
    # max_r: float = 100.0

    # output / data paths
    figs_root: str = "results/"
    figs_path: str = "results/"
    # true_path: Optional[str] = None
    #
    n_per_bin: int = 1000
    n_min_bound: int = 200
    n_dist_sample: int = 1000

    def knots_logr(self):
        return self._knots_logr

    def to_small(self):
        return SmallConfig(
            knots_logr=self.knots_logr(),
            # lsr_info=self.lsr_info,
            num_knots=self.num_knots,
            min_knot=self.min_knot,
            max_knot=self.max_knot,
        )

    def set_quantile_knots(self, r):
        print(f"Creating Knots! n per m{self.n_per_bin}, n_min: {self.n_min_bound}")
        r = np.sort(np.asarray(r, dtype=float))
        r_trimmed = r[self.n_min_bound : -self.n_min_bound]

        n_intervals = len(r_trimmed) // self.n_per_bin
        quantiles = np.linspace(0, 100, n_intervals + 1)
        knots = np.percentile(r_trimmed, quantiles)
        self._knots_logr = np.log(knots)
        self.min_knot = knots[0]
        self.max_knot = knots[-1]
        self.num_knots = len(self._knots_logr)
        print(
            f"Created {self.num_knots} Knots"
            f"Min: {self.min_knot:.2f} - Max: {self.max_knot:.2f} "
        )

    def set_quantile_knots_with_gaurd(self, r, r_min=5, r_max=100):
        print(
            f"Creating Knots WITH GAURD. n per m{self.n_per_bin}, n_min: {self.n_min_bound}"
        )
        r = np.sort(np.asarray(r, dtype=float))
        r_trimmed = r[self.n_min_bound : -self.n_min_bound]

        n_intervals = len(r_trimmed) // self.n_per_bin
        quantiles = np.linspace(0, 100, n_intervals + 1)
        per_knots = np.percentile(r_trimmed, quantiles)

        knots = np.zeros((len(per_knots) + 2), dtype=float)
        knots[1:-1] = per_knots
        knots[0] = r_min
        knots[-1] = r_max

        self._knots_logr = np.log(knots)
        self.min_knot = knots[1]
        self.max_knot = knots[-2]
        self.num_knots = len(self._knots_logr)
        print(
            f"Created {self.num_knots} Knots"
            f"Min: {self.min_knot:.2f} - Max: {self.max_knot:.2f} "
        )


# A super slim data container to feed into fitting functions
@dataclass
class SmallConfig:
    knots_logr: np.ndarray
    num_knots: int  # THIS could be found do I JUST NEED knots
    min_knot: float
    max_knot: float
