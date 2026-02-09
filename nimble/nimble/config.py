from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass()
class Config:
    # --- selection / flags ---
    subsample: bool = False
    verbose: bool = True
    use_external_density: bool = False

    # --- magnitude / distance parameters (shared, but values depend on run_type) ---
    # Gmax: float = 20.0
    # Gmin: float = 16.0
    # Grrl: float = 0.58
    # bmin: float = 30.0
    # decmin: float = -35.0
    #
    DMerr: float = 0.24

    # lsr_info: Optional[dict] = None

    # --- spline / knot settings ---
    min_knot: float = 5.0
    max_knot: float = 80.0
    num_knots: int = 4
    min_r: float = 1.0
    max_r: float = 100.0

    # output / data paths
    figs_root: str = "results/"
    figs_path: str = "results/"
    # true_path: Optional[str] = None

    def knots_logr(self):
        return np.linspace(np.log(self.min_knot), np.log(self.max_knot), self.num_knots)

    def to_small(self):
        return SmallConfig(
            knots_logr=self.knots_logr(),
            # lsr_info=self.lsr_info,
            num_knots=self.num_knots,
        )

    def set_knot_range_from_data(self, r, min_percentile=1, max_percentile=99):
        print(f"Original min-max knot: {self.min_knot:.1f}-{self.max_knot:.1f}")
        self.min_knot, self.max_knot = np.percentile(
            r, [min_percentile, max_percentile]
        )
        print(f"New min-max knot: {self.min_knot:.1f}-{self.max_knot:.1f}")


# A super slim data container to feed into fitting functions
@dataclass
class SmallConfig:
    knots_logr: np.ndarray
    num_knots: int
    # lsr_info: Any
    # Grrl: float
    # bupp: float
    # DMerr: float
    # bmin: float
    # blow_rad: float
    # Gmin: float
    # Gmax: float
    # lsym: float
    # lmin_func: Callable
