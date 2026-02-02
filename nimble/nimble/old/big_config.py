from dataclasses import dataclass, fields
from typing import Any, Callable, Optional

import numpy as np

from .helpers.auridesi.auridesi_params import AuriDesiParams

d2r = np.pi / 180  # conversion from degrees to radians


run_types = ["auridesi"]  # , "iron"]
# RunTypeParams = Union[AuriDesiParams, IronParams]
RunTypeParams = AuriDesiParams

run_type_dispatch_dict = {
    "auridesi": AuriDesiParams,
    # "desi": DesiParams,
}


# ---------------------------------------------------------------------
# Main shared config
# ---------------------------------------------------------------------
@dataclass()
class Config:
    run_type: str
    run_type_params: RunTypeParams

    # --- selection / flags ---
    subsample: bool
    verbose: bool
    use_external_density: bool

    # --- magnitude / distance parameters (shared, but values depend on run_type) ---
    Gmax: float
    Gmin: float
    Grrl: float
    DMerr: float
    bmin: float
    decmin: float

    # --- spline / knot settings ---
    min_knot: float
    max_knot: float
    num_knots: int

    # --- radial range ---
    min_r: float
    max_r: float

    # output / data paths
    figs_root: str
    figs_path: str
    true_path: Optional[str]

    @classmethod
    def from_config_dict(cls, config_dict: dict[str, Any]) -> "Config":
        run_type = config_dict["run_type"]
        rt_config_dict = config_dict[run_type]

        main_config_fields = [f.name for f in fields(Config)]

        class_dict = {
            key: val for key, val in config_dict.items() if key in main_config_fields
        }

        run_params = run_type_dispatch_dict[run_type].from_config_rt_dict(
            rt_config_dict
        )
        class_dict["run_type_params"] = run_params

        for f in fields(run_params):
            f_key = f.name
            if f_key in main_config_fields:
                class_dict[f_key] = getattr(f_key, run_params)

        return cls(**class_dict)


@dataclass
class SmallConfig:
    d2r: float
    bmin: float
    blow_rad: float
    Gmin: float
    Gmax: float
    DMerr: float
    knots_logr: np.ndarray
    lmin_func: Callable
    lsym: float
    lsr_info: Any
    Grrl: float
    bupp: float
    num_knots: int
