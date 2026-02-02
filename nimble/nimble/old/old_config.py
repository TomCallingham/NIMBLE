from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, ClassVar, Literal, Optional, Sequence, Tuple, Union

from .footprint_boundary import getSurveyFootprintBoundary

import numpy as np

from .helpers.auridesi.auridesi_params import AuriDesiParams, AuriDesiParams_from_args
from .helpers.desi.desi_config import IronParams

d2r = np.pi / 180  # conversion from degrees to radians


run_types = ["auridesi", "iron"]
RunTypeParams = Union[AuriDesiParams, IronParams]


# ---------------------------------------------------------------------
# Main shared config
# ---------------------------------------------------------------------
@dataclass
class Config:
    run_type: RunTypeParams
    run_type_params: Optional[RunTypeParams] = None

    # --- selection / flags ---
    subsample: bool = False
    verbose: bool = True
    use_external_density: bool = False

    # --- magnitude / distance parameters (shared, but values depend on run_type) ---
    Gmax: float = 20.0
    Gmin: float = 16.0
    Grrl: float = 0.58
    DMerr: float = 0.24
    bmin: float = 30.0
    decmin: float = -35.0

    # --- spline / knot settings ---
    min_knot: float = 5.0
    max_knot: float = 80.0
    num_knots: int = 4

    # --- radial range ---
    min_r: float = 1.0
    max_r: float = 100.0

    # --- derived constants ---
    d2r: float = field(default=np.pi / 180.0, init=False)

    # output / data paths
    figs_root: str = "results/"
    figs_path: str = "results/"
    true_path: Optional[str] = None

    # loader wiring
    load_params: Tuple[Any, ...] = field(default_factory=tuple)
    knot_override: Optional[Tuple[float, float, int]] = None
    timestamp: str = ""

    # USAGE: ClassVar[str] = (
    #     "Usage:\n"
    #     "  deconv auridesi HALONUM LSRDEG [min_knot max_knot num_knots]\n"
    #     "  deconv iron [min_knot max_knot num_knots]"
    # )

    # Volume. Bad defaults
    blow_rad: float = 0.0
    bupp: float = 0.0
    # lmin_func: Callable = lambda x: x
    lsym: float = 0.0

    def __post_init__(self) -> None:
        self.apply_run_type_defaults()

    # ------------------------------------------------------------------
    # Run-type defaults (single source of truth for Gmax/Gmin/etc)
    # ------------------------------------------------------------------
    def apply_run_type_defaults(self) -> None:
        if self.run_type == "auridesi":
            self.Gmax = 19.0
            self.Gmin = 16.0
            self.bmin = 30.0
            self.min_r = 1.0
            self.max_r = 100.0
        elif self.run_type == "iron":
            self.Gmax = 20.0
            self.Gmin = 16.0
            self.bmin = 30.0
            self.min_r = 1.0
            self.max_r = 100.0
        else:
            raise ValueError(f"Unknown run_type {self.run_type!r}")

    # ------------------------------------------------------------------
    # CLI construction
    # ------------------------------------------------------------------
    @classmethod
    def from_args(data_type, argv: Sequence[str]) -> "Config":
        """
        Build a Config from command-line style arguments (sys.argv).

        argv[0] is the program name, argv[1] is the run_type.
        """
        if len(argv) < 2:
            raise SystemExit(data_type.USAGE)

        data_type = argv[1].lower()
        if data_type not in run_types:
            raise AssertionError(f"Unknown run type {data_type}")

        # Create base config with run_type; __post_init__ sets shared defaults
        cfg = data_type(run_type=data_type)

        # # optional knot override: min_knot max_knot num_knots
        # if len(argv) == 7:
        #     knot_override = data_type._parse_knots(argv[4:])

        # knot_override: Optional[Tuple[float, float, int]] = None

        if data_type == "auridesi":
            cfg.run_type_params = AuriDesiParams.from_args(argv)
            cfg.figs_root = cfg.run_type_params.figs_root
            cfg.true_path = cfg.run_type_params.true_path

        elif kind == "iron":
            print("\033[1;33m**** RUNNING IRON ****\033[0m")

            cfg.run_type_params = IronParams()
            cfg.figs_root = "results/iron/"
            cfg.true_path = None
            cfg.load_params = ()

            if len(argv) == 5:
                knot_override = data_type._parse_knots(argv[2:])

        # Apply knot override if present
        if knot_override is not None:
            cfg.min_knot, cfg.max_knot, cfg.num_knots = knot_override
            cfg.knot_override = knot_override

        # Build timestamped figs_path
        cfg.timestamp = datetime.now().strftime("%m%d%y_%H%M")
        cfg.figs_path = (
            f"{cfg.figs_root}"
            f"{cfg.min_knot}_{cfg.max_knot}_{cfg.num_knots}/"
            f"{cfg.timestamp}/"
        )
        if cfg.subsample:
            cfg.figs_path += "sub/"

        print("Output to", cfg.figs_path)
        cfg._init_volume()
        return cfg

    @staticmethod
    def _parse_knots(args: Sequence[str]) -> Tuple[float, float, int]:
        if len(args) != 3:
            raise SystemExit(
                "Knot override requires 3 values: min_knot max_knot num_knots"
            )
        min_k = float(args[0])
        max_k = float(args[1])
        n_k = int(args[2])
        return min_k, max_k, n_k

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _get_loader(self) -> Callable[..., Any]:
        if self.run_type == "auridesi":
            from .helpers import auridesi_helpers as auridesi

            # Mirror legacy globals if helpers still use them:
            auridesi.Gmax = self.Gmax
            auridesi.bmin = self.bmin
            return auridesi.load

        elif self.run_type == "iron":
            from .helpers import iron_helpers as iron

            iron.Gmax = self.Gmax
            iron.Gmin = self.Gmin
            return iron.load_RRL

        else:
            raise ValueError(f"Unknown run_type {self.run_type!r}")

    def load_data(self):
        """
        Load data using the appropriate helper.

        If no args/kwargs are passed, uses self.load_params by default.
        """
        loader = self._get_loader()
        return loader(*self.load_params, self.subsample, self.verbose)

    def _init_volume(self):
        blow_rad, bupp, lmin_func, lsym = getSurveyFootprintBoundary(self.decmin)
        self.blow_rad = float(blow_rad)
        self.bupp = float(bupp)
        self.lmin_func = lmin_func
        self.lsym = lsym

    def to_small(self) -> "SmallConfig":
        return SmallConfig(
            d2r=self.d2r,
            bmin=self.bmin,
            blow_rad=self.blow_rad,
            Gmin=self.Gmin,
            Gmax=self.Gmax,
            DMerr=self.DMerr,
            knots_logr=self.knots_logr,
            lmin_func=self.lmin_func,
            lsym=self.lsym,
            lsr_info=self.lsr_info,
            Grrl=self.Grrl,
            bupp=self.bupp,
            num_knots=self.num_knots,
        )

    def write_csv(self, data, filename, column_titles):
        np.savetxt(
            fname=self.figs_path + filename, X=data, delimiter=",", header=column_titles
        )


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
