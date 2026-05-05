"""Single point that imports every concrete registered implementation.

Importing this module triggers all `@register_*` side effects. The framework
calls it lazily from `frasian.run_experiment` so `import frasian` itself stays
side-effect-free.

Until concrete methods are ported (Step 2+), this module imports nothing.
"""

from __future__ import annotations

_BOOTSTRAPPED = False


def bootstrap() -> None:
    """Import every concrete implementation exactly once."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    from .experiments import confidence_distribution as _exp_cd  # noqa: F401
    from .experiments import coverage as _exp_cov  # noqa: F401
    from .experiments import smoothness as _exp_smooth  # noqa: F401
    from .experiments import width as _exp_width  # noqa: F401
    from .models import bernoulli as _models_bern  # noqa: F401
    from .models import normal_normal as _models_nn  # noqa: F401
    from .statistics import bartlett as _stat_bartlett  # noqa: F401
    from .statistics import lrt as _stat_lrt  # noqa: F401
    from .statistics import signed_root as _stat_sr  # noqa: F401
    from .statistics import wald as _stat_wald  # noqa: F401
    from .statistics import waldo as _stat_waldo  # noqa: F401
    from .tilting import fisher_rao as _tilt_fr  # noqa: F401
    from .tilting import identity as _tilt_identity  # noqa: F401
    from .tilting import mixture as _tilt_mix  # noqa: F401
    from .tilting import ot as _tilt_ot  # noqa: F401
    from .tilting import power_law as _tilt_power  # noqa: F401

    _BOOTSTRAPPED = True
