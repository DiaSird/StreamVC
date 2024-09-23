from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class StepLR:
    """StepLR learning rate scheduler"""

    step: int = 100
    """Step interval for learning rate scheduler updates"""

    gamma: float = 0.1
    """Controls the decay rate"""


@dataclass(frozen=True)
class LinearLR:
    """LinearLR learning rate scheduler"""

    start: float = 1.0
    """Initial learning rate mutiplier"""
    end: float = 1.0
    """Final learning rate mutiplier"""


@dataclass(frozen=True)
class ExponentialLR:
    """ExponentialLR learning rate scheduler"""

    gamma: float = 0.1
    """Controls the decay rate"""


@dataclass(frozen=True)
class OneCycleLR:
    """OneCycleLR learning rate scheduler"""

    max: float = 1e-3
    """Max learning rate"""
    pct_start: float = 0.3
    """The percentage of the cycle spent increasing the learning rate"""
    div_factor: float = 25
    """Determines the initial learning rate via initial_lr = max_lr/div_factor"""
    final_div_factor: float = 1e4
    """Determines the minimum learning rate via min_lr = initial_lr/final_div_factor"""


@dataclass(frozen=True)
class CosineAnnealingWarmRestarts:
    """CosineAnnealingWarmRestarts learning rate scheduler"""

    T_0: int = 100
    """Number of iterations until the first restart"""

    eta_min: float = 0
    """Minimum learning rate"""


LRScheduler = Union[
    StepLR, LinearLR, ExponentialLR, OneCycleLR, CosineAnnealingWarmRestarts
]
