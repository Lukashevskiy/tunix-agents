"""Environment boundaries that normalize CrafText-family transitions for training.

Adapters in this package convert vendor-specific reset/step APIs into a stable
contract of observations, action masks, rewards, and terminal signals that
higher-level learners can consume without leaking vendor implementation details.
"""

from .craftext import (
    AdapterContractError,
    CagedCrafTextAdapter,
    CrafTextAdapter,
    EnvironmentReset,
    EnvironmentStep,
)

__all__ = [
    "AdapterContractError",
    "CagedCrafTextAdapter",
    "CrafTextAdapter",
    "EnvironmentReset",
    "EnvironmentStep",
]
