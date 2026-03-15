"""Models package for the H(AI)LP project."""

from .baseline_gpt import BaselineConfig, BaselineGPT
from .hailp_model import HAILPConfig, HAILPModel

# Backwards-compatibility: old LifeLink names still importable
from .hailp_model import LifeLinkConfig, LifeLinkRWKV

__all__ = [
    "BaselineConfig",
    "BaselineGPT",
    "HAILPConfig",
    "HAILPModel",
    # legacy aliases
    "LifeLinkConfig",
    "LifeLinkRWKV",
]
