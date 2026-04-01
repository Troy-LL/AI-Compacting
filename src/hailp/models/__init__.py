"""Models package for the H(AI)LP project."""

from .baseline_gpt import BaselineConfig, BaselineGPT
from .hailp_model import HAILPConfig, HAILPModel

__all__ = [
    "BaselineConfig",
    "BaselineGPT",
    "HAILPConfig",
    "HAILPModel",
]
