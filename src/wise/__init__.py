"""
WISE core package.

This package provides:
- model and norm abstractions (wise.model, wise.norm)
- pluggable layer implementations (wise.layers.*)
- scoring and aggregation (wise.scoring.*)
- I/O helpers (wise.io.*)
"""

from .model import Constraint, Norm, View  # convenience re-exports
