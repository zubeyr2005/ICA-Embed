from pathlib import Path

from .components import ComponentSpace


ROOT = Path(__file__).resolve().parents[1]
IGNORE = ROOT / "ignore"
DATA = IGNORE / "data"
RESULTS = IGNORE / "results"

__all__ = ["ComponentSpace"]
