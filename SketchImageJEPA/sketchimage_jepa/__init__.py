"""SketchImage-JEPA standalone molecular planning prototype."""

from .schema import BenchmarkExample, Candidate, TaskType

__all__ = [
    "BenchmarkExample",
    "Candidate",
    "JEPAConfig",
    "SketchImageJEPAPredictor",
    "TaskType",
]


def __getattr__(name: str):
    if name == "JEPAConfig":
        from .jepa import JEPAConfig

        return JEPAConfig
    if name == "SketchImageJEPAPredictor":
        from .jepa import SketchImageJEPAPredictor

        return SketchImageJEPAPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
