"""Evidence gathering for the automatic editor (sound, picture, scene cuts, faces, optional speech)."""
from .evidence import AnalyzeParams, Evidence  # noqa: F401


def __getattr__(name):                      # lazy: ``pipeline`` is only needed when analysing
    if name in ("analyze_asset", "refine_cut"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)
