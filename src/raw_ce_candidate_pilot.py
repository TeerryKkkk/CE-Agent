from __future__ import annotations

# Compatibility wrapper for p99 extreme-rainfall candidate construction.

from .climate_pipeline import candidate_construction as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
__all__ = [name for name in globals() if not name.startswith("__")]


if __name__ == "__main__":
    from .climate_pipeline.candidate_construction import main

    raise SystemExit(main())
