from __future__ import annotations

# Compatibility shim.
#
# New raw-component-event-first code should prefer importing src.climate_pipeline.config.
# Historical MVP, DTER-first, China fixture, and older schema modules still import
# src.config, so this file re-exports both active runtime settings and legacy
# constants until those modules are retired or migrated.

from .active_config import *  # noqa: F403
from .legacy_config import *  # noqa: F403
from .active_config import __all__ as _active_all
from .legacy_config import __all__ as _legacy_all

__all__ = [*_active_all, *_legacy_all]
