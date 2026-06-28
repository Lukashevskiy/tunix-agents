"""Compatibility shim for :mod:`tunix_craftext.tunix.preflight`.

New code should import preflight checks from ``tunix_craftext.tunix``. This
module keeps existing scripts and notebooks stable during the reorganization.
"""

from .tunix.preflight import *  # noqa: F403
