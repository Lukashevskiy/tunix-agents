"""Compatibility shim for :mod:`tunix_craftext.tunix.topology`.

New code should import topology contracts from ``tunix_craftext.tunix`` or
``tunix_craftext.tunix.topology``. This module keeps existing notebooks,
scripts and third-party experiments working during the package reorganization.
"""

from .tunix.topology import *  # noqa: F403
