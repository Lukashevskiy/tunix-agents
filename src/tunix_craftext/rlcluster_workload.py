"""Compatibility shim for :mod:`tunix_craftext.tunix.rlcluster_workload`.

New code should import RLCluster workload contracts from ``tunix_craftext.tunix``.
This module preserves older import paths while the codebase migrates.
"""

from .tunix.rlcluster_workload import *  # noqa: F403
