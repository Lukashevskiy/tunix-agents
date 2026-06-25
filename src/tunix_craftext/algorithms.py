"""Compatibility shim for research-only PPO primitives.

Prefer importing from :mod:`tunix_craftext.research.algorithms`. This legacy
module is kept so older notebooks continue to run while the production path
moves to Tunix Agentic GRPO/PPO.
"""

from .research.algorithms import *  # noqa: F403
