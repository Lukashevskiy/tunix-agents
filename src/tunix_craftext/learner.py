"""Compatibility shim for the local research PPO learner.

Prefer importing from :mod:`tunix_craftext.research.learner`. Production
training should use Tunix ``RLCluster`` and Agentic learners instead.
"""

from .research.learner import *  # noqa: F403
