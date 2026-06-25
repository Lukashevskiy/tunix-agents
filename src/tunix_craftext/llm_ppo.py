"""Compatibility shim for research PPO evaluation helpers.

Prefer importing from :mod:`tunix_craftext.research.llm_ppo`. This legacy path
remains for notebooks that inspect token-level PPO quantities.
"""

from .research.llm_ppo import *  # noqa: F403
