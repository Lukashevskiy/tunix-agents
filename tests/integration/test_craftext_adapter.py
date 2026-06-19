"""Integration smoke test intentionally skips until environment extras are installed."""

import importlib.util

import pytest


@pytest.mark.integration
@pytest.mark.skipif(importlib.util.find_spec("craftext") is None, reason="install the envs extra")
def test_craftext_imports_from_vendored_or_installed_package() -> None:
    import craftext  # noqa: F401
