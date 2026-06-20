"""Real-vendor smoke test; deliberately skipped without CrafText/Craftax extras."""

import importlib.util

import pytest


@pytest.mark.integration
@pytest.mark.skipif(importlib.util.find_spec("craftext") is None, reason="install the envs extra")
def test_vendor_craftext_adapter_module_is_available() -> None:
    from tunix_craftext.adapters import CrafTextAdapter

    assert CrafTextAdapter.__name__ == "CrafTextAdapter"
