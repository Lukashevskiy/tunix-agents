"""Keep public tutorial notebooks valid and discoverable without requiring Jupyter at test time."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "examples" / "notebooks"


def test_example_notebooks_are_valid_nbformat_with_runnable_imports() -> None:
    """Validate the notebook JSON and its primary public API import.

    :raises AssertionError: If a tutorial notebook disappears, has invalid format or no longer
        teaches the corresponding public symbol.
    """
    expected = {
        "01_rollout_contract.ipynb": "collect_rollout",
        "02_craftext_adapter.ipynb": "CrafTextAdapter",
        "03_model_interop_lora.ipynb": "merge_lora_adapters",
        "04_megaprompts_environment_to_prompt.ipynb": "MegaPromptRenderer",
    }
    for filename, required_symbol in expected.items():
        notebook = json.loads((NOTEBOOKS / filename).read_text(encoding="utf-8"))
        source = "".join("".join(cell["source"]) for cell in notebook["cells"])
        assert notebook["nbformat"] == 4
        assert required_symbol in source
