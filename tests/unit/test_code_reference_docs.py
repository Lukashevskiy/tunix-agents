"""Ensure the MkDocs code reference stays wired to public pipeline abstractions."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_code_reference_is_in_site_nav_and_mentions_pipeline_abstractions() -> None:
    """The human-facing site must expose code/API abstractions, not only prose docs."""
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    page = (ROOT / "docs" / "code-reference.md").read_text(encoding="utf-8")

    assert "Код:" in nav
    assert "Обзор API: code-reference.md" in nav
    assert "Автодока API: _generated/api-reference.md" in nav
    assert "Типы и docstrings: code-quality.md" in nav
    assert "Автодока API" in page
    for required in (
        "collect_batched_text_rollout",
        "replays_from_batched_rollout",
        "text_trajectory_from_replay",
        "masked_token_ppo_loss",
        "get_algorithm",
        "masked_action",
    ):
        assert required in page


def test_generated_api_reference_is_built_from_docstrings() -> None:
    """The MkDocs build helper should generate a docstring-based API page."""
    import subprocess

    subprocess.run(["python3", "scripts/generate_dashboard.py"], cwd=ROOT, check=True)
    generated = (ROOT / "docs" / "_generated" / "api-reference.md").read_text(
        encoding="utf-8"
    )

    assert "Автодока API" in generated
    assert "tunix_craftext.batched_rollout" in generated
    assert "collect_batched_text_rollout" in generated
    assert "tunix_craftext.text_trajectory" in generated
