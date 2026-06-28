import json
from pathlib import Path

from tunix_craftext.artifacts.replay import ReplayArtifact, ReplayStep, save_replay


def test_replay_is_versioned_and_preserves_raw_completion(tmp_path: Path) -> None:
    path = tmp_path / "replay.json"
    save_replay(
        path,
        ReplayArtifact(
            "config.yaml",
            "abc",
            "scripted",
            (
                ReplayStep(
                    0,
                    "p",
                    "<action>DO</action>",
                    2,
                    "DO",
                    1.0,
                    False,
                    masked_action=1,
                ),
            ),
        ),
    )
    data = json.loads(path.read_text())
    assert data["schema"] == "tunix-craftext.replay/v3"
    assert data["steps"][0]["raw_completion"] == "<action>DO</action>"
    assert data["steps"][0]["masked_action"] == 1
    assert data["steps"][0]["fallback_used"] is False
