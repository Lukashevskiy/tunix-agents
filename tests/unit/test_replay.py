import json
from pathlib import Path

from tunix_craftext.replay import ReplayArtifact, ReplayStep, save_replay


def test_replay_is_versioned_and_preserves_raw_completion(tmp_path: Path) -> None:
    path = tmp_path / "replay.json"
    save_replay(
        path,
        ReplayArtifact(
            "config.yaml",
            "abc",
            "scripted",
            (ReplayStep(0, "p", "<action>DO</action>", 2, "DO", 1.0, False),),
        ),
    )
    data = json.loads(path.read_text())
    assert data["schema"] == "tunix-craftext.replay/v1"
    assert data["steps"][0]["raw_completion"] == "<action>DO</action>"
