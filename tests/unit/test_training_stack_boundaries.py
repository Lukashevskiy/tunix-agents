"""Architectural guardrails for the Tunix-first training stack."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "tunix_craftext"

RESEARCH_MODULES = {
    "tunix_craftext.research",
    "tunix_craftext.research.algorithms",
    "tunix_craftext.research.algorithm_registry",
    "tunix_craftext.research.learner",
    "tunix_craftext.research.llm_ppo",
}
SEMANTIC_PACKAGES = {
    "core": {"contracts.py", "resources.py", "tensor_types.py"},
    "env": {"agentic_craftext.py", "config.py", "prompts.py", "runtime.py", "text_policy.py"},
    "rollouts": {
        "batched.py",
        "hybrid.py",
        "random_policy.py",
        "reference.py",
        "text_episode.py",
    },
    "models": {"llm.py", "llm_actor.py", "profile.py", "tunix_actor.py", "tunix_adapter.py"},
    "training": {
        "agentic_grpo_smoke.py",
        "agentic_ppo.py",
        "experience_builders.py",
        "flashbax_replay.py",
        "grpo_profile.py",
    },
    "artifacts": {
        "checkpoints.py",
        "comet_adapter.py",
        "observability.py",
        "profiling.py",
        "provenance.py",
        "replay.py",
        "text_trajectory.py",
        "trajectory_gif.py",
    },
    "tunix": {"preflight.py", "rlcluster_workload.py", "topology.py"},
}
ALLOWED_ROOT_FILES = {"__init__.py"}
ALLOWED_ROOT_DIRECTORIES = set(SEMANTIC_PACKAGES) | {"adapters", "interop", "research"}


def test_research_ppo_stack_is_physically_separated_from_production_modules() -> None:
    """Old local PPO mechanics live under research and are not production imports."""
    expected = {
        SRC / "research" / "__init__.py",
        SRC / "research" / "algorithms.py",
        SRC / "research" / "algorithm_registry.py",
        SRC / "research" / "learner.py",
        SRC / "research" / "llm_ppo.py",
    }
    for path in expected:
        assert path.is_file(), f"missing research module: {path.relative_to(ROOT)}"

    for path in SRC.rglob("*.py"):
        if path.name == "__init__.py" or "research" in path.parts:
            continue
        imports = _absolute_imports(path)
        forbidden = imports & RESEARCH_MODULES
        assert not forbidden, f"{path.relative_to(ROOT)} imports research-only PPO: {forbidden}"


def test_root_package_contains_only_public_facade_and_semantic_packages() -> None:
    """Runtime modules live in semantic packages, not as duplicate root shims."""
    root_files = {path.name for path in SRC.iterdir() if path.is_file()}
    root_directories = {
        path.name for path in SRC.iterdir() if path.is_dir() and path.name != "__pycache__"
    }

    assert root_files == ALLOWED_ROOT_FILES
    assert root_directories == ALLOWED_ROOT_DIRECTORIES


def test_semantic_packages_own_the_runtime_code() -> None:
    """Runtime logic lives in domain packages with explicit ownership."""
    for package, filenames in SEMANTIC_PACKAGES.items():
        package_root = SRC / package
        assert (package_root / "__init__.py").is_file(), f"missing package {package}"
        for filename in filenames:
            assert (package_root / filename).is_file(), f"missing {package}/{filename}"


def test_training_stack_audit_documents_current_ownership_and_migration() -> None:
    """The docs must explain which PPO/GRPO path is authoritative."""
    audit = (ROOT / "docs" / "training-stack-audit.md").read_text(encoding="utf-8")

    assert "Tunix Agentic GRPO first" in audit
    assert "Agentic PPO extension" in audit
    assert "research/smoke" in audit
    assert "hardware-gated one-update" in audit
    assert "cost critic" in audit


def test_cpu_quality_gate_installs_tunix_extra_for_tunix_unit_contracts() -> None:
    """Unit tests import Tunix bridge APIs, so CI must install the Tunix extra."""
    workflow = (ROOT / ".github" / "workflows" / "quality.yml").read_text(
        encoding="utf-8"
    )

    assert "uv sync --locked --extra dev --extra tunix" in workflow
    assert "pytest tests/unit" in workflow


def test_jupyter_checkpoints_are_not_tracked_by_git() -> None:
    """Generated notebook checkpoints are local editor state, never repository evidence."""
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()

    assert not [path for path in tracked if ".ipynb_checkpoints" in path]


def _absolute_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module)
    return imports
