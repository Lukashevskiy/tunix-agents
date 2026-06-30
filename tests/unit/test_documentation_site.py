import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_docs() -> None:
    """Build the docs with a project-local uv cache for hermetic test runs."""

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(ROOT / ".uv-cache"))
    env.setdefault("UV_RUN_FLAGS", "--no-sync")
    subprocess.run(["make", "docs"], cwd=ROOT, env=env, check=True, capture_output=True, text=True)


def test_mermaid_fences_render_as_diagram_nodes_in_built_site() -> None:
    build_docs()

    architecture_page = (ROOT / "site" / "architecture" / "index.html").read_text(encoding="utf-8")
    assert '<pre class="mermaid"><code>flowchart LR' in architecture_page
    assert "mermaid@10.9.1/dist/mermaid.min.js" in architecture_page


def test_task_graph_page_includes_interactive_runtime_and_data() -> None:
    build_docs()

    graph_page = (ROOT / "site" / "_generated" / "task-graph" / "index.html").read_text(
        encoding="utf-8"
    )
    assert 'id="task-graph-data"' in graph_page
    assert "cytoscape@3.30.2/dist/cytoscape.min.js" in graph_page
    assert "javascripts/task-graph.js" in graph_page
