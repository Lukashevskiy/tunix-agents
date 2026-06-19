import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mermaid_fences_render_as_diagram_nodes_in_built_site() -> None:
    subprocess.run(["make", "docs"], cwd=ROOT, check=True, capture_output=True, text=True)

    architecture_page = (ROOT / "site" / "architecture" / "index.html").read_text(encoding="utf-8")
    assert '<pre class="mermaid"><code>flowchart LR' in architecture_page
    assert "mermaid@10.9.1/dist/mermaid.min.js" in architecture_page
