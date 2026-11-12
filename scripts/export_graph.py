"""Export a Mermaid diagram of the implemented graphs to docs/graph.mmd.

Usage:  python scripts/export_graph.py
The diagram is generated from the compiled graph, so it always matches the code.
Paste the file into https://mermaid.live to view it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.container import build_container  # noqa: E402

container = build_container(get_settings(), InMemorySaver())
mermaid = container.chat_graph.get_graph(xray=True).draw_mermaid()   # xray expands the subgraph
output_path = Path("docs/graph.mmd")
output_path.parent.mkdir(exist_ok=True)
output_path.write_text(mermaid, encoding="utf-8")
print(f"wrote {output_path}")
