from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from atlaswiki.ast import ParsedDocument


class NodeType(str, Enum):
    DOCUMENT = "document"
    DANGLING = "dangling"
    TAG = "tag"


@dataclass
class NoteNode:
    id: str
    title: str
    path: Optional[Path]
    node_type: NodeType
    tags: List[str] = field(default_factory=list)
    word_count: int = 0
    pagerank: float = 0.0


@dataclass
class LinkEdge:
    source_title: str
    target_title: str
    line_number: int
    context_snippet: Optional[str] = None


class KnowledgeGraph:
    """Directional knowledge graph with PageRank, shortest paths, and D3 serialization."""

    def __init__(self) -> None:
        self.nodes: Dict[str, NoteNode] = {}
        self.adj_out: Dict[str, List[str]] = {}
        self.adj_in: Dict[str, List[str]] = {}
        self.edges: List[LinkEdge] = []
        self.title_map: Dict[str, str] = {}  # lowercase title -> original title
        self.alias_map: Dict[str, str] = {}  # lowercase alias -> original title

    @classmethod
    def from_documents(cls, docs: List[ParsedDocument]) -> KnowledgeGraph:
        kg = cls()

        # Pass 1: Insert all existing document nodes
        path_to_title: Dict[Path, str] = {}
        for doc in docs:
            tags = [t.name for t in doc.tags]
            node = NoteNode(
                id=doc.title,
                title=doc.title,
                path=doc.path,
                node_type=NodeType.DOCUMENT,
                tags=tags,
                word_count=doc.word_count,
            )
            kg._add_node(node)
            path_to_title[doc.path] = doc.title

            for alias in doc.frontmatter.aliases:
                kg.alias_map[alias.lower()] = doc.title

        # Pass 2: Insert edges (and create dangling nodes for missing targets)
        for doc in docs:
            for link in doc.links:
                target_title = link.target_note.strip()
                if not target_title:
                    continue

                clean = target_title.rstrip(".md").strip()
                clean_norm = clean[2:] if clean.startswith("./") else clean
                p = Path(clean_norm)
                p_md = Path(f"{clean_norm}.md")

                resolved_target = None
                if p in path_to_title:
                    resolved_target = path_to_title[p]
                elif p_md in path_to_title:
                    resolved_target = path_to_title[p_md]
                elif doc.path.parent:
                    parent = doc.path.parent
                    if (parent / p) in path_to_title:
                        resolved_target = path_to_title[parent / p]
                    elif (parent / p_md) in path_to_title:
                        resolved_target = path_to_title[parent / p_md]

                if not resolved_target:
                    resolved_target = (
                        kg.title_map.get(clean.lower())
                        or kg.title_map.get(target_title.lower())
                        or kg.alias_map.get(clean.lower())
                        or kg.alias_map.get(target_title.lower())
                    )

                if not resolved_target:
                    # Create dangling node
                    resolved_target = target_title
                    dangling_node = NoteNode(
                        id=resolved_target,
                        title=resolved_target,
                        path=None,
                        node_type=NodeType.DANGLING,
                    )
                    kg._add_node(dangling_node)

                kg._add_edge(
                    doc.title, resolved_target, link.line_number, link.context_snippet
                )

        return kg

    def _add_node(self, node: NoteNode) -> None:
        if node.title not in self.nodes:
            self.nodes[node.title] = node
            self.adj_out[node.title] = []
            self.adj_in[node.title] = []
            self.title_map[node.title.lower()] = node.title

    def _add_edge(
        self, source: str, target: str, line_no: int, snippet: Optional[str]
    ) -> None:
        self.adj_out[source].append(target)
        self.adj_in[target].append(source)
        self.edges.append(
            LinkEdge(
                source_title=source,
                target_title=target,
                line_number=line_no,
                context_snippet=snippet,
            )
        )

    def compute_pagerank(self, damping: float = 0.85, max_iterations: int = 50) -> None:
        """Dangling-Safe PageRank using power iteration."""
        n = len(self.nodes)
        if n == 0:
            return

        node_keys = list(self.nodes.keys())
        pr = {k: 1.0 / n for k in node_keys}

        for _ in range(max_iterations):
            dangling_sum = sum(pr[k] for k in node_keys if len(self.adj_out[k]) == 0)
            base = (1.0 - damping + damping * dangling_sum) / n

            next_pr: Dict[str, float] = {}
            for k in node_keys:
                incoming_sum = sum(
                    pr[src] / len(self.adj_out[src])
                    for src in self.adj_in[k]
                    if len(self.adj_out[src]) > 0
                )
                next_pr[k] = base + damping * incoming_sum

            pr = next_pr

        # Normalize 0.0 to 1.0
        max_pr = max(pr.values()) if pr else 1.0
        min_pr = min(pr.values()) if pr else 0.0
        rng = max(max_pr - min_pr, 1e-6)

        for k, score in pr.items():
            norm_score = max(0.0, min(1.0, (score - min_pr) / rng))
            self.nodes[k].pagerank = norm_score

    def get_pagerank(self, title: str) -> float:
        node = self.nodes.get(title)
        return node.pagerank if node else 0.0

    def get_document_nodes(self) -> List[NoteNode]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.DOCUMENT]

    def degree(self, title: str) -> Optional[Tuple[int, int]]:
        lower = title.lower()
        orig = self.title_map.get(lower) or self.alias_map.get(lower)
        if not orig or orig not in self.nodes:
            return None
        in_deg = len(self.adj_in.get(orig, []))
        out_deg = len(self.adj_out.get(orig, []))
        return (in_deg, out_deg)

    def shortest_path(self, from_title: str, to_title: str) -> Optional[List[str]]:
        """Bidirectional BFS shortest path between Note A and Note B."""
        src = self.title_map.get(from_title.lower())
        dst = self.title_map.get(to_title.lower())

        if not src or not dst:
            return None
        if src == dst:
            return [src]

        fwd_queue = deque([src])
        bwd_queue = deque([dst])

        fwd_parents: Dict[str, Optional[str]] = {src: None}
        bwd_parents: Dict[str, Optional[str]] = {dst: None}

        meeting_node = None

        while fwd_queue and bwd_queue:
            # Forward step
            curr_fwd = fwd_queue.popleft()
            for neighbor in self.adj_out.get(curr_fwd, []):
                if neighbor not in fwd_parents:
                    fwd_parents[neighbor] = curr_fwd
                    fwd_queue.append(neighbor)

                    if neighbor in bwd_parents or neighbor == dst:
                        meeting_node = neighbor
                        break

            if meeting_node:
                break

            # Backward step
            curr_bwd = bwd_queue.popleft()
            for neighbor in self.adj_in.get(curr_bwd, []):
                if neighbor not in bwd_parents:
                    bwd_parents[neighbor] = curr_bwd
                    bwd_queue.append(neighbor)

                    if neighbor in fwd_parents or neighbor == src:
                        meeting_node = neighbor
                        break

            if meeting_node:
                break

        if not meeting_node:
            return None

        # Reconstruct path from src -> meeting_node -> dst
        path_from_src = []
        curr = meeting_node
        while curr:
            path_from_src.append(curr)
            curr = fwd_parents.get(curr)
        path_from_src.reverse()

        path_to_dst = []
        curr = bwd_parents.get(meeting_node)
        while curr:
            path_to_dst.append(curr)
            curr = bwd_parents.get(curr)

        return path_from_src + path_to_dst

    def get_orphans(self) -> List[NoteNode]:
        """Detect isolated notes with in-degree = 0 and out-degree = 0."""
        return [
            node
            for title, node in self.nodes.items()
            if len(self.adj_in[title]) == 0
            and len(self.adj_out[title]) == 0
            and node.node_type == NodeType.DOCUMENT
        ]

    def get_wanted_pages(self) -> List[Tuple[str, int]]:
        """List unresolved / dangling wikilinks sorted by reference count."""
        wanted = [
            (node.title, len(self.adj_in[node.title]))
            for node in self.nodes.values()
            if node.node_type == NodeType.DANGLING
        ]
        wanted.sort(key=lambda x: x[1], reverse=True)
        return wanted

    def to_d3_json(self) -> Dict[str, Any]:
        """Serialize complete knowledge graph for D3.js interactive visualization."""
        nodes = []
        for n in self.nodes.values():
            group = (
                n.tags[0]
                if n.tags
                else ("dangling" if n.node_type == NodeType.DANGLING else "document")
            )
            nodes.append(
                {
                    "id": n.title,
                    "label": n.title,
                    "group": group,
                    "word_count": n.word_count,
                    "pagerank": round(n.pagerank, 4),
                }
            )

        links = [
            {
                "source": e.source_title,
                "target": e.target_title,
                "type": "wikilink",
                "weight": 1.0,
            }
            for e in self.edges
        ]

        return {
            "nodes": nodes,
            "links": links,
            "stats": {
                "total_nodes": len(nodes),
                "total_links": len(links),
            },
        }
