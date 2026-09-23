"""Graph RAG & Multi-Hop Context Extraction Engine for AtlasWiki (Python edition).

Provides Steiner-tree connective subgraph extraction, K-hop connective path discovery,
and linearized markdown context serialization for LLM retrieval-augmented generation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import heapq
from typing import Dict, List, Optional, Set, Tuple

from atlaswiki.ast import LinkType, ParsedDocument
from atlaswiki.graph import KnowledgeGraph
from atlaswiki.storage import StorageEngine


@dataclass
class PathEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0


@dataclass
class GraphPath:
    nodes: List[str]
    edges: List[PathEdge]
    total_weight: float
    hop_count: int

    def to_linearized_string(self) -> str:
        if not self.nodes:
            return ""
        if not self.edges:
            return f"[[{self.nodes[0]}]]"
        out = f"[[{self.nodes[0]}]]"
        for i, edge in enumerate(self.edges):
            next_node = self.nodes[i + 1] if i + 1 < len(self.nodes) else edge.target
            out += f" --[{edge.relation}]--> [[{next_node}]]"
        return out


@dataclass
class SteinerTree:
    terminals: List[str]
    steiner_nodes: List[str]
    edges: List[PathEdge]
    total_weight: float


@dataclass
class NodeContext:
    title: str
    path: Optional[str]
    tags: List[str]
    word_count: int
    pagerank: float
    snippet: Optional[str] = None


@dataclass
class GraphRagResult:
    seeds: List[str]
    paths: List[GraphPath]
    steiner_tree: Optional[SteinerTree]
    involved_nodes: List[NodeContext]
    markdown_context: str


class GraphRagEngine:
    """Connective path and Steiner-tree subgraph extractor for Graph RAG."""

    def __init__(self) -> None:
        self.adj: Dict[str, List[Tuple[str, str, float]]] = {}  # u -> list of (v, relation, weight)
        self.node_info: Dict[str, Dict] = {}  # title -> {path, tags, word_count, snippet}
        self.pagerank: Dict[str, float] = {}

    @classmethod
    def from_documents(cls, docs: List[ParsedDocument]) -> GraphRagEngine:
        engine = cls()
        kg = KnowledgeGraph.from_documents(docs)
        kg.compute_pagerank()
        engine.pagerank = {k: node.pagerank for k, node in kg.nodes.items()}

        for doc in docs:
            engine.node_info[doc.title] = {
                "path": str(doc.path),
                "tags": [t.name for t in doc.tags],
                "word_count": doc.word_count,
                "snippet": doc.chunks[0].content[:200] if doc.chunks else None,
            }
            if doc.title not in engine.adj:
                engine.adj[doc.title] = []

            for link in doc.links:
                target = link.target_note
                rel = "embed" if link.link_type == LinkType.EMBED else "wikilink"
                engine.adj[doc.title].append((target, rel, 1.0))
                if target not in engine.adj:
                    engine.adj[target] = []
                # Add undirected backlink with slightly higher cost for navigation
                engine.adj[target].append((doc.title, "backlink", 1.2))

        return engine

    @classmethod
    def from_storage(cls, storage: StorageEngine) -> GraphRagEngine:
        engine = cls()
        docs = storage.get_all_documents()
        for doc in docs:
            title = doc["title"]
            engine.node_info[title] = {
                "path": doc["path"],
                "tags": [],
                "word_count": doc.get("word_count", 0),
                "snippet": None,
            }
            engine.adj[title] = []

        all_links = storage.get_all_links()
        for link in all_links:
            source = link.get("source_title") or link.get("source_doc_id", "")
            target = link.get("target_note", "")
            if not source or not target:
                continue
            if source not in engine.adj:
                engine.adj[source] = []
            if target not in engine.adj:
                engine.adj[target] = []

            engine.adj[source].append((target, "wikilink", 1.0))
            engine.adj[target].append((source, "backlink", 1.2))

        # Basic degree-based pagerank approximation if not precomputed
        for node in engine.adj:
            engine.pagerank[node] = 1.0 / max(1, len(engine.adj))

        return engine

    def find_shortest_path(self, start: str, target: str, max_hops: int = 4) -> Optional[GraphPath]:
        """Dijkstra shortest path search between two notes."""
        if start not in self.adj or target not in self.adj:
            return None
        if start == target:
            return GraphPath([start], [], 0.0, 0)

        # dist, node, path_nodes, path_edges
        pq: List[Tuple[float, str, List[str], List[PathEdge]]] = [(0.0, start, [start], [])]
        visited: Dict[str, float] = {start: 0.0}

        while pq:
            cost, curr, path_nodes, path_edges = heapq.heappop(pq)
            if curr == target:
                return GraphPath(path_nodes, path_edges, cost, len(path_edges))

            if len(path_edges) >= max_hops:
                continue

            for neighbor, rel, weight in self.adj.get(curr, []):
                new_cost = cost + weight
                if neighbor not in visited or new_cost < visited[neighbor]:
                    visited[neighbor] = new_cost
                    edge = PathEdge(curr, neighbor, rel, weight)
                    heapq.heappush(pq, (new_cost, neighbor, path_nodes + [neighbor], path_edges + [edge]))

        return None

    def extract_paths(self, seeds: List[str], max_hops: int = 3, max_paths: int = 15) -> List[GraphPath]:
        paths: List[GraphPath] = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                p = self.find_shortest_path(seeds[i], seeds[j], max_hops)
                if p:
                    paths.append(p)
                if len(paths) >= max_paths:
                    return paths
        return paths

    def extract_steiner_tree(self, seeds: List[str]) -> Optional[SteinerTree]:
        """Approximate minimum Steiner Tree connecting all seed notes."""
        if len(seeds) < 2:
            return None

        valid_seeds = [s for s in seeds if s in self.adj]
        if len(valid_seeds) < 2:
            return None

        in_tree: Set[str] = {valid_seeds[0]}
        unconnected: Set[str] = set(valid_seeds[1:])
        tree_edges: List[PathEdge] = []
        total_weight = 0.0

        while unconnected:
            best_path: Optional[GraphPath] = None
            best_target: Optional[str] = None

            for u in in_tree:
                for target in unconnected:
                    p = self.find_shortest_path(u, target, max_hops=5)
                    if p:
                        if best_path is None or p.total_weight < best_path.total_weight:
                            best_path = p
                            best_target = target

            if not best_path or not best_target:
                break  # graph is disconnected

            for node in best_path.nodes:
                in_tree.add(node)
            for edge in best_path.edges:
                tree_edges.append(edge)
                total_weight += edge.weight

            unconnected.remove(best_target)

        steiner_nodes = [n for n in in_tree if n not in valid_seeds]
        return SteinerTree(
            terminals=valid_seeds,
            steiner_nodes=steiner_nodes,
            edges=tree_edges,
            total_weight=total_weight,
        )

    def extract_context(
        self,
        seeds: List[str],
        max_hops: int = 3,
        max_paths: int = 10,
        include_content: bool = True,
    ) -> GraphRagResult:
        paths = self.extract_paths(seeds, max_hops, max_paths)
        steiner = self.extract_steiner_tree(seeds)

        involved_titles: Set[str] = set(seeds)
        for p in paths:
            involved_titles.update(p.nodes)
        if steiner:
            involved_titles.update(steiner.terminals)
            involved_titles.update(steiner.steiner_nodes)

        node_contexts = []
        for title in involved_titles:
            info = self.node_info.get(title, {})
            node_contexts.append(
                NodeContext(
                    title=title,
                    path=info.get("path"),
                    tags=info.get("tags", []),
                    word_count=info.get("word_count", 0),
                    pagerank=self.pagerank.get(title, 0.0),
                    snippet=info.get("snippet") if include_content else None,
                )
            )

        node_contexts.sort(key=lambda n: n.pagerank, reverse=True)
        md_context = self.build_markdown_context(seeds, paths, steiner, node_contexts, include_content)

        return GraphRagResult(
            seeds=seeds,
            paths=paths,
            steiner_tree=steiner,
            involved_nodes=node_contexts,
            markdown_context=md_context,
        )

    @staticmethod
    def build_markdown_context(
        seeds: List[str],
        paths: List[GraphPath],
        steiner: Optional[SteinerTree],
        nodes: List[NodeContext],
        include_content: bool,
    ) -> str:
        lines = [
            "<!-- ATLASWIKI_GRAPH_RAG_CONTEXT_START -->",
            "### Knowledge Graph Connective Subgraph",
            f"**Seed Concepts:** {', '.join(f'[[{s}]]' for s in seeds)}",
            "",
            "#### Connective Paths",
        ]

        if not paths:
            lines.append("_No direct connective paths found within hop distance._")
        else:
            for p in paths:
                lines.append(f"- {p.to_linearized_string()} `(hops: {p.hop_count}, cost: {p.total_weight:.2f})`")

        if steiner and steiner.steiner_nodes:
            lines.extend(
                [
                    "",
                    "#### Conceptual Bridge Nodes (Steiner Connectors)",
                    f"Bridge notes binding these topics: {', '.join(f'[[{b}]]' for b in steiner.steiner_nodes)}",
                ]
            )

        lines.extend(["", "#### Note Excerpts"])
        for n in nodes:
            lines.append(f"##### [[{n.title}]]")
            if n.tags:
                lines.append(f"**Tags:** {', '.join(f'#{t}' for t in n.tags)}")
            if include_content and n.snippet:
                lines.append(f"> {n.snippet.strip()}")
            lines.append("")

        lines.append("<!-- ATLASWIKI_GRAPH_RAG_CONTEXT_END -->")
        return "\n".join(lines)
