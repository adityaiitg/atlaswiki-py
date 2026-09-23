"""Hierarchical Navigable Small World (HNSW) vector search in pure Python."""

from __future__ import annotations
import heapq
import math
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from atlaswiki.quantization import Sq8Vector


class HnswNode:
    __slots__ = ("node_id", "vector", "sq8", "neighbors", "max_level")

    def __init__(
        self, node_id: int, vector: List[float], sq8: Sq8Vector, max_level: int
    ) -> None:
        self.node_id = node_id
        self.vector = vector
        self.sq8 = sq8
        self.max_level = max_level
        # neighbors[level] = list of neighbor node_ids
        self.neighbors: List[List[int]] = [[] for _ in range(max_level + 1)]


def cosine_distance_f32(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a < 1e-7 or norm_b < 1e-7:
        return 1.0
    sim = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    return 1.0 - max(-1.0, min(1.0, sim))


class HnswIndex:
    """Pure-Python Hierarchical Navigable Small World index with SQ8 quantization."""

    def __init__(
        self,
        m: int = 16,
        ef_construction: int = 64,
        m0: Optional[int] = None,
        use_sq8: bool = True,
    ) -> None:
        self.m = m
        self.m0 = m0 or (2 * m)
        self.ef_construction = ef_construction
        self.use_sq8 = use_sq8
        self.ml = 1.0 / math.log(m)

        self.nodes: Dict[int, HnswNode] = {}
        self.entry_point: Optional[int] = None
        self.max_level = -1
        self._rng = random.Random(42)

    def __len__(self) -> int:
        return len(self.nodes)

    def _random_level(self) -> int:
        r = self._rng.random()
        if r == 0.0:
            r = 1e-9
        return int(-math.log(r) * self.ml)

    def _dist_query(self, query: Sequence[float], node_id: int) -> float:
        node = self.nodes[node_id]
        if self.use_sq8:
            sim = node.sq8.asymmetric_cosine_similarity(query)
            return 1.0 - sim
        return cosine_distance_f32(query, node.vector)

    def _dist_nodes(self, a_id: int, b_id: int) -> float:
        node_a = self.nodes[a_id]
        node_b = self.nodes[b_id]
        if self.use_sq8:
            sim = node_a.sq8.cosine_similarity(node_b.sq8)
            return 1.0 - sim
        return cosine_distance_f32(node_a.vector, node_b.vector)

    def insert(self, node_id: int, vector: Sequence[float]) -> None:
        """Insert a vector into the HNSW graph."""
        vec_list = list(vector)
        sq8 = Sq8Vector.from_float(vec_list)
        level = self._random_level()
        new_node = HnswNode(node_id, vec_list, sq8, level)
        self.nodes[node_id] = new_node

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = level
            return

        curr_ep = self.entry_point
        # 1. Greedy search down from top level to level + 1
        for lc in range(self.max_level, level, -1):
            curr_dist = self._dist_query(vector, curr_ep)
            changed = True
            while changed:
                changed = False
                for neighbor in self.nodes[curr_ep].neighbors[lc]:
                    d = self._dist_query(vector, neighbor)
                    if d < curr_dist:
                        curr_dist = d
                        curr_ep = neighbor
                        changed = True

        # 2. Search and connect for levels min(level, max_level) down to 0
        ep_candidates = [curr_ep]
        for lc in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer_candidates(
                vector, ep_candidates, self.ef_construction, lc
            )
            # Select M nearest neighbors
            candidates.sort(key=lambda x: x[0])
            max_m = self.m0 if lc == 0 else self.m
            selected_neighbors = [nid for _, nid in candidates[:max_m]]

            new_node.neighbors[lc] = selected_neighbors
            for neighbor_id in selected_neighbors:
                neighbor_node = self.nodes[neighbor_id]
                neighbor_node.neighbors[lc].append(node_id)
                # Prune if exceeding max connections
                if len(neighbor_node.neighbors[lc]) > max_m:
                    n_dists = [
                        (self._dist_nodes(neighbor_id, n), n)
                        for n in neighbor_node.neighbors[lc]
                    ]
                    n_dists.sort(key=lambda x: x[0])
                    neighbor_node.neighbors[lc] = [n for _, n in n_dists[:max_m]]

            ep_candidates = [nid for _, nid in candidates]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = node_id

    def _search_layer_candidates(
        self,
        query: Sequence[float],
        eps: List[int],
        ef: int,
        level: int,
    ) -> List[Tuple[float, int]]:
        v: Set[int] = set(eps)
        # c: min-heap of (dist, node_id)
        c: List[Tuple[float, int]] = []
        # w: max-heap of (-dist, node_id)
        w: List[Tuple[float, int]] = []

        for ep in eps:
            d = self._dist_query(query, ep)
            heapq.heappush(c, (d, ep))
            heapq.heappush(w, (-d, ep))

        while c:
            dist_c, cand_id = heapq.heappop(c)
            furthest_w_dist = -w[0][0]
            if dist_c > furthest_w_dist:
                break

            for neighbor in self.nodes[cand_id].neighbors[level]:
                if neighbor not in v:
                    v.add(neighbor)
                    d_neighbor = self._dist_query(query, neighbor)
                    furthest_w_dist = -w[0][0]
                    if d_neighbor < furthest_w_dist or len(w) < ef:
                        heapq.heappush(c, (d_neighbor, neighbor))
                        heapq.heappush(w, (-d_neighbor, neighbor))
                        if len(w) > ef:
                            heapq.heappop(w)

        return [(-neg_d, nid) for neg_d, nid in w]

    def search(
        self,
        query: Sequence[float],
        k: int = 10,
        ef_search: int = 32,
    ) -> List[Tuple[int, float]]:
        """Search top-k nearest neighbors. Returns list of (node_id, cosine_similarity)."""
        if self.entry_point is None or not self.nodes:
            return []

        curr_ep = self.entry_point
        # 1. Greedy descent down to level 0
        for lc in range(self.max_level, 0, -1):
            curr_dist = self._dist_query(query, curr_ep)
            changed = True
            while changed:
                changed = False
                for neighbor in self.nodes[curr_ep].neighbors[lc]:
                    d = self._dist_query(query, neighbor)
                    if d < curr_dist:
                        curr_dist = d
                        curr_ep = neighbor
                        changed = True

        # 2. Search bottom layer with ef_search
        candidates = self._search_layer_candidates(
            query, [curr_ep], max(ef_search, k), 0
        )
        candidates.sort(key=lambda x: x[0])

        # Convert distance (1 - sim) back to cosine similarity
        results = []
        for d, nid in candidates[:k]:
            sim = 1.0 - d
            results.append((nid, max(-1.0, min(1.0, sim))))
        return results
