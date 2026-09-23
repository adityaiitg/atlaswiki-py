from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from atlaswiki.storage import StorageEngine


class QueryIntent(str, Enum):
    CODE_SYMBOL = "CodeSymbol"
    NATURAL_LANGUAGE = "NaturalLanguage"
    BALANCED_HYBRID = "BalancedHybrid"


@dataclass
class HybridWeights:
    bm25_weight: float
    vector_weight: float
    intent: QueryIntent


class QueryClassifier:
    """Classifies search queries into code symbol, natural language, or balanced hybrid."""

    CODE_PREFIXES = (
        "fn ",
        "struct ",
        "class ",
        "impl ",
        "enum ",
        "trait ",
        "interface ",
        "type ",
        "def ",
        "let ",
        "val ",
        "var ",
    )
    CODE_EXTS = (".rs", ".py", ".ts", ".js", ".go", ".c", ".cpp", ".md")
    NL_WORDS = {
        "how",
        "why",
        "what",
        "where",
        "when",
        "explain",
        "describe",
        "difference",
    }

    @classmethod
    def classify(cls, query: str) -> HybridWeights:
        q = query.strip()

        # 1. Code symbol check
        is_code = (
            "::" in q
            or "->" in q
            or "()" in q
            or q.startswith("#")
            or (q.startswith('"') and q.endswith('"'))
            or any(q.endswith(ext) for ext in cls.CODE_EXTS)
            or "_" in q
            or any(q.startswith(pfx) for pfx in cls.CODE_PREFIXES)
        )

        if is_code:
            return HybridWeights(
                bm25_weight=0.85, vector_weight=0.15, intent=QueryIntent.CODE_SYMBOL
            )

        # 2. Natural language check
        lower = q.lower()
        words = lower.split()
        is_nl = (
            lower.endswith("?")
            or any(w in cls.NL_WORDS for w in words)
            or len(words) >= 5
        )

        if is_nl:
            return HybridWeights(
                bm25_weight=0.25,
                vector_weight=0.75,
                intent=QueryIntent.NATURAL_LANGUAGE,
            )

        # 3. Balanced hybrid
        return HybridWeights(
            bm25_weight=0.50, vector_weight=0.50, intent=QueryIntent.BALANCED_HYBRID
        )


class HybridRetriever:
    """Retrieves and reranks chunks using BM25 and PKB-specific multipliers."""

    def __init__(self, storage: StorageEngine) -> None:
        self.storage = storage

    def search(
        self,
        query: str,
        limit: int = 10,
        tag_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        weights = QueryClassifier.classify(query)
        raw_hits = self.storage.search_fts(query, limit=limit * 2)

        results = []
        for chunk_id, title, breadcrumbs, snippet, bm25_score in raw_hits:
            if tag_filter:
                clean_tag = tag_filter.lstrip("#").lower()
                if (
                    clean_tag not in title.lower()
                    and clean_tag not in breadcrumbs.lower()
                ):
                    continue

            # PKB Reranker Multipliers
            multiplier = 1.0
            if query.lower() in title.lower():
                multiplier *= 2.5
            if query.lower() in breadcrumbs.lower():
                multiplier *= 2.0

            final_score = abs(bm25_score) * multiplier

            results.append(
                {
                    "chunk_id": chunk_id,
                    "title": title,
                    "breadcrumbs": breadcrumbs,
                    "snippet": snippet,
                    "score": round(final_score, 4),
                    "intent": weights.intent.value,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
