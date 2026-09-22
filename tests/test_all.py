import sys
from pathlib import Path
import tempfile
import unittest

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atlaswiki.ast import LinkType
from atlaswiki.diagnostics import DiagnosticsEngine
from atlaswiki.graph import KnowledgeGraph
from atlaswiki.parser import MarkdownParser
from atlaswiki.retrieval import QueryClassifier, QueryIntent
from atlaswiki.storage import StorageEngine


class TestMarkdownParser(unittest.TestCase):
    def setUp(self):
        self.parser = MarkdownParser()

    def test_frontmatter_and_wikilinks(self):
        md = """---
title: Machine Learning
tags:
  - ai/ml
  - cs
aliases:
  - ML
---

# Overview

Machine learning is a subset of AI.
See [[Deep Learning]] and [[Statistics#Bayes]] and [[Neural Networks#^feedforward-def]].
Also check ![[Embedded Note]] and [[Target Note|Custom Alias]].
"""
        doc = self.parser.parse_file(Path("Machine Learning.md"), md)
        self.assertEqual(doc.title, "Machine Learning")
        self.assertIn("ML", doc.frontmatter.aliases)
        self.assertTrue(any(t.name == "ai/ml" for t in doc.tags))

        # Check links
        targets = [l.target_note for l in doc.links]
        self.assertIn("Deep Learning", targets)
        self.assertIn("Statistics", targets)
        self.assertIn("Neural Networks", targets)
        self.assertIn("Embedded Note", targets)
        self.assertIn("Target Note", targets)

        # Check embed
        embed = next(l for l in doc.links if l.target_note == "Embedded Note")
        self.assertEqual(embed.link_type, LinkType.EMBED)

        # Check anchor & block
        nn_link = next(l for l in doc.links if l.target_note == "Neural Networks")
        self.assertTrue(nn_link.target_block == "feedforward-def" or nn_link.target_heading == "feedforward-def")

    def test_code_block_isolation(self):
        md = """# Code Test

```python
# This is a comment, not a #tag
link = "[[Not A Real Link]]"
```

Real text with [[Actual Link]] and #actual/tag.
"""
        doc = self.parser.parse_file(Path("Code.md"), md)
        targets = [l.target_note for l in doc.links]
        self.assertIn("Actual Link", targets)
        self.assertNotIn("Not A Real Link", targets)

        tag_names = [t.name for t in doc.tags]
        self.assertIn("actual/tag", tag_names)
        self.assertNotIn("tag", tag_names)


class TestStorageEngine(unittest.TestCase):
    def test_document_lifecycle_and_fts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.db"
            storage = StorageEngine.open(db_path)
            parser = MarkdownParser()

            doc_a = parser.parse_file(
                Path("A.md"),
                "# Note A\nAlgorithms and data structures.\nConnects to [[Note B]].\n",
            )
            doc_b = parser.parse_file(
                Path("B.md"),
                "# Note B\nMachine learning foundations.\nReferences [[Note A]].\n",
            )

            storage.sync_document(doc_a, "doc-a", "A.md", 1000, 100)
            storage.sync_document(doc_b, "doc-b", "B.md", 1001, 120)

            stats = storage.get_stats()
            self.assertEqual(stats.total_documents, 2)
            self.assertEqual(stats.total_links, 2)

            # Search FTS
            hits = storage.search_fts("algorithms data", limit=10)
            self.assertTrue(len(hits) >= 1)
            self.assertEqual(hits[0][1], "Note A")

            # Backlinks to Note A
            backlinks = storage.get_backlinks("Note A")
            self.assertEqual(len(backlinks), 1)
            self.assertEqual(backlinks[0].source_title, "Note B")


class TestKnowledgeGraph(unittest.TestCase):
    def test_graph_pagerank_and_shortest_path(self):
        parser = MarkdownParser()
        doc_a = parser.parse_file(Path("A.md"), "# Note A\nConnects to [[Note B]].")
        doc_b = parser.parse_file(Path("B.md"), "# Note B\nConnects to [[Note C]].")
        doc_c = parser.parse_file(Path("C.md"), "# Note C\nEnd of chain. Mentions [[Unwritten Note]].")
        doc_d = parser.parse_file(Path("D.md"), "# Note D\nStandalone orphan.")

        kg = KnowledgeGraph.from_documents([doc_a, doc_b, doc_c, doc_d])

        # Shortest path A -> C
        path = kg.shortest_path("Note A", "Note C")
        self.assertEqual(path, ["Note A", "Note B", "Note C"])

        # Disconnected path A -> D
        disconn = kg.shortest_path("Note A", "Note D")
        self.assertIsNone(disconn)

        # Orphans: Note D
        orphans = kg.get_orphans()
        self.assertEqual(len(orphans), 1)
        self.assertEqual(orphans[0].title, "Note D")

        # Wanted pages: "Unwritten Note"
        wanted = kg.get_wanted_pages()
        self.assertEqual(len(wanted), 1)
        self.assertEqual(wanted[0][0], "Unwritten Note")

        # PageRank
        kg.compute_pagerank(0.85, 30)
        self.assertTrue(kg.get_pagerank("Note B") >= 0.0)


class TestDiagnosticsEngine(unittest.TestCase):
    def test_broken_links_and_typos(self):
        parser = MarkdownParser()
        doc_ml = parser.parse_file(
            Path("Machine Learning.md"),
            "---\ntitle: Machine Learning\naliases: [ML]\n---\n# Overview\nContent.\n",
        )
        doc_query = parser.parse_file(
            Path("Query.md"),
            "# Query\nUsing [[Machne Learning]] and [[Unwritten Ghost]].\n",
        )

        engine = DiagnosticsEngine()
        engine.index_document(doc_ml)
        engine.index_document(doc_query)

        report = engine.run([doc_query])
        self.assertTrue(len(report.diagnostics) >= 2)

        typo = next((d for d in report.diagnostics if "Machne Learning" in d.message), None)
        self.assertIsNotNone(typo)
        self.assertEqual(typo.suggestion, "[[Machine Learning]]")


class TestQueryClassifier(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(QueryClassifier.classify("fn compute_hash()").intent, QueryIntent.CODE_SYMBOL)
        self.assertEqual(QueryClassifier.classify("struct VaultConfig").intent, QueryIntent.CODE_SYMBOL)
        self.assertEqual(QueryClassifier.classify("what is the main idea of pagerank in graphs?").intent, QueryIntent.NATURAL_LANGUAGE)
        self.assertEqual(QueryClassifier.classify("hybrid search algorithm").intent, QueryIntent.BALANCED_HYBRID)


if __name__ == "__main__":
    unittest.main()
