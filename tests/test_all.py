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

            # Documents and tags helpers
            docs = storage.get_all_documents()
            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0]["title"], "Note A")

            # Node embeddings persistence and retrieval
            sample_vec = [0.1, 0.2, 0.3, 0.4]
            storage.save_node_embeddings([("Note A", sample_vec)], model="test_model")
            retrieved = storage.get_node_embedding("Note A", model="test_model")
            self.assertIsNotNone(retrieved)
            self.assertEqual(len(retrieved), 4)
            self.assertAlmostEqual(retrieved[0], 0.1, places=5)

            all_embs = storage.get_all_node_embeddings(model="test_model")
            self.assertIn("Note A", all_embs)
            self.assertEqual(len(all_embs["Note A"]), 4)


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


class TestQuantizationAndHnsw(unittest.TestCase):
    def test_sq8_quantization_and_recall(self):
        from atlaswiki.quantization import Sq8Vector
        from atlaswiki.hnsw import HnswIndex

        v1 = [0.1 * i for i in range(32)]
        v2 = [0.1 * (i + 1) for i in range(32)]
        sq1 = Sq8Vector.from_float(v1)
        sq2 = Sq8Vector.from_float(v2)

        # Dequantize
        f1 = sq1.to_float()
        self.assertEqual(len(f1), 32)

        # Cosine similarity error
        sim_q = sq1.cosine_similarity(sq2)
        self.assertTrue(sim_q > 0.95)

        # HNSW test
        index = HnswIndex(m=8, ef_construction=32)
        for idx in range(20):
            vec = [float(idx + j) for j in range(16)]
            index.insert(idx, vec)

        results = index.search(v1[:16], k=3)
        self.assertTrue(len(results) > 0)


class TestGraphRagAndSynthesis(unittest.TestCase):
    def test_graph_rag_connective_context(self):
        from atlaswiki.graph_rag import GraphRagEngine

        parser = MarkdownParser()
        doc_a = parser.parse_file(Path("A.md"), "# Note A\nConnects to [[Note B]]")
        doc_b = parser.parse_file(Path("B.md"), "# Note B\nConnects to [[Note C]]")
        doc_c = parser.parse_file(Path("C.md"), "# Note C\nTerminal node")

        engine = GraphRagEngine.from_documents([doc_a, doc_b, doc_c])
        res = engine.extract_context(["Note A", "Note C"], max_hops=3, max_paths=5)
        self.assertTrue(len(res.paths) > 0)
        self.assertIn("[[Note A]]", res.markdown_context)
        self.assertIn("[[Note C]]", res.markdown_context)

    def test_synthesis_idempotency(self):
        from atlaswiki.synthesis import update_synthesis_content

        user_content = "# My Manual Notes\nUser line 1\n<!-- ATLASWIKI:BEGIN_SYNTHESIS -->\nOld Index\n<!-- ATLASWIKI:END_SYNTHESIS -->\n## User section 2\nKeep this!"
        new_synthesis = "New Synthesized Index 2.0"
        updated = update_synthesis_content(user_content, new_synthesis)

        self.assertIn("User line 1", updated)
        self.assertIn("Keep this!", updated)
        self.assertIn("New Synthesized Index 2.0", updated)
        self.assertNotIn("Old Index", updated)


class TestMathAndDataview(unittest.TestCase):
    def test_math_isolation_and_dataview_fields(self):
        parser = MarkdownParser()
        md = """# Formula Note
Here is inline math: $E = mc^2$ and $x_i #not_a_tag$.
And block math:
$$
\\begin{matrix} 1 & 0 \\\\ 0 & 1 \\end{matrix} #also_not_a_tag [[NotALink]]
$$
Dataview fields:
[status:: active]
priority:: high
"""
        doc = parser.parse_file(Path("Math.md"), md)
        # Verify no false tags or links extracted from math
        tag_names = [t.name for t in doc.tags]
        self.assertNotIn("not_a_tag", tag_names)
        self.assertNotIn("also_not_a_tag", tag_names)
        link_targets = [l.target_note for l in doc.links]
        self.assertNotIn("NotALink", link_targets)

        # Verify Dataview attributes
        attr_keys = {a.key: a.value for a in doc.attributes}
        self.assertEqual(attr_keys.get("status"), "active")
        self.assertEqual(attr_keys.get("priority"), "high")


if __name__ == "__main__":
    unittest.main()
