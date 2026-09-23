"""Cross-platform binary SQLite WAL interoperability tests between Rust and Python editions."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from atlaswiki.storage import StorageEngine


class TestCrossPlatformParity(unittest.TestCase):
    def setUp(self):
        self.rust_bin = Path(
            "/Volumes/T7/Personal_MAC_DATA/Personal/github/atlaswiki/target/debug/atlaswiki"
        )

    def test_python_reads_database_indexed_by_rust(self):
        if not self.rust_bin.exists():
            self.skipTest("Rust atlaswiki binary not built")

        with tempfile.TemporaryDirectory() as tmp_dir:
            vault = Path(tmp_dir)
            # Create sample markdown files
            (vault / "Alpha.md").write_text(
                "---\ntags: [research, ai]\n---\n# Note Alpha\nReferences [[Beta]] and [[Gamma]].\n",
                encoding="utf-8",
            )
            (vault / "Beta.md").write_text(
                "# Note Beta\nReferences [[Gamma]].\n",
                encoding="utf-8",
            )
            (vault / "Gamma.md").write_text(
                "# Note Gamma\nTerminal note with details about neural networks.\n",
                encoding="utf-8",
            )

            # 1. Index with Rust binary
            res = subprocess.run(
                [str(self.rust_bin), "index", str(vault)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, f"Rust index failed: {res.stderr}")

            # 2. Verify .atlaswiki/index.db exists
            db_path = vault / ".atlaswiki" / "index.db"
            self.assertTrue(db_path.exists())

            # 3. Read directly with Python StorageEngine
            storage = StorageEngine.open(db_path)
            stats = storage.get_stats()
            self.assertEqual(stats.total_documents, 3)
            self.assertEqual(stats.total_links, 3)

            # 4. Check backlinks in Python
            gamma_bl = storage.get_backlinks("Gamma")
            bl_sources = {b.source_title for b in gamma_bl}
            self.assertEqual(bl_sources, {"Note Alpha", "Note Beta"})

            # 5. Check FTS search in Python
            hits = storage.search_fts("neural networks", limit=5)
            self.assertTrue(len(hits) > 0)
            self.assertEqual(hits[0][1], "Note Gamma")

    def test_rust_reads_database_indexed_by_python(self):
        if not self.rust_bin.exists():
            self.skipTest("Rust atlaswiki binary not built")

        with tempfile.TemporaryDirectory() as tmp_dir:
            vault = Path(tmp_dir)
            # Create sample markdown files
            (vault / "Delta.md").write_text(
                "# Note Delta\nReferences [[Epsilon]].\n",
                encoding="utf-8",
            )
            (vault / "Epsilon.md").write_text(
                "# Note Epsilon\nContains quantum computing algorithms.\n",
                encoding="utf-8",
            )

            # 1. Index with Python CLI engine
            from atlaswiki.cli import cmd_index

            cmd_index(vault, full=True)

            # 2. Query with Rust CLI: search
            res = subprocess.run(
                [
                    str(self.rust_bin),
                    "-C",
                    str(vault),
                    "search",
                    "quantum computing",
                    "--json",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, f"Rust search failed: {res.stderr}")
            self.assertIn("Note Epsilon", res.stdout)

            # 3. Query with Rust CLI: backlinks
            res_bl = subprocess.run(
                [
                    str(self.rust_bin),
                    "-C",
                    str(vault),
                    "backlinks",
                    "Epsilon",
                    "--json",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                res_bl.returncode, 0, f"Rust backlinks failed: {res_bl.stderr}"
            )
            self.assertIn("Note Delta", res_bl.stdout)


if __name__ == "__main__":
    unittest.main()
