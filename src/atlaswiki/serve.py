from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, unquote, urlparse

from atlaswiki.graph import KnowledgeGraph
from atlaswiki.parser import MarkdownParser
from atlaswiki.storage import StorageEngine

VIEWER_HTML_PATH = Path(__file__).parent / "assets" / "viewer.html"


class GraphServerHandler(BaseHTTPRequestHandler):
    storage: StorageEngine
    vault_path: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            self._serve_viewer()
        elif parsed.path == "/api/graph":
            self._serve_graph()
        elif parsed.path == "/api/stats":
            self._serve_stats()
        elif parsed.path == "/api/backlinks":
            title = params.get("title", [""])[0]
            self._serve_backlinks(title)
        elif parsed.path == "/api/note":
            title = params.get("title", [""])[0]
            self._serve_note(title)
        elif parsed.path == "/api/search":
            q = params.get("q", [""])[0]
            limit = int(params.get("limit", [20])[0])
            self._serve_search(q, limit)
        else:
            self.send_error(404, "Not Found")

    def _serve_viewer(self) -> None:
        if VIEWER_HTML_PATH.exists():
            content = VIEWER_HTML_PATH.read_bytes()
        else:
            content = b"<h1>AtlasWiki Visualizer: viewer.html missing</h1>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_graph(self) -> None:
        parser = MarkdownParser()
        paths = self.storage.get_all_document_paths()
        docs = []

        for p in paths:
            full_p = self.vault_path / p
            if full_p.is_file():
                try:
                    text = full_p.read_text(encoding="utf-8")
                    docs.append(parser.parse_file(Path(p), text))
                except Exception:
                    pass

        kg = KnowledgeGraph.from_documents(docs)
        kg.compute_pagerank(0.85, 30)
        data = kg.to_d3_json()
        self._send_json(data)

    def _serve_stats(self) -> None:
        stats = self.storage.get_stats()
        self._send_json(
            {
                "total_documents": stats.total_documents,
                "total_sections": stats.total_sections,
                "total_links": stats.total_links,
                "total_tags": stats.total_tags,
                "total_chunks": stats.total_chunks,
                "total_embeddings": stats.total_embeddings,
                "total_words": stats.total_words,
            }
        )

    def _serve_backlinks(self, title: str) -> None:
        records = self.storage.get_backlinks(title)
        data = [
            {
                "source_doc_id": r.source_doc_id,
                "source_path": r.source_path,
                "source_title": r.source_title,
                "line_number": r.line_number,
                "link_type": r.link_type,
                "target_heading": r.target_heading,
                "target_block": r.target_block,
                "alias": r.alias,
                "snippet": r.snippet,
            }
            for r in records
        ]
        self._send_json(data)

    def _serve_note(self, title: str) -> None:
        doc = self.storage.get_document(title)
        if not doc:
            self.send_error(404, "Note not found")
            return

        full_p = self.vault_path / doc["path"]
        content = full_p.read_text(encoding="utf-8") if full_p.is_file() else ""

        self._send_json(
            {
                "title": doc["title"],
                "path": doc["path"],
                "word_count": doc["word_count"],
                "content": content,
            }
        )

    def _serve_search(self, q: str, limit: int) -> None:
        raw_hits = self.storage.search_fts(q, limit)
        hits = [
            {
                "chunk_id": cid,
                "title": title,
                "breadcrumbs": breadcrumbs,
                "snippet": snippet,
                "score": score,
            }
            for cid, title, breadcrumbs, snippet, score in raw_hits
        ]
        self._send_json(hits)

    def _send_json(self, data: Any) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy HTTP request logging in terminal
        pass


def run_server(vault_path: Path, storage: StorageEngine, port: int = 8888) -> None:
    handler = type("ConfiguredHandler", (GraphServerHandler,), {"storage": storage, "vault_path": vault_path})
    server = HTTPServer(("127.0.0.1", port), handler)
    print(f"\033[1;32m✓\033[0m AtlasWiki graph visualizer running at \033[1;36mhttp://127.0.0.1:{port}\033[0m")
    print("\033[2mPress Ctrl+C to stop the server.\n\033[0m")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web visualizer.")
    finally:
        server.server_close()
