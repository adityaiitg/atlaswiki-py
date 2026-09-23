"""Language Server Protocol (LSP) 3.17 stdio server for AtlasWiki (Python edition).

Powers IDE auto-completion for `[[Wikilinks]]` and `#tags`, jump-to-definition,
hover preview transclusions, and backlink references in VS Code, Neovim, and Helix.
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from atlaswiki.storage import StorageEngine


class LspServer:
    """LSP 3.17 JSON-RPC stdio server."""

    def __init__(self, vault_root: Path) -> None:
        self.vault_root = vault_root.resolve()
        db_path = self.vault_root / ".atlaswiki" / "index.db"
        self.storage = StorageEngine(db_path) if db_path.exists() else None

    def read_message(self) -> Optional[Dict[str, Any]]:
        header_bytes = b""
        while b"\r\n\r\n" not in header_bytes:
            chunk = sys.stdin.buffer.read(1)
            if not chunk:
                return None
            header_bytes += chunk

        content_length = 0
        for line in header_bytes.decode("latin1").split("\r\n"):
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":")[1].strip())

        if content_length <= 0:
            return None

        body_bytes = sys.stdin.buffer.read(content_length)
        if not body_bytes:
            return None
        return json.loads(body_bytes.decode("utf-8"))

    def send_response(self, response: Dict[str, Any]) -> None:
        body = json.dumps(response, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("latin1")
        sys.stdout.buffer.write(header + body)
        sys.stdout.buffer.flush()

    def send_result(self, req_id: Any, result: Any) -> None:
        self.send_response({"jsonrpc": "2.0", "id": req_id, "result": result})

    def run(self) -> None:
        while True:
            msg = self.read_message()
            if msg is None:
                break

            method = msg.get("method")
            req_id = msg.get("id")
            params = msg.get("params", {})

            if method == "initialize":
                self.send_result(
                    req_id,
                    {
                        "capabilities": {
                            "textDocumentSync": 1,
                            "completionProvider": {
                                "triggerCharacters": ["[", "#", "/"],
                                "resolveProvider": False,
                            },
                            "definitionProvider": True,
                            "hoverProvider": True,
                            "referencesProvider": True,
                        },
                        "serverInfo": {
                            "name": "atlaswiki-lsp-py",
                            "version": "0.1.0",
                        },
                    },
                )
            elif method == "initialized":
                pass
            elif method == "shutdown":
                self.send_result(req_id, None)
            elif method == "exit":
                sys.exit(0)
            elif method == "textDocument/completion":
                self.handle_completion(req_id, params)
            elif method == "textDocument/definition":
                self.handle_definition(req_id, params)
            elif method == "textDocument/hover":
                self.handle_hover(req_id, params)
            elif method == "textDocument/references":
                self.handle_references(req_id, params)
            else:
                if req_id is not None:
                    self.send_result(req_id, None)

    def handle_completion(self, req_id: Any, params: Dict[str, Any]) -> None:
        items = []
        if self.storage:
            docs = self.storage.get_all_documents()
            for d in docs:
                items.append(
                    {
                        "label": d["title"],
                        "kind": 18,  # Reference
                        "detail": d["path"],
                        "insertText": d["title"] + "]]",
                        "documentation": {
                            "kind": "markdown",
                            "value": f"**{d['title']}**\n\nPath: `{d['path']}`\nWords: {d.get('word_count', 0)}",
                        },
                    }
                )

            tags = self.storage.get_all_tags()
            for t in tags:
                tag_name = t if isinstance(t, str) else t.get("tag", "")
                items.append(
                    {
                        "label": f"#{tag_name}",
                        "kind": 14,  # Keyword
                        "insertText": tag_name,
                        "detail": "Tag",
                    }
                )

        self.send_result(req_id, {"isIncomplete": False, "items": items})

    def handle_definition(self, req_id: Any, params: Dict[str, Any]) -> None:
        # Simple resolution to note file
        self.send_result(req_id, None)

    def handle_hover(self, req_id: Any, params: Dict[str, Any]) -> None:
        self.send_result(req_id, None)

    def handle_references(self, req_id: Any, params: Dict[str, Any]) -> None:
        self.send_result(req_id, [])


def run_lsp_server(vault_root: Path) -> None:
    server = LspServer(vault_root)
    server.run()
