import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlaswiki.graph import KnowledgeGraph
from atlaswiki.parser import MarkdownParser
from atlaswiki.storage import StorageEngine


def list_tools() -> List[Dict[str, Any]]:
    return [
        {
            "name": "atlaswiki_search",
            "description": "Perform full-text BM25 and semantic hybrid search across notes and chunks in the AtlasWiki knowledge base.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, phrase, or keyword.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of search results to return (default: 10).",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "atlaswiki_get_note",
            "description": "Retrieve full markdown content, frontmatter metadata, sections hierarchy, outgoing links, and incoming backlinks for a given note.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title or relative path of the note to retrieve.",
                    }
                },
                "required": ["title"],
            },
        },
        {
            "name": "atlaswiki_backlinks",
            "description": "Get all incoming backlinks referencing a note, including referencing files, line numbers, and context snippets.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the note to inspect incoming references for.",
                    }
                },
                "required": ["title"],
            },
        },
        {
            "name": "atlaswiki_graph_neighbors",
            "description": "Retrieve the local subgraph (1-hop or 2-hop connected neighbors and wikilink connections) around a given note.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The center note title.",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Hop distance (1 or 2, default: 1).",
                    },
                },
                "required": ["title"],
            },
        },
        {
            "name": "atlaswiki_unresolved_links",
            "description": "List all unresolved or dangling wikilinks in the vault sorted by reference frequency (wanted unwritten notes).",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "atlaswiki_stats",
            "description": "Get comprehensive metrics for the AtlasWiki knowledge vault (notes count, sections, links, chunks, words).",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
    ]


def call_tool(
    name: str, args: Dict[str, Any], vault_root: Path, storage: StorageEngine
) -> str:
    if name == "atlaswiki_search":
        query = args.get("query", "")
        limit = int(args.get("limit", 10))
        raw_hits = storage.search_fts(query, limit)
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
        return json.dumps(hits, indent=2)

    elif name == "atlaswiki_get_note":
        title = args.get("title", "")
        doc = storage.get_document(title)
        if not doc:
            raise ValueError(f"Note '{title}' not found in vault index")

        full_p = vault_root / doc["path"]
        content = full_p.read_text(encoding="utf-8") if full_p.is_file() else ""
        sections = storage.get_sections_for_doc(doc["doc_id"])
        outlinks = storage.get_outlinks_for_doc(doc["doc_id"])
        backlinks = [
            {
                "source_title": b.source_title,
                "source_path": b.source_path,
                "line_number": b.line_number,
                "snippet": b.snippet,
            }
            for b in storage.get_backlinks(doc["title"])
        ]

        payload = {
            "document": doc,
            "sections": sections,
            "outlinks": outlinks,
            "backlinks": backlinks,
            "content": content,
        }
        return json.dumps(payload, indent=2)

    elif name == "atlaswiki_backlinks":
        title = args.get("title", "")
        backlinks = [
            {
                "source_title": b.source_title,
                "source_path": b.source_path,
                "line_number": b.line_number,
                "snippet": b.snippet,
            }
            for b in storage.get_backlinks(title)
        ]
        return json.dumps(backlinks, indent=2)

    elif name == "atlaswiki_graph_neighbors":
        title = args.get("title", "")
        parser = MarkdownParser()
        paths = storage.get_all_document_paths()
        docs = []

        for p in paths:
            full_p = vault_root / p
            if full_p.is_file():
                try:
                    text = full_p.read_text(encoding="utf-8")
                    docs.append(parser.parse_file(Path(p), text))
                except Exception:
                    pass

        kg = KnowledgeGraph.from_documents(docs)
        in_links = [e.source_title for e in kg.edges if e.target_title.lower() == title.lower()]
        out_links = [e.target_title for e in kg.edges if e.source_title.lower() == title.lower()]

        payload = {
            "center_note": title,
            "incoming_links": in_links,
            "outgoing_links": out_links,
        }
        return json.dumps(payload, indent=2)

    elif name == "atlaswiki_unresolved_links":
        records = [
            {
                "target_note": r.target_note,
                "reference_count": r.reference_count,
                "sources": r.source_notes,
            }
            for r in storage.get_unresolved_links()
        ]
        return json.dumps(records, indent=2)

    elif name == "atlaswiki_stats":
        s = storage.get_stats()
        payload = {
            "total_documents": s.total_documents,
            "total_sections": s.total_sections,
            "total_links": s.total_links,
            "total_tags": s.total_tags,
            "total_chunks": s.total_chunks,
            "total_embeddings": s.total_embeddings,
            "total_words": s.total_words,
        }
        return json.dumps(payload, indent=2)

    else:
        raise ValueError(f"Unknown tool: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AtlasWiki Model Context Protocol (MCP) Server")
    parser.add_argument("-C", "--vault", default=".", help="Path to markdown vault directory")
    args = parser.parse_args()

    vault_root = Path(args.vault).resolve()
    db_path = vault_root / ".atlaswiki" / "index.db"

    sys.stderr.write(f"[atlaswiki-mcp-py] Starting server for vault: {vault_root}\n")
    sys.stderr.flush()

    storage = StorageEngine.open(db_path)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}

        if method == "initialize":
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "atlaswiki-mcp-py", "version": "0.1.0"},
                },
            }
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()

        elif method == "notifications/initialized":
            pass

        elif method == "ping":
            res = {"jsonrpc": "2.0", "id": req_id, "result": {}}
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()

        elif method == "tools/list":
            tools = list_tools()
            res = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments") or {}

            try:
                content = call_tool(tool_name, arguments, vault_root, storage)
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": content}]
                    },
                }
            except Exception as e:
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": str(e)},
                }

            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()

        else:
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
