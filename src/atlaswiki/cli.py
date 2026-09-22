import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlaswiki.diagnostics import DiagnosticsEngine, DiagnosticSeverity
from atlaswiki.graph import KnowledgeGraph
from atlaswiki.parser import MarkdownParser
from atlaswiki.retrieval import HybridRetriever
from atlaswiki.serve import run_server
from atlaswiki.storage import StorageEngine


def get_storage(vault_path: Path) -> StorageEngine:
    db_path = vault_path / ".atlaswiki" / "index.db"
    return StorageEngine.open(db_path)


def cmd_index(vault_path: Path, full: bool = False) -> None:
    t0 = time.perf_counter()
    print(f"\033[1;36m⚙\033[0m Indexing vault at \033[1m{vault_path}\033[0m")

    storage = get_storage(vault_path)
    parser = MarkdownParser()

    # Discover all markdown files
    files = []
    for p in vault_path.rglob("*.md"):
        rel = p.relative_to(vault_path)
        rel_str = str(rel)

        if (
            rel_str.startswith(".atlaswiki")
            or rel_str.startswith(".git")
            or rel_str.startswith(".obsidian")
            or "node_modules" in rel_str
        ):
            continue

        files.append((p, rel))

    print(f"  Discovered \033[1;36m{len(files)}\033[0m markdown notes.")

    # Incremental filtering
    to_process = []
    for abs_p, rel_p in files:
        stat = abs_p.stat()
        mtime_ms = int(stat.st_mtime * 1000)
        size = stat.st_size
        rel_str = str(rel_p)

        if not full:
            manifest = storage.get_manifest(rel_str)
            if manifest:
                stored_hash, stored_mtime, stored_size = manifest
                if stored_mtime == mtime_ms and stored_size == size:
                    continue

        try:
            content = abs_p.read_text(encoding="utf-8")
            to_process.append((abs_p, rel_str, content, mtime_ms, size))
        except Exception:
            pass

    if not to_process:
        print("\033[1;32m✓\033[0m All notes are up to date. Zero changes detected.")
        return

    print(f"  Parsing & indexing \033[1;33m{len(to_process)}\033[0m modified notes...")

    for abs_p, rel_str, content, mtime_ms, size in to_process:
        doc = parser.parse_file(Path(rel_str), content)
        doc_id = f"doc_{doc.content_hash[:16]}"
        storage.sync_document(doc, doc_id, rel_str, mtime_ms, size)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    stats = storage.get_stats()

    print(f"\n\033[1;32m✓\033[0m Indexed \033[1m{len(to_process)}\033[0m notes in \033[1m{elapsed_ms:.1f}ms\033[0m")
    print(f"  Total Documents: \033[1;36m{stats.total_documents}\033[0m")
    print(f"  Total Sections:  \033[1;36m{stats.total_sections}\033[0m")
    print(f"  Total Links:     \033[1;36m{stats.total_links}\033[0m")
    print(f"  Total Chunks:    \033[1;36m{stats.total_chunks}\033[0m")
    print(f"  Total Words:     \033[1;36m{stats.total_words}\033[0m")


def cmd_search(
    vault_path: Path, query: str, limit: int = 10, as_json: bool = False, tag: Optional[str] = None
) -> None:
    storage = get_storage(vault_path)
    retriever = HybridRetriever(storage)
    results = retriever.search(query, limit=limit, tag_filter=tag)

    if as_json:
        print(json.dumps(results, indent=2))
        return

    if not results:
        print(f"\033[1;33mℹ\033[0m No results found for query: '{query}'")
        return

    print(f"\n\033[1;32m✓\033[0m Found \033[1m{len(results)}\033[0m results for '\033[1;36m{query}\033[0m':\n")
    for i, hit in enumerate(results, start=1):
        print(f"{i}. \033[1;36m{hit['title']}\033[0m \033[2m({hit['breadcrumbs']})\033[0m")
        clean_snippet = hit["snippet"].replace("<b>", "\033[1;33m").replace("</b>", "\033[0m")
        print(f"   {clean_snippet.strip()}\n")


def cmd_note(vault_path: Path, title: str, as_json: bool = False) -> None:
    storage = get_storage(vault_path)
    doc = storage.get_document(title)
    if not doc:
        print(f"\033[1;31m✗\033[0m Note '{title}' not found in vault index.", file=sys.stderr)
        sys.exit(1)

    sections = storage.get_sections_for_doc(doc["doc_id"])
    outlinks = storage.get_outlinks_for_doc(doc["doc_id"])
    backlinks = storage.get_backlinks(doc["title"])

    if as_json:
        payload = {
            "document": doc,
            "sections": sections,
            "outlinks": outlinks,
            "backlinks": [
                {
                    "source_title": b.source_title,
                    "source_path": b.source_path,
                    "line_number": b.line_number,
                    "snippet": b.snippet,
                }
                for b in backlinks
            ],
        }
        print(json.dumps(payload, indent=2))
        return

    print(f"\n\033[1mNote:\033[0m \033[1;36m{doc['title']}\033[0m")
    print(f"  Path:        \033[2m{doc['path']}\033[0m")
    print(f"  Word Count:  {doc['word_count']}")

    if sections:
        print(f"\n\033[1mSections:\033[0m")
        for s in sections:
            indent = "  " * s["level"]
            print(f"{indent}- \033[1m{s['heading']}\033[0m (lines {s['line_start']}-{s['line_end']})")

    if outlinks:
        print(f"\n\033[1mOutgoing Links:\033[0m")
        for l in outlinks:
            target = l["target_note"]
            h = f"#{l['target_heading']}" if l["target_heading"] else ""
            print(f"  → [[\033[1;36m{target}{h}\033[0m]] (line {l['line_number']})")

    if backlinks:
        print(f"\n\033[1mIncoming Backlinks:\033[0m")
        for b in backlinks:
            print(f"  ← [[\033[1;33m{b.source_title}\033[0m]] (line {b.line_number})")
            if b.snippet:
                print(f"    \033[3;2m\"{b.snippet}\"\033[0m")

    print()


def cmd_backlinks(vault_path: Path, title: str, as_json: bool = False) -> None:
    storage = get_storage(vault_path)
    backlinks = storage.get_backlinks(title)

    if as_json:
        payload = [
            {
                "source_title": b.source_title,
                "source_path": b.source_path,
                "line_number": b.line_number,
                "snippet": b.snippet,
            }
            for b in backlinks
        ]
        print(json.dumps(payload, indent=2))
        return

    if not backlinks:
        print(f"\033[1;33mℹ\033[0m No backlinks found pointing to [[\033[1m{title}\033[0m]]")
        return

    print(f"\n\033[1;32m✓\033[0m \033[1m{len(backlinks)}\033[0m backlinks found pointing to [[\033[1;36m{title}\033[0m]]:\n")
    for b in backlinks:
        print(f"  • [[\033[1;36m{b.source_title}\033[0m]] \033[2m{b.source_path}\033[0m (line {b.line_number})")
        if b.snippet:
            print(f"    \033[3;2m\"{b.snippet.strip()}\"\033[0m")
    print()


def cmd_graph(vault_path: Path, subcmd: str, as_json: bool = False, from_note: str = "", to_note: str = "") -> None:
    storage = get_storage(vault_path)
    parser = MarkdownParser()
    paths = storage.get_all_document_paths()
    docs = []

    for p in paths:
        full_p = vault_path / p
        if full_p.is_file():
            try:
                docs.append(parser.parse_file(Path(p), full_p.read_text(encoding="utf-8")))
            except Exception:
                pass

    kg = KnowledgeGraph.from_documents(docs)
    kg.compute_pagerank(0.85, 50)

    if subcmd == "stats":
        stats = storage.get_stats()
        if as_json:
            print(json.dumps({
                "documents": stats.total_documents,
                "sections": stats.total_sections,
                "links": stats.total_links,
                "tags": stats.total_tags,
                "chunks": stats.total_chunks,
                "words": stats.total_words,
            }, indent=2))
        else:
            print(f"\n\033[1mVault Knowledge Graph Metrics:\033[0m")
            print(f"  Documents:   \033[1;36m{stats.total_documents}\033[0m")
            print(f"  Sections:    \033[1;36m{stats.total_sections}\033[0m")
            print(f"  Connections: \033[1;36m{stats.total_links}\033[0m")
            print(f"  Tags:        \033[1;36m{stats.total_tags}\033[0m")
            print(f"  Chunks:      \033[1;36m{stats.total_chunks}\033[0m")
            print(f"  Total Words: \033[1;36m{stats.total_words}\033[0m\n")

    elif subcmd == "orphans":
        orphans = kg.get_orphans()
        if as_json:
            print(json.dumps([o.title for o in orphans], indent=2))
        else:
            print(f"\n\033[1;33mℹ\033[0m \033[1mIsolated Orphan Notes (0 in, 0 out):\033[0m\n")
            if not orphans:
                print("  No orphan notes detected in vault.")
            else:
                for o in orphans:
                    print(f"  • [[\033[1;33m{o.title}\033[0m]]")
            print()

    elif subcmd == "wanted":
        wanted = kg.get_wanted_pages()
        if as_json:
            print(json.dumps([{"target": t, "count": c} for t, c in wanted], indent=2))
        else:
            print(f"\n\033[1;33m⚡\033[0m \033[1mWanted / Dangling Pages (Unwritten Notes):\033[0m\n")
            if not wanted:
                print("  All wikilinks resolve to existing notes.")
            else:
                for target, count in wanted:
                    print(f"  • [[\033[1;31m{target}\033[0m]] (referenced \033[1m{count}\033[0m times)")
            print()

    elif subcmd == "path":
        path = kg.shortest_path(from_note, to_note)
        if as_json:
            print(json.dumps(path, indent=2))
        else:
            print(f"\n\033[1;36m⚲\033[0m \033[1mShortest Path from [[\033[1;36m{from_note}\033[0m]] to [[\033[1;36m{to_note}\033[0m]]:\033[0m\n")
            if path:
                formatted = "  →  ".join(f"[[\033[1;36m{n}\033[0m]]" for n in path)
                print(f"  {formatted}")
            else:
                print(f"  No connected path found between [[{from_note}]] and [[{to_note}]].")
            print()


def cmd_check(vault_path: Path, strict: bool = False, as_json: bool = False) -> None:
    parser = MarkdownParser()
    diagnostics = DiagnosticsEngine()
    docs = []

    for p in vault_path.rglob("*.md"):
        rel = p.relative_to(vault_path)
        rel_str = str(rel)
        if rel_str.startswith(".atlaswiki") or rel_str.startswith(".git") or rel_str.startswith(".obsidian"):
            continue

        try:
            doc = parser.parse_file(rel, p.read_text(encoding="utf-8"))
            diagnostics.index_document(doc)
            docs.append(doc)
        except Exception:
            pass

    report = diagnostics.run(docs, strict_mode=strict)

    if as_json:
        payload = {
            "total_files_scanned": report.total_files_scanned,
            "total_links_checked": report.total_links_checked,
            "error_count": report.error_count(),
            "warning_count": report.warning_count(),
            "diagnostics": [
                {
                    "code": d.code.value,
                    "severity": d.severity.value,
                    "message": d.message,
                    "file_path": str(d.location.file_path),
                    "line_number": d.location.line_number,
                    "suggestion": d.suggestion,
                }
                for d in report.diagnostics
            ],
        }
        print(json.dumps(payload, indent=2))
        if strict and report.error_count() > 0:
            sys.exit(1)
        return

    print(
        f"\n\033[1;32m✓\033[0m Scanned \033[1m{report.total_files_scanned}\033[0m notes, checked \033[1m{report.total_links_checked}\033[0m links.\n"
    )

    if not report.diagnostics:
        print("\033[1;32m✓\033[0m Vault is completely healthy! Zero dead links or anchors found.\n")
        return

    for diag in report.diagnostics:
        badge = (
            f"\033[1;31merror[{diag.code.value}]\033[0m"
            if diag.severity == DiagnosticSeverity.ERROR
            else f"\033[1;33mwarning[{diag.code.value}]\033[0m"
        )
        print(f"{badge}: \033[1m{diag.location.file_path}\033[0m:{diag.location.line_number} - {diag.message}")
        if diag.location.source_line:
            print(f"   | \033[2m{diag.location.source_line}\033[0m")
        if diag.suggestion:
            print(f"   = \033[1;36mhelp: did you mean:\033[0m \033[1;32m{diag.suggestion}\033[0m\n")
        else:
            print()

    errs = report.error_count()
    warns = report.warning_count()
    print(f"Result: \033[1;31m{errs} errors\033[0m, \033[1;33m{warns} warnings\033[0m found.\n")

    if strict and errs > 0:
        sys.exit(1)


def main() -> None:
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("-C", "--vault", default=".", help="Path to markdown vault directory")

    parser = argparse.ArgumentParser(
        prog="atlaswiki",
        parents=[common_parser],
        description="High-performance Markdown wiki indexer & semantic search engine (Python edition)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # index
    p_index = subparsers.add_parser("index", parents=[common_parser], help="Index markdown vault into SQLite database")
    p_index.add_argument("path", nargs="?", default=None, help="Path to vault directory")
    p_index.add_argument("--full", action="store_true", help="Force full re-indexing")

    # search
    p_search = subparsers.add_parser("search", parents=[common_parser], help="Search vault notes")
    p_search.add_argument("query", help="Search query string")
    p_search.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    p_search.add_argument("--json", action="store_true", help="Output JSON")
    p_search.add_argument("--tag", default=None, help="Filter by tag")

    # note
    p_note = subparsers.add_parser("note", parents=[common_parser], help="Inspect note details")
    p_note.add_argument("title", help="Note title or path")
    p_note.add_argument("--json", action="store_true", help="Output JSON")

    # backlinks
    p_bl = subparsers.add_parser("backlinks", parents=[common_parser], help="List incoming backlinks")
    p_bl.add_argument("title", help="Note title")
    p_bl.add_argument("--json", action="store_true", help="Output JSON")

    # graph
    p_graph = subparsers.add_parser("graph", parents=[common_parser], help="Knowledge graph analysis")
    p_graph.add_argument("subcommand", choices=["stats", "orphans", "wanted", "path"])
    p_graph.add_argument("from_note", nargs="?", default="", help="Source note for path")
    p_graph.add_argument("to_note", nargs="?", default="", help="Target note for path")
    p_graph.add_argument("--json", action="store_true", help="Output JSON")

    # check
    p_check = subparsers.add_parser("check", parents=[common_parser], help="Link diagnostics and typo checking")
    p_check.add_argument("path", nargs="?", default=None, help="Path to vault")
    p_check.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    p_check.add_argument("--json", action="store_true", help="Output JSON")

    # serve
    p_serve = subparsers.add_parser("serve", parents=[common_parser], help="Launch interactive D3.js web visualizer")
    p_serve.add_argument("path", nargs="?", default=None, help="Path to vault")
    p_serve.add_argument("-p", "--port", type=int, default=8888, help="Port to bind")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    vault_dir = Path(args.path or args.vault).resolve()

    if args.command == "index":
        cmd_index(vault_dir, full=args.full)
    elif args.command == "search":
        cmd_search(vault_dir, args.query, limit=args.limit, as_json=args.json, tag=args.tag)
    elif args.command == "note":
        cmd_note(vault_dir, args.title, as_json=args.json)
    elif args.command == "backlinks":
        cmd_backlinks(vault_dir, args.title, as_json=args.json)
    elif args.command == "graph":
        cmd_graph(vault_dir, args.subcommand, as_json=args.json, from_note=args.from_note, to_note=args.to_note)
    elif args.command == "check":
        cmd_check(vault_dir, strict=args.strict, as_json=args.json)
    elif args.command == "serve":
        storage = get_storage(vault_dir)
        run_server(vault_dir, storage, port=args.port)


if __name__ == "__main__":
    main()
