from __future__ import annotations

import json
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from atlaswiki.ast import ParsedDocument


@dataclass
class VaultStats:
    total_documents: int
    total_sections: int
    total_links: int
    total_tags: int
    total_chunks: int
    total_embeddings: int
    total_words: int


@dataclass
class BacklinkRecord:
    source_doc_id: str
    source_path: str
    source_title: str
    line_number: int
    link_type: str
    target_heading: Optional[str]
    target_block: Optional[str]
    alias: Optional[str]
    snippet: Optional[str]


@dataclass
class UnresolvedLinkRecord:
    target_note: str
    reference_count: int
    source_notes: List[str]


class StorageEngine:
    """SQLite WAL storage engine with FTS5 virtual tables and synchronization triggers."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        if db_path != Path(":memory:"):
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            timeout=10.0,
        )
        self.conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        self._init_schema()

    @classmethod
    def open(cls, db_path: Path) -> StorageEngine:
        return cls(db_path)

    @classmethod
    def open_in_memory(cls) -> StorageEngine:
        return cls(Path(":memory:"))

    def _apply_pragmas(self) -> None:
        if self.db_path != Path(":memory:"):
            self.conn.execute("PRAGMA journal_mode = WAL;")
            self.conn.execute("PRAGMA synchronous = NORMAL;")
            self.conn.execute("PRAGMA wal_autocheckpoint = 1000;")

        self.conn.execute("PRAGMA mmap_size = 268435456;")
        self.conn.execute("PRAGMA cache_size = -64000;")
        self.conn.execute("PRAGMA temp_store = MEMORY;")
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA recursive_triggers = ON;")
        self.conn.execute("PRAGMA busy_timeout = 5000;")

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id              TEXT PRIMARY KEY NOT NULL,
                    path                TEXT NOT NULL UNIQUE,
                    title               TEXT NOT NULL,
                    frontmatter_json    TEXT,
                    word_count          INTEGER NOT NULL DEFAULT 0,
                    mtime               INTEGER NOT NULL,
                    hash                TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sections (
                    section_id          TEXT PRIMARY KEY NOT NULL,
                    doc_id              TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    heading             TEXT NOT NULL,
                    level               INTEGER NOT NULL,
                    parent_id           TEXT REFERENCES sections(section_id) ON DELETE SET NULL,
                    breadcrumbs         TEXT NOT NULL,
                    line_start          INTEGER NOT NULL,
                    line_end            INTEGER NOT NULL,
                    content             TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS links (
                    link_id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    target_note         TEXT NOT NULL,
                    target_heading      TEXT,
                    target_block        TEXT,
                    alias               TEXT,
                    link_type           TEXT NOT NULL,
                    line_number         INTEGER NOT NULL,
                    context_snippet     TEXT,
                    source_section_id   TEXT REFERENCES sections(section_id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS tags (
                    tag_id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id              TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    tag                 TEXT NOT NULL,
                    section_id          TEXT REFERENCES sections(section_id) ON DELETE SET NULL,
                    line_number         INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id            TEXT PRIMARY KEY NOT NULL,
                    doc_id              TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    section_id          TEXT REFERENCES sections(section_id) ON DELETE SET NULL,
                    title               TEXT NOT NULL,
                    breadcrumbs         TEXT NOT NULL,
                    line_start          INTEGER NOT NULL,
                    line_end            INTEGER NOT NULL,
                    content             TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    title,
                    breadcrumbs,
                    content,
                    content='chunks',
                    content_rowid='rowid',
                    tokenize='porter unicode61 remove_diacritics 2'
                );

                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, chunk_id, title, breadcrumbs, content)
                    VALUES (new.rowid, new.chunk_id, new.title, new.breadcrumbs, new.content);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, title, breadcrumbs, content)
                    VALUES ('delete', old.rowid, old.chunk_id, old.title, old.breadcrumbs, old.content);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, title, breadcrumbs, content)
                    VALUES ('delete', old.rowid, old.chunk_id, old.title, old.breadcrumbs, old.content);
                    INSERT INTO chunks_fts(rowid, chunk_id, title, breadcrumbs, content)
                    VALUES (new.rowid, new.chunk_id, new.title, new.breadcrumbs, new.content);
                END;

                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id            TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                    embedding           BLOB NOT NULL,
                    dimensions          INTEGER NOT NULL,
                    model               TEXT NOT NULL,
                    created_at          INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
                );

                CREATE TABLE IF NOT EXISTS node_embeddings (
                    node_title          TEXT PRIMARY KEY NOT NULL,
                    embedding           BLOB NOT NULL,
                    dimensions          INTEGER NOT NULL,
                    model               TEXT NOT NULL,
                    updated_at          INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
                );

                CREATE TABLE IF NOT EXISTS sync_manifest (
                    path                TEXT PRIMARY KEY NOT NULL,
                    doc_id              TEXT NOT NULL,
                    mtime               INTEGER NOT NULL,
                    file_size           INTEGER NOT NULL,
                    hash                TEXT NOT NULL,
                    synced_at           INTEGER NOT NULL,
                    status              TEXT NOT NULL DEFAULT 'synced'
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
                CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
                CREATE INDEX IF NOT EXISTS idx_sections_doc_id ON sections(doc_id);
                CREATE INDEX IF NOT EXISTS idx_links_source_doc ON links(source_doc_id);
                CREATE INDEX IF NOT EXISTS idx_links_target_note ON links(target_note);
                CREATE INDEX IF NOT EXISTS idx_tags_doc_id ON tags(doc_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_node_embeddings_model ON node_embeddings(model);
                CREATE INDEX IF NOT EXISTS idx_tags_doc_id ON tags(doc_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
                """
            )

    def sync_document(
        self,
        doc: ParsedDocument,
        doc_id: str,
        rel_path: str,
        mtime_ms: int,
        file_size: int,
        embeddings: Optional[List[Tuple[str, List[float], str]]] = None,
    ) -> None:
        with self.conn:
            # Atomic delete existing rows
            self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

            fm_dict = {
                "title": doc.frontmatter.title,
                "aliases": doc.frontmatter.aliases,
                "tags": doc.frontmatter.tags,
                "custom": doc.frontmatter.custom,
            }
            frontmatter_json = json.dumps(fm_dict)

            self.conn.execute(
                """
                INSERT INTO documents (doc_id, path, title, frontmatter_json, word_count, mtime, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    rel_path,
                    doc.title,
                    frontmatter_json,
                    doc.word_count,
                    mtime_ms,
                    doc.content_hash,
                ),
            )

            # Insert sections
            for sec in doc.sections:
                self.conn.execute(
                    """
                    INSERT INTO sections (section_id, doc_id, heading, level, parent_id, breadcrumbs, line_start, line_end, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sec.id,
                        doc_id,
                        sec.heading,
                        sec.level,
                        sec.parent_id,
                        json.dumps(sec.breadcrumbs),
                        sec.line_start,
                        sec.line_end,
                        sec.content,
                    ),
                )

            # Insert links
            for link in doc.links:
                self.conn.execute(
                    """
                    INSERT INTO links (source_doc_id, target_note, target_heading, target_block, alias, link_type, line_number, context_snippet, source_section_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        link.target_note,
                        link.target_heading,
                        link.target_block,
                        link.alias,
                        link.link_type.value,
                        link.line_number,
                        link.context_snippet,
                        link.source_section_id,
                    ),
                )

            # Insert tags
            for tag in doc.tags:
                self.conn.execute(
                    """
                    INSERT INTO tags (doc_id, tag, section_id, line_number)
                    VALUES (?, ?, ?, ?)
                    """,
                    (doc_id, tag.name, tag.section_id, tag.line_number),
                )

            # Insert chunks
            for chunk in doc.chunks:
                bc_val = (
                    " > ".join(chunk.breadcrumbs)
                    if isinstance(chunk.breadcrumbs, (list, tuple))
                    else str(chunk.breadcrumbs)
                )
                self.conn.execute(
                    """
                    INSERT INTO chunks (chunk_id, doc_id, section_id, title, breadcrumbs, line_start, line_end, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        doc_id,
                        chunk.section_id,
                        chunk.title,
                        bc_val,
                        chunk.line_start,
                        chunk.line_end,
                        chunk.content,
                    ),
                )

            # Update manifest
            self.conn.execute(
                """
                INSERT INTO sync_manifest (path, doc_id, mtime, file_size, hash, synced_at, status)
                VALUES (?, ?, ?, ?, ?, strftime('%s', 'now'), 'synced')
                ON CONFLICT(path) DO UPDATE SET
                    doc_id = excluded.doc_id,
                    mtime = excluded.mtime,
                    file_size = excluded.file_size,
                    hash = excluded.hash,
                    synced_at = excluded.synced_at,
                    status = 'synced'
                """,
                (rel_path, doc_id, mtime_ms, file_size, doc.content_hash),
            )

    def get_document(self, query: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT doc_id, path, title, frontmatter_json, word_count, mtime, hash
            FROM documents
            WHERE doc_id = ? OR path = ? OR LOWER(title) = LOWER(?)
            LIMIT 1
            """,
            (query, query, query),
        )
        row = cur.fetchone()
        if not row:
            return None

        return {
            "doc_id": row["doc_id"],
            "path": row["path"],
            "title": row["title"],
            "frontmatter": json.loads(row["frontmatter_json"] or "{}"),
            "word_count": row["word_count"],
            "mtime": row["mtime"],
            "hash": row["hash"],
        }

    def get_chunks_for_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT chunk_id, doc_id, section_id, title, breadcrumbs, line_start, line_end, content
            FROM chunks
            WHERE doc_id = ?
            ORDER BY line_start ASC
            """,
            (doc_id,),
        )
        return [
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "section_id": row["section_id"],
                "title": row["title"],
                "breadcrumbs": json.loads(row["breadcrumbs"] or "[]"),
                "line_start": row["line_start"],
                "line_end": row["line_end"],
                "content": row["content"],
            }
            for row in cur.fetchall()
        ]

    def get_sections_for_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT section_id, doc_id, heading, level, parent_id, breadcrumbs, line_start, line_end, content
            FROM sections
            WHERE doc_id = ?
            ORDER BY line_start ASC
            """,
            (doc_id,),
        )
        return [
            {
                "section_id": row["section_id"],
                "doc_id": row["doc_id"],
                "heading": row["heading"],
                "level": row["level"],
                "parent_id": row["parent_id"],
                "breadcrumbs": json.loads(row["breadcrumbs"] or "[]"),
                "line_start": row["line_start"],
                "line_end": row["line_end"],
                "content": row["content"],
            }
            for row in cur.fetchall()
        ]

    def get_outlinks_for_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT source_doc_id, target_note, target_heading, target_block, alias, link_type, line_number, context_snippet
            FROM links
            WHERE source_doc_id = ?
            ORDER BY line_number ASC
            """,
            (doc_id,),
        )
        return [
            {
                "source_doc_id": row["source_doc_id"],
                "target_note": row["target_note"],
                "target_heading": row["target_heading"],
                "target_block": row["target_block"],
                "alias": row["alias"],
                "link_type": row["link_type"],
                "line_number": row["line_number"],
                "context_snippet": row["context_snippet"],
            }
            for row in cur.fetchall()
        ]

    def get_all_document_paths(self) -> List[str]:
        cur = self.conn.execute("SELECT path FROM documents ORDER BY path ASC")
        return [row["path"] for row in cur.fetchall()]

    def get_manifest(self, rel_path: str) -> Optional[Tuple[str, int, int]]:
        cur = self.conn.execute(
            "SELECT hash, mtime, file_size FROM sync_manifest WHERE path = ?", (rel_path,)
        )
        row = cur.fetchone()
        if row:
            return row["hash"], row["mtime"], row["file_size"]
        return None

    def search_fts(self, fts_query: str, limit: int = 10) -> List[Tuple[str, str, str, str, float]]:
        """Run BM25 search over chunks returning (chunk_id, title, breadcrumbs, snippet, score)."""
        cur = self.conn.execute(
            """
            SELECT chunk_id, title, breadcrumbs, snippet(chunks_fts, 3, '<b>', '</b>', '...', 15),
                   bm25(chunks_fts, 5.0, 2.0, 1.0) as score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?
            """,
            (fts_query, limit),
        )
        results: List[Tuple[str, str, str, str, float]] = []
        for r in cur.fetchall():
            raw_bc = r[2] or ""
            if raw_bc.startswith("["):
                try:
                    breadcrumbs = " > ".join(json.loads(raw_bc))
                except Exception:
                    breadcrumbs = raw_bc
            else:
                breadcrumbs = raw_bc
            results.append((r[0], r[1], breadcrumbs, r[3], float(r[4])))
        return results

    def get_backlinks(self, note_title: str) -> List[BacklinkRecord]:
        cur = self.conn.execute(
            """
            SELECT d.doc_id, d.path, d.title, l.line_number, l.link_type,
                   l.target_heading, l.target_block, l.alias, l.context_snippet
            FROM links l
            JOIN documents d ON l.source_doc_id = d.doc_id
            WHERE l.target_note = ? COLLATE NOCASE
            ORDER BY d.title, l.line_number
            """,
            (note_title,),
        )
        return [
            BacklinkRecord(
                source_doc_id=row["doc_id"],
                source_path=row["path"],
                source_title=row["title"],
                line_number=row["line_number"],
                link_type=row["link_type"],
                target_heading=row["target_heading"],
                target_block=row["target_block"],
                alias=row["alias"],
                snippet=row["context_snippet"],
            )
            for row in cur.fetchall()
        ]

    def get_unresolved_links(self) -> List[UnresolvedLinkRecord]:
        cur = self.conn.execute(
            """
            SELECT l.target_note, COUNT(DISTINCT l.source_doc_id) as ref_count,
                   GROUP_CONCAT(DISTINCT d.title) as sources
            FROM links l
            JOIN documents d ON l.source_doc_id = d.doc_id
            WHERE l.target_note NOT IN (SELECT title FROM documents)
              AND l.target_note NOT IN (SELECT path FROM documents)
            GROUP BY LOWER(l.target_note)
            ORDER BY ref_count DESC
            """
        )
        return [
            UnresolvedLinkRecord(
                target_note=row["target_note"],
                reference_count=row["ref_count"],
                source_notes=(row["sources"] or "").split(","),
            )
            for row in cur.fetchall()
        ]

    def get_stats(self) -> VaultStats:
        c = self.conn
        doc_count = c.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        sec_count = c.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
        link_count = c.execute("SELECT COUNT(*) FROM links").fetchone()[0]
        tag_count = c.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
        chunk_count = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        embed_count = c.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
        word_sum = c.execute("SELECT COALESCE(SUM(word_count), 0) FROM documents").fetchone()[0]

        return VaultStats(
            total_documents=doc_count,
            total_sections=sec_count,
            total_links=link_count,
            total_tags=tag_count,
            total_chunks=chunk_count,
            total_embeddings=embed_count,
            total_words=word_sum,
        )

    def get_all_documents(self) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT doc_id, path, title, frontmatter_json, word_count, mtime, hash
            FROM documents
            ORDER BY title ASC
            """
        )
        return [
            {
                "doc_id": row["doc_id"],
                "path": row["path"],
                "title": row["title"],
                "frontmatter_json": row["frontmatter_json"],
                "word_count": row["word_count"],
                "mtime": row["mtime"],
                "hash": row["hash"],
            }
            for row in cur.fetchall()
        ]

    def get_all_tags(self) -> List[str]:
        cur = self.conn.execute("SELECT DISTINCT tag FROM tags ORDER BY tag ASC")
        return [row["tag"] for row in cur.fetchall()]

    def get_tag_count(self, tag: str) -> int:
        clean_tag = tag.lstrip("#")
        with_hash = f"#{clean_tag}"
        cur = self.conn.execute(
            "SELECT COUNT(DISTINCT doc_id) FROM tags WHERE tag = ? OR tag = ?",
            (clean_tag, with_hash),
        )
        row = cur.fetchone()
        return row[0] if row else 0

    def save_node_embeddings(
        self, embeddings: List[Tuple[str, List[float]]], model: str = "node2vec"
    ) -> None:
        """Save topological node embeddings to SQLite node_embeddings table."""
        with self.conn:
            for title, vec in embeddings:
                blob = struct.pack(f"<{len(vec)}f", *vec)
                self.conn.execute(
                    """
                    INSERT INTO node_embeddings (node_title, embedding, dimensions, model, updated_at)
                    VALUES (?, ?, ?, ?, strftime('%s', 'now'))
                    ON CONFLICT(node_title) DO UPDATE SET
                        embedding = excluded.embedding,
                        dimensions = excluded.dimensions,
                        model = excluded.model,
                        updated_at = excluded.updated_at
                    """,
                    (title, blob, len(vec), model),
                )

    def get_node_embedding(self, note_title: str, model: str = "node2vec") -> Optional[List[float]]:
        cur = self.conn.execute(
            "SELECT embedding, dimensions FROM node_embeddings WHERE node_title = ? AND model = ?",
            (note_title, model),
        )
        row = cur.fetchone()
        if not row:
            return None
        blob, dims = row[0], row[1]
        return list(struct.unpack(f"<{dims}f", blob))

    def get_all_node_embeddings(self, model: str = "node2vec") -> Dict[str, List[float]]:
        cur = self.conn.execute(
            "SELECT node_title, embedding, dimensions FROM node_embeddings WHERE model = ? ORDER BY node_title ASC",
            (model,),
        )
        results = {}
        for row in cur.fetchall():
            title, blob, dims = row[0], row[1], row[2]
            results[title] = list(struct.unpack(f"<{dims}f", blob))
        return results

