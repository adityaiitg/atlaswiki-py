# AtlasWiki (Python Edition) 🪐🐍

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AtlasWiki Python** is the high-performance Python implementation of the AtlasWiki personal knowledge base (PKB) indexer, knowledge graph engine, and semantic search system designed for Markdown vaults (compatible with Obsidian, Logseq, Foam, and Dendron).

It shares complete database and schema parity with the Rust version of AtlasWiki, utilizing SQLite WAL mode, FTS5 full-text indexing, Dangling-Safe PageRank, Damerau-Levenshtein diagnostics, and a Model Context Protocol (MCP) stdio server.

---

## Key Features

- 🌳 **Extended Markdown AST Parser (`atlaswiki.parser`)**:
  - Full support for `[[Wikilinks]]`, `[[Note|Alias]]`, `[[Note#Heading]]`, and `[[Note#^block-id]]`.
  - Transclusions & embeds `![[Note]]` and `![[Note#Heading]]`.
  - Hierarchical nested tags (`#category/subcategory`), ignoring `#` in hex colors, numbers, or headings.
  - End-of-line block identifiers `^block-id` for granular referencing.
  - Code block and inline code span isolation (never triggers false positive links or tags).
  - Robust YAML frontmatter parsing (`title`, `aliases`, `tags`, arbitrary metadata).
  - Semantic section chunking that preserves structural ancestry breadcrumbs (`Note > H1 > H2`).

- ⚡ **SQLite WAL Storage Engine (`atlaswiki.storage`)**:
  - Binary-compatible schema with the Rust edition: Foreign Keys and ON DELETE CASCADE.
  - SQLite FTS5 full-text search with Unicode61 tokenizer and Porter stemming.
  - Automated SQLite triggers synchronizing FTS virtual tables.
  - Incremental sync engine tracking SHA-256 hashes and modification timestamps.

- 🕸️ **Bidirectional Knowledge Graph & Graph Theory (`atlaswiki.graph`)**:
  - In-memory directional graph representation.
  - Dangling-Safe PageRank power iteration to surface Map-of-Content (MOC) cornerstone notes.
  - Bidirectional BFS shortest conceptual pathfinder between any two notes (e.g., `Quantum -> Information Theory -> Cryptography`).
  - Structural diagnostics: automated orphan detection (0 in, 0 out) and wanted unwritten pages.

- 🔍 **Hybrid Query Classification & Ranking (`atlaswiki.retrieval`)**:
  - Dynamic query classification: `CodeSymbol`, `NaturalLanguage`, and `BalancedHybrid`.
  - FTS5 BM25 lexical ranking combined with PKB-specific reranking multipliers:
    - Note title exact match ($2.5\times$).
    - Heading match ($2.0\times$).
    - Tag match ($1.5\times$).
    - Graph PageRank centrality boost (+20%).

- 🌐 **Interactive D3.js Web Graph Visualizer (`atlaswiki serve`)**:
  - Embedded zero-dependency HTTP server (`http.server`).
  - Dark Obsidian/GitHub theme canvas/SVG force-directed graph.
  - Tag cluster color palette, PageRank node sizing, real-time search filter.
  - Slide-in inspection drawer showing note preview, frontmatter, and incoming backlinks.
  - Focus mode isolating 1-hop and 2-hop neighborhood subgraphs.

- 🩺 **Dead Link & Typo Diagnostics Linter (`atlaswiki check`)**:
  - Detects dead wikilinks, broken heading anchors, and missing block references.
  - Embedded Damerau-Levenshtein typo correction engine suggesting `Did you mean [[Machine Learning]]?`.
  - Compiler-style terminal formatting with exact line and column pointers.

- 🤖 **Model Context Protocol (MCP) Server (`atlaswiki-mcp`)**:
  - Conforms to MCP stdio JSON-RPC 2.0 specification (`2024-11-05`).
  - Exposes 6 AI tools for Cursor, Claude Desktop, and Gemini (`atlaswiki_search`, `atlaswiki_get_note`, `atlaswiki_backlinks`, `atlaswiki_graph_neighbors`, `atlaswiki_unresolved_links`, `atlaswiki_stats`).

---

## Installation

Prerequisites: Python 3.10+.

```bash
# Clone the repository
git clone https://github.com/adityaiitg/atlaswiki-py.git
cd atlaswiki-py

# Install via pip
pip install -e .
```

---

## CLI Usage

### 1. Indexing a Vault
```bash
# Index current directory or specified vault
atlaswiki index /path/to/vault

# Force a full re-indexing of all files
atlaswiki index /path/to/vault --full
```

### 2. Search
```bash
# Search vault notes and chunks
atlaswiki search "transformer self-attention" -C /path/to/vault

# Output results as JSON (for piping or editor integration)
atlaswiki search "neural networks" --json -C /path/to/vault

# Filter search results by tag
atlaswiki search "gradient descent" --tag "#ai/ml" -C /path/to/vault
```

### 3. Note & Backlinks Inspection
```bash
# Inspect note metadata, sections hierarchy, outlinks, and backlinks
atlaswiki note "Machine Learning" -C /path/to/vault

# List all incoming backlinks referencing a note
atlaswiki backlinks "Machine Learning" -C /path/to/vault
```

### 4. Knowledge Graph Analysis
```bash
# General vault metrics (nodes, links, chunks, words)
atlaswiki graph stats -C /path/to/vault

# List orphan notes (notes with zero incoming and outgoing links)
atlaswiki graph orphans -C /path/to/vault

# List wanted / dangling pages (referenced notes that do not exist yet)
atlaswiki graph wanted -C /path/to/vault

# Find the shortest conceptual path between two notes
atlaswiki graph path "AtlasWiki Index" "Neural Networks" -C /path/to/vault
```

### 5. Link Diagnostics Linter
```bash
# Scan vault for dead wikilinks, broken headings, and missing blocks
atlaswiki check /path/to/vault

# Strict mode (fails CI with non-zero exit code if warnings exist)
atlaswiki check /path/to/vault --strict
```

### 6. Interactive Web Visualizer
```bash
# Launch embedded HTTP web server on default port 8888
atlaswiki serve /path/to/vault

# Specify custom port and open browser automatically
atlaswiki serve /path/to/vault --port 9000 --open
```

### 7. AI Agent Model Context Protocol (MCP)
```bash
# Launch MCP stdio server
atlaswiki-mcp -C /path/to/vault
```

To configure in Claude Desktop or Cursor (`~/.config/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "atlaswiki": {
      "command": "atlaswiki-mcp",
      "args": ["-C", "/absolute/path/to/your/markdown/vault"]
    }
  }
}
```

---

## Running Unit Tests

```bash
python3 tests/test_all.py
```

---

## License

This project is licensed under the [MIT License](LICENSE).
