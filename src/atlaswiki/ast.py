from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class LinkType(str, Enum):
    WIKILINK = "wikilink"
    EMBED = "embed"
    MARKDOWN = "markdown"


@dataclass
class AstLink:
    target_note: str
    target_heading: Optional[str] = None
    target_block: Optional[str] = None
    alias: Optional[str] = None
    link_type: LinkType = LinkType.WIKILINK
    line_number: int = 1
    context_snippet: Optional[str] = None
    source_section_id: Optional[str] = None


@dataclass
class AstTag:
    name: str
    line_number: int
    section_id: Optional[str] = None


@dataclass
class AstSection:
    id: str
    heading: str
    level: int
    parent_id: Optional[str]
    breadcrumbs: List[str]
    line_start: int
    line_end: int
    content: str


@dataclass
class AstChunk:
    chunk_id: str
    section_id: str
    title: str
    breadcrumbs: List[str]
    line_start: int
    line_end: int
    content: str


@dataclass
class Frontmatter:
    title: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    path: Path
    title: str
    frontmatter: Frontmatter
    sections: List[AstSection]
    links: List[AstLink]
    tags: List[AstTag]
    chunks: List[AstChunk]
    word_count: int
    content_hash: str
