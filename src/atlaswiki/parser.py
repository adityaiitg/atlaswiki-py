from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Optional, Tuple
import yaml

from atlaswiki.ast import (
    AstChunk,
    AstLink,
    AstSection,
    AstTag,
    Frontmatter,
    LinkType,
    ParsedDocument,
)


def slugify(s: str) -> str:
    cleaned = "".join(c.lower() if c.isalnum() or c.isspace() else "-" for c in s)
    tokens = [t for t in cleaned.split() if t]
    return "-".join(tokens) or "section"


class MarkdownParser:
    """Parser for extended Markdown PKB syntax (Obsidian, Logseq, Foam)."""

    WIKILINK_RE = re.compile(
        r"(?P<embed>!)?\[\[(?P<target>[^\]|#^]+)(?:#(?:(?:\^(?P<block>[^\]|]+))|(?P<heading>[^\]|^|]+)(?:\^(?P<block2>[^\]|]+))?))?(?:\^(?P<block_direct>[^\]|]+))?(?:\|(?P<alias>[^\]]+))?\]\]"
    )

    MD_LINK_RE = re.compile(
        r"(?P<embed>!)?\[(?P<text>[^\]]+)\]\((?P<target>[^\s\)]+)(?:\s+\"(?P<title>[^\"]+)\")?\)"
    )

    TAG_RE = re.compile(
        r"(?<![\w/])#(?P<tag>[a-zA-Z][a-zA-Z0-9_\-]*(?:/[a-zA-Z][a-zA-Z0-9_\-]*)*)(?![\w/])"
    )

    BLOCK_ID_RE = re.compile(r"\s+\^(?P<id>[a-zA-Z0-9_\-]+)\s*$")

    def __init__(self) -> None:
        pass

    def parse_file(self, rel_path: Path, content: str) -> ParsedDocument:
        # 1. Compute SHA-256 hash
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # 2. Extract YAML Frontmatter
        frontmatter, body, body_line_offset = self._extract_frontmatter(content)

        # Scan for first H1
        first_h1 = None
        for line in body.splitlines():
            m = re.match(r"^#\s+(.+)$", line)
            if m:
                first_h1 = m.group(1).strip()
                break

        # 3. Determine Document Title
        stem = rel_path.stem
        title = (frontmatter.title.strip() if frontmatter.title else None) or first_h1 or stem

        # 4. Mask code blocks and inline spans to prevent false positive link/tag matches
        masked_body = self._mask_code(body)

        # 5. Build hierarchical sections
        sections = self._parse_sections(body, title, body_line_offset)

        # 6. Extract links and tags
        links, tags = self._extract_links_and_tags(masked_body, sections, body_line_offset)

        # Add tags from frontmatter
        for fm_tag in frontmatter.tags:
            clean_tag = fm_tag.lstrip("#")
            if clean_tag and not any(t.name.lower() == clean_tag.lower() and t.line_number == 1 for t in tags):
                tags.append(AstTag(name=clean_tag, line_number=1, section_id=None))

        # 7. Generate semantic chunks with breadcrumbs
        chunks = self._generate_chunks(sections, title)

        # 8. Word count
        word_count = len(body.split())

        return ParsedDocument(
            path=rel_path,
            title=title,
            frontmatter=frontmatter,
            sections=sections,
            links=links,
            tags=tags,
            chunks=chunks,
            word_count=word_count,
            content_hash=content_hash,
        )

    def _extract_frontmatter(self, text: str) -> Tuple[Frontmatter, str, int]:
        lines = text.splitlines(keepends=True)
        if not lines or not lines[0].startswith("---"):
            return Frontmatter(), text, 1

        end_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break

        if end_idx is None:
            return Frontmatter(), text, 1

        fm_text = "".join(lines[1:end_idx])
        body_text = "".join(lines[end_idx + 1:])
        body_line_offset = end_idx + 2

        try:
            data = yaml.safe_load(fm_text) or {}
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}

        title = data.get("title")
        if title is not None:
            title = str(title)

        aliases_raw = data.get("aliases", [])
        if isinstance(aliases_raw, str):
            aliases = [aliases_raw]
        elif isinstance(aliases_raw, list):
            aliases = [str(a) for a in aliases_raw if a is not None]
        else:
            aliases = []

        tags_raw = data.get("tags", [])
        if isinstance(tags_raw, str):
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        elif isinstance(tags_raw, list):
            tags = [str(t) for t in tags_raw if t is not None]
        else:
            tags = []

        return Frontmatter(title=title, aliases=aliases, tags=tags, custom=data), body_text, body_line_offset

    def _mask_code(self, text: str) -> str:
        lines = text.splitlines(keepends=True)
        in_fence = False
        fence_marker = ""
        output: List[str] = []

        for line in lines:
            trimmed = line.strip()
            if not in_fence:
                if trimmed.startswith("```") or trimmed.startswith("~~~"):
                    in_fence = True
                    fence_marker = trimmed[:3]
                    output.append(" " * len(line))
                    continue
                else:
                    # Mask inline code spans
                    masked_line = re.sub(r"`[^`\n]+`", lambda m: " " * len(m.group(0)), line)
                    output.append(masked_line)
            else:
                if trimmed.startswith(fence_marker):
                    in_fence = False
                    output.append(" " * len(line))
                else:
                    output.append(" " * len(line))

        return "".join(output)

    def _parse_sections(self, body: str, doc_title: str, line_offset: int) -> List[AstSection]:
        lines = body.splitlines()
        if not lines:
            return []

        heading_re = re.compile(r"^(#{1,6})\s+(.+)$")
        raw_sections: List[Tuple[int, str, int]] = []  # (level, heading, line_idx)

        for idx, line in enumerate(lines):
            m = heading_re.match(line)
            if m:
                level = len(m.group(1))
                heading = m.group(2).strip()
                raw_sections.append((level, heading, idx))

        if not raw_sections:
            return [
                AstSection(
                    id=slugify(doc_title),
                    heading=doc_title,
                    level=1,
                    parent_id=None,
                    breadcrumbs=[doc_title],
                    line_start=line_offset,
                    line_end=line_offset + max(len(lines) - 1, 0),
                    content=body,
                )
            ]

        sections: List[AstSection] = []
        hierarchy: List[Tuple[int, str, str]] = []  # (level, id, heading)

        # Preamble before first heading
        if raw_sections[0][2] > 0:
            preamble_content = "\n".join(lines[:raw_sections[0][2]])
            sections.append(
                AstSection(
                    id=slugify(doc_title),
                    heading=doc_title,
                    level=1,
                    parent_id=None,
                    breadcrumbs=[doc_title],
                    line_start=line_offset,
                    line_end=line_offset + raw_sections[0][2] - 1,
                    content=preamble_content,
                )
            )
            hierarchy.append((1, slugify(doc_title), doc_title))

        for i, (level, heading, line_idx) in enumerate(raw_sections):
            next_line = raw_sections[i + 1][2] if i + 1 < len(raw_sections) else len(lines)
            sec_content = "\n".join(lines[line_idx:next_line])

            # Pop hierarchy until finding a lower level
            while hierarchy and hierarchy[-1][0] >= level:
                hierarchy.pop()

            parent_id = hierarchy[-1][1] if hierarchy else None
            breadcrumbs = [h[2] for h in hierarchy] + [heading]
            sec_id = slugify(f"{heading}-{i}")

            sections.append(
                AstSection(
                    id=sec_id,
                    heading=heading,
                    level=level,
                    parent_id=parent_id,
                    breadcrumbs=breadcrumbs,
                    line_start=line_offset + line_idx,
                    line_end=line_offset + next_line - 1,
                    content=sec_content,
                )
            )

            hierarchy.append((level, sec_id, heading))

        return sections

    def _extract_links_and_tags(
        self, masked_text: str, sections: List[AstSection], line_offset: int
    ) -> Tuple[List[AstLink], List[AstTag]]:
        links: List[AstLink] = []
        tags: List[AstTag] = []

        lines = masked_text.splitlines()
        for idx, line in enumerate(lines):
            line_no = line_offset + idx
            sec_id = self._find_section_id(line_no, sections)

            # 1. Wikilinks & Embeds
            for m in self.WIKILINK_RE.finditer(line):
                is_embed = bool(m.group("embed"))
                target = m.group("target").strip()
                block = m.group("block") or m.group("block2") or m.group("block_direct")
                heading = m.group("heading")
                alias = m.group("alias")

                links.append(
                    AstLink(
                        target_note=target,
                        target_heading=heading.strip() if heading else None,
                        target_block=block.strip() if block else None,
                        alias=alias.strip() if alias else None,
                        link_type=LinkType.EMBED if is_embed else LinkType.WIKILINK,
                        line_number=line_no,
                        context_snippet=line.strip() or None,
                        source_section_id=sec_id,
                    )
                )

            # 2. Standard Markdown Links
            for m in self.MD_LINK_RE.finditer(line):
                target = m.group("target").strip()
                # Skip external links
                if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", target):
                    continue
                if target.startswith("#"):
                    continue

                clean_target = target.rstrip(".md")
                is_embed = bool(m.group("embed"))

                links.append(
                    AstLink(
                        target_note=clean_target,
                        target_heading=None,
                        target_block=None,
                        alias=m.group("text").strip(),
                        link_type=LinkType.EMBED if is_embed else LinkType.MARKDOWN,
                        line_number=line_no,
                        context_snippet=line.strip() or None,
                        source_section_id=sec_id,
                    )
                )

            # 3. Tags
            for m in self.TAG_RE.finditer(line):
                tag_name = m.group("tag")
                tags.append(
                    AstTag(
                        name=tag_name,
                        line_number=line_no,
                        section_id=sec_id,
                    )
                )

        return links, tags

    def _find_section_id(self, line_no: int, sections: List[AstSection]) -> Optional[str]:
        for s in sections:
            if s.line_start <= line_no <= s.line_end:
                return s.id
        return None

    def _generate_chunks(self, sections: List[AstSection], doc_title: str) -> List[AstChunk]:
        chunks: List[AstChunk] = []

        for sec in sections:
            paragraphs = sec.content.split("\n\n")
            current_paras: List[str] = []
            chunk_idx = 0

            for p in paragraphs:
                p_trimmed = p.strip()
                if not p_trimmed:
                    continue

                current_paras.append(p_trimmed)
                current_text = "\n\n".join(current_paras)

                if len(current_text.split()) >= 350:
                    cid = f"{sec.id}::{chunk_idx}"
                    chunks.append(
                        AstChunk(
                            chunk_id=cid,
                            section_id=sec.id,
                            title=sec.heading,
                            breadcrumbs=sec.breadcrumbs,
                            line_start=sec.line_start,
                            line_end=sec.line_end,
                            content=current_text,
                        )
                    )
                    current_paras = []
                    chunk_idx += 1

            if current_paras:
                cid = f"{sec.id}::{chunk_idx}"
                chunks.append(
                    AstChunk(
                        chunk_id=cid,
                        section_id=sec.id,
                        title=sec.heading,
                        breadcrumbs=sec.breadcrumbs,
                        line_start=sec.line_start,
                        line_end=sec.line_end,
                        content="\n\n".join(current_paras),
                    )
                )

        return chunks
