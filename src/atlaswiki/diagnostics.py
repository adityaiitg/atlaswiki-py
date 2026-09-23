from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from atlaswiki.ast import LinkType, ParsedDocument
from atlaswiki.parser import slugify


class DiagnosticCode(str, Enum):
    UNRESOLVED_WIKILINK = "W001"
    BROKEN_HEADING_ANCHOR = "W002"
    BROKEN_BLOCK_REFERENCE = "W003"
    UNRESOLVED_EMBED = "W004"
    DEAD_MARKDOWN_LINK = "W005"


class DiagnosticSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class DiagnosticLocation:
    file_path: Path
    line_number: int
    col_start: int
    col_end: int
    source_line: str


@dataclass
class Diagnostic:
    code: DiagnosticCode
    severity: DiagnosticSeverity
    message: str
    location: DiagnosticLocation
    suggestion: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class WantedPage:
    target: str
    reference_count: int
    referencing_files: List[Path]
    occurrences: List[DiagnosticLocation]
    suggestions: List[str] = field(default_factory=list)


@dataclass
class DiagnosticsReport:
    total_files_scanned: int
    total_links_checked: int
    diagnostics: List[Diagnostic] = field(default_factory=list)
    wanted_pages: List[WantedPage] = field(default_factory=list)

    def warning_count(self) -> int:
        return sum(
            1 for d in self.diagnostics if d.severity == DiagnosticSeverity.WARNING
        )

    def error_count(self) -> int:
        return sum(
            1 for d in self.diagnostics if d.severity == DiagnosticSeverity.ERROR
        )


class TypoCorrectionEngine:
    """Damerau-Levenshtein typo correction engine with transposition support."""

    def __init__(
        self, max_edit_distance: int = 3, min_similarity_ratio: float = 0.55
    ) -> None:
        self.max_edit_distance = max_edit_distance
        self.min_similarity_ratio = min_similarity_ratio

    def distance(self, s1: str, s2: str) -> int:
        a = s1.lower()
        b = s2.lower()
        m, n = len(a), len(b)

        if abs(m - n) > self.max_edit_distance:
            return self.max_edit_distance + 1

        d = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            d[i][0] = i
        for j in range(n + 1):
            d[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                d[i][j] = min(
                    d[i - 1][j] + 1,  # deletion
                    d[i][j - 1] + 1,  # insertion
                    d[i - 1][j - 1] + cost,  # substitution
                )
                if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                    d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # transposition

        return d[m][n]

    def suggest(self, target: str, candidates: Iterable[str]) -> List[str]:
        scored: List[Tuple[float, int, str]] = []
        clean_target = target.lower().strip()
        len_t = len(clean_target)

        for cand in candidates:
            clean_c = cand.lower().strip()
            dist = self.distance(clean_target, clean_c)
            if dist <= self.max_edit_distance:
                max_len = max(len_t, len(clean_c))
                sim = 1.0 - (dist / max_len) if max_len > 0 else 1.0
                if sim >= self.min_similarity_ratio:
                    scored.append((sim, dist, cand))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [c for _, _, c in scored[:3]]


class DiagnosticsEngine:
    """Detects unresolved links, broken headings, missing block references, and typos."""

    def __init__(self) -> None:
        self.typo_engine = TypoCorrectionEngine()
        self.notes: Dict[Path, Dict[str, Any]] = {}
        self.title_to_path: Dict[str, Path] = {}
        self.alias_to_path: Dict[str, Path] = {}
        self.stem_to_path: Dict[str, Path] = {}

    def index_document(self, doc: ParsedDocument) -> None:
        block_ids: Set[str] = set()
        for sec in doc.sections:
            for line in sec.content.splitlines():
                if "^" in line:
                    pid = line.split("^")[-1].strip()
                    if pid and all(c.isalnum() or c in "-_" for c in pid):
                        block_ids.add(pid)

        headings = [s.heading for s in doc.sections]

        self.notes[doc.path] = {
            "title": doc.title,
            "headings": headings,
            "block_ids": block_ids,
        }

        stem = doc.path.stem.lower()
        self.stem_to_path[stem] = doc.path
        self.title_to_path[doc.title.lower()] = doc.path

        for alias in doc.frontmatter.aliases:
            self.alias_to_path[alias.lower()] = doc.path

    def resolve_target(
        self, target: str, from_doc_path: Optional[Path] = None
    ) -> Optional[Path]:
        clean = target.rstrip(".md").strip()
        if not clean:
            return None

        clean_norm = clean[2:] if clean.startswith("./") else clean
        p = Path(clean_norm)
        if p in self.notes:
            return p
        p_md = Path(f"{clean_norm}.md")
        if p_md in self.notes:
            return p_md

        if from_doc_path is not None and from_doc_path.parent:
            parent = from_doc_path.parent
            rel = parent / p
            if rel in self.notes:
                return rel
            rel_md = parent / p_md
            if rel_md in self.notes:
                return rel_md

        lower = clean.lower()
        if lower in self.title_to_path:
            return self.title_to_path[lower]
        if lower in self.alias_to_path:
            return self.alias_to_path[lower]
        if lower in self.stem_to_path:
            return self.stem_to_path[lower]

        return None

    def all_known_targets(self) -> List[str]:
        targets = set()
        for meta in self.notes.values():
            targets.add(meta["title"])
        for alias in self.alias_to_path:
            targets.add(alias)
        return list(targets)

    def run(
        self, docs: List[ParsedDocument], strict_mode: bool = False
    ) -> DiagnosticsReport:
        diagnostics: List[Diagnostic] = []
        wanted_map: Dict[str, List[DiagnosticLocation]] = {}
        total_links = 0
        all_targets = self.all_known_targets()

        for doc in docs:
            for link in doc.links:
                total_links += 1
                resolved_path = self.resolve_target(
                    link.target_note, from_doc_path=doc.path
                )

                loc = DiagnosticLocation(
                    file_path=doc.path,
                    line_number=link.line_number,
                    col_start=1,
                    col_end=len(link.target_note) + 4,
                    source_line=link.context_snippet or "",
                )

                if resolved_path is None:
                    suggestions = self.typo_engine.suggest(
                        link.target_note, all_targets
                    )
                    primary_sug = f"[[{suggestions[0]}]]" if suggestions else None

                    diag_code = (
                        DiagnosticCode.UNRESOLVED_EMBED
                        if link.link_type == LinkType.EMBED
                        else (
                            DiagnosticCode.DEAD_MARKDOWN_LINK
                            if link.link_type == LinkType.MARKDOWN
                            else DiagnosticCode.UNRESOLVED_WIKILINK
                        )
                    )

                    diagnostics.append(
                        Diagnostic(
                            code=diag_code,
                            severity=DiagnosticSeverity.ERROR
                            if strict_mode
                            else DiagnosticSeverity.WARNING,
                            message=f"target note '{link.target_note}' does not exist in vault",
                            location=loc,
                            suggestion=primary_sug,
                        )
                    )

                    wanted_map.setdefault(link.target_note, []).append(loc)
                    continue

                target_meta = self.notes[resolved_path]

                # Broken heading check
                if link.target_heading:
                    clean_h = link.target_heading.strip().lstrip("#")
                    clean_slug = slugify(clean_h)
                    exists = any(
                        h.lower() == clean_h.lower() or slugify(h) == clean_slug
                        for h in target_meta["headings"]
                    )
                    if not exists:
                        heading_sugs = self.typo_engine.suggest(
                            clean_h, target_meta["headings"]
                        )
                        sug = (
                            f"[[{link.target_note}#{heading_sugs[0]}]]"
                            if heading_sugs
                            else None
                        )

                        diagnostics.append(
                            Diagnostic(
                                code=DiagnosticCode.BROKEN_HEADING_ANCHOR,
                                severity=DiagnosticSeverity.ERROR
                                if strict_mode
                                else DiagnosticSeverity.WARNING,
                                message=f"heading anchor '#{clean_h}' not found in note '{link.target_note}'",
                                location=loc,
                                suggestion=sug,
                            )
                        )

                # Broken block reference check
                if link.target_block:
                    clean_b = link.target_block.strip().lstrip("^")
                    if clean_b not in target_meta["block_ids"]:
                        diagnostics.append(
                            Diagnostic(
                                code=DiagnosticCode.BROKEN_BLOCK_REFERENCE,
                                severity=DiagnosticSeverity.ERROR
                                if strict_mode
                                else DiagnosticSeverity.WARNING,
                                message=f"block reference '^{clean_b}' not found in note '{link.target_note}'",
                                location=loc,
                            )
                        )

        wanted_pages = [
            WantedPage(
                target=target,
                reference_count=len(locs),
                referencing_files=list({loc.file_path for loc in locs}),
                occurrences=locs,
                suggestions=self.typo_engine.suggest(target, all_targets),
            )
            for target, locs in wanted_map.items()
        ]
        wanted_pages.sort(key=lambda x: x.reference_count, reverse=True)

        return DiagnosticsReport(
            total_files_scanned=len(docs),
            total_links_checked=total_links,
            diagnostics=diagnostics,
            wanted_pages=wanted_pages,
        )
