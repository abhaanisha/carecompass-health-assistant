"""Load the markdown knowledge base into retrievable, citable chunks.

Chunking is done at the ``##`` section boundary rather than at a fixed token
count. Each section in the corpus is already a self-contained clinical idea
("Dengue warning signs that need hospital care"), so section-level chunks keep
retrieved context coherent and make citations meaningful to a reader: the
citation names a heading a human can find, not "chunk 47".

Sections longer than ``MAX_CHARS`` are split on paragraph boundaries with a
one-paragraph overlap so that a list of red flags is never cut in half.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import KB_DIR

MAX_CHARS = 1600
MIN_CHARS = 120

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    """One citable passage."""

    chunk_id: str
    doc_id: str
    doc_title: str
    heading: str
    text: str
    source: str
    source_url: str
    last_reviewed: str
    topics: tuple[str, ...]

    @property
    def citation(self) -> str:
        return f"{self.doc_title} — {self.heading}"

    @property
    def searchable_text(self) -> str:
        """Heading and topics are repeated so they carry weight in BM25."""
        topics = " ".join(self.topics)
        return f"{self.doc_title}. {self.heading}. {topics}. {self.text}"


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta = yaml.safe_load(match.group(1)) or {}
    return meta, raw[match.end():]


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split a document body into ``(heading, text)`` pairs."""
    matches = list(_SECTION_RE.finditer(body))
    if not matches:
        return [("Overview", body.strip())]

    sections: list[tuple[str, str]] = []
    preamble = body[: matches[0].start()].strip()
    if len(preamble) >= MIN_CHARS:
        sections.append(("Overview", preamble))

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            sections.append((match.group(1).strip(), text))
    return sections


def _split_long(text: str) -> list[str]:
    """Paragraph-wise split with one paragraph of overlap."""
    if len(text) <= MAX_CHARS:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts: list[str] = []
    buffer: list[str] = []
    size = 0

    for para in paragraphs:
        if size and size + len(para) > MAX_CHARS:
            parts.append("\n\n".join(buffer))
            buffer = buffer[-1:]  # overlap
            size = sum(len(p) for p in buffer)
        buffer.append(para)
        size += len(para)

    if buffer:
        parts.append("\n\n".join(buffer))
    return parts


def load_chunks(kb_dir: Path | None = None) -> list[Chunk]:
    """Read every ``.md`` file in the knowledge directory into chunks."""
    kb_dir = Path(kb_dir or KB_DIR)
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {kb_dir}")

    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        doc_id = str(meta.get("doc_id") or path.stem)
        topics = tuple(str(t) for t in (meta.get("topics") or []))

        for heading, text in split_sections(body):
            for i, part in enumerate(_split_long(text)):
                if len(part) < MIN_CHARS:
                    continue
                slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")[:48]
                suffix = f"-{i}" if i else ""
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::{slug}{suffix}",
                        doc_id=doc_id,
                        doc_title=str(meta.get("title") or doc_id),
                        heading=heading,
                        text=part,
                        source=str(meta.get("source") or "Curated public-health guidance"),
                        source_url=str(meta.get("source_url") or ""),
                        last_reviewed=str(meta.get("last_reviewed") or ""),
                        topics=topics,
                    )
                )

    if not chunks:
        raise ValueError(f"No knowledge chunks parsed from {kb_dir}")
    return chunks


def corpus_fingerprint(chunks: list[Chunk], extra: str = "") -> str:
    """Stable hash of corpus content, used to invalidate the embedding cache."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(chunk.text.encode("utf-8"))
    digest.update(extra.encode("utf-8"))
    return digest.hexdigest()[:16]


def corpus_stats(chunks: list[Chunk]) -> dict:
    docs = {c.doc_id for c in chunks}
    words = sum(len(c.text.split()) for c in chunks)
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "words": words,
        "avg_chunk_words": round(words / max(len(chunks), 1)),
    }
