"""Hybrid retrieval: BM25 + dense embeddings fused with Reciprocal Rank Fusion.

Why hybrid rather than embeddings alone, which is the usual starting point:

* Dense retrieval alone misses exact clinical tokens. A query containing
  "HbA1c" or "112" must land on the passage containing that literal string, and
  a 384-dimension MiniLM vector does not reliably preserve a rare token.
* Lexical retrieval alone misses paraphrase. "I feel like I can't catch my
  breath" shares no content word with "severe breathlessness".

RRF is used instead of score interpolation because BM25 scores and cosine
similarities live on different, corpus-dependent scales, so any fixed alpha
weighting needs retuning whenever the corpus changes. Rank fusion does not.

The dense half degrades gracefully: if ``sentence-transformers`` is missing or
fails to load, the retriever drops to lexical-only and says so, rather than
taking the whole app down. That keeps the Space bootable on a cold CPU box.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .config import INDEX_DIR, LEXICON_PATH, Settings, get_settings
from .knowledge import Chunk, corpus_fingerprint, load_chunks

# Latin, Devanagari and Bengali word characters.
TOKEN_RE = re.compile(r"[0-9A-Za-zऀ-ॿঀ-৿]+")

RRF_K = 60
CANDIDATE_DEPTH = 15

# Below these, the question is treated as outside the corpus and the assistant
# says so instead of stretching an unrelated passage to fit.
#
# A BM25 score threshold alone is not enough: "write me a python function to
# sort a list" scores respectably against a long passage purely on common
# words. So a lexical match must ALSO account for a reasonable share of the
# query's *information content* before it counts as grounding -- see
# ``Retriever._term_coverage``.
MIN_LEXICAL_SCORE = 3.0
MIN_DENSE_SCORE = 0.28
MIN_TERM_COVERAGE = 0.30


def _load_lexicon() -> tuple[dict[str, list[str]], set[str]]:
    if not LEXICON_PATH.exists():
        return {}, set()
    raw = yaml.safe_load(LEXICON_PATH.read_text(encoding="utf-8")) or {}
    expansions = {
        str(k).lower(): [str(v) for v in vals]
        for k, vals in (raw.get("expansions") or {}).items()
    }
    stopwords = {str(w).lower() for w in (raw.get("stopwords") or [])}
    return expansions, stopwords


# Suffix order matters: longer endings are tried first so that "infections"
# reduces the same way "infection" does.
_SUFFIXES = ("ions", "ing", "ies", "ied", "ion", "ed", "es", "s")
_MIN_STEM = 4


def stem(token: str) -> str:
    """A deliberately small suffix stripper.

    Full Porter stemming is more than this corpus needs and brings a
    dependency. What it does need is for "dehydrated" and "dehydration" to
    collide, because a user types the first and the knowledge base is written
    with the second. Stripping is applied repeatedly with a minimum stem
    length, which keeps short clinical words like "ORS" and "flu" intact.
    """
    if len(token) <= _MIN_STEM or not token.isascii():
        return token
    changed = True
    while changed:
        changed = False
        for suffix in _SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
                token = token[: -len(suffix)]
                if suffix == "ies":
                    token += "y"
                changed = True
                break
    return token


def tokenize(text: str, stopwords: set[str] | None = None) -> list[str]:
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return [stem(t) for t in tokens]


class BM25Index:
    """Textbook BM25 over an in-memory posting list.

    A dependency-free implementation is deliberate: ``rank_bm25`` would add a
    package for roughly forty lines of arithmetic, and a hand-rolled index lets
    the evaluation harness inspect per-term IDF when a retrieval miss needs
    explaining.
    """

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if self.n_docs else 1.0

        postings: dict[str, list[tuple[int, int]]] = {}
        for idx, tokens in enumerate(docs):
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                postings.setdefault(token, []).append((idx, count))

        self.postings = postings
        self.idf = {
            term: math.log(1 + (self.n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in postings.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        for term in query_tokens:
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for idx, tf in plist:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avg_len)
                out[idx] += idf * (tf * (self.k1 + 1)) / (tf + norm)
        return out


class DenseIndex:
    """Sentence-transformer embeddings with an on-disk cache.

    The cache is keyed by a hash of the corpus text plus the model name, so
    editing a knowledge file automatically invalidates it. Without this, every
    Space restart re-encodes the whole corpus.
    """

    def __init__(self, chunks: list[Chunk], model_name: str, cache_dir: Path = INDEX_DIR):
        self.chunks = chunks
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.model = None
        self.matrix: np.ndarray | None = None
        self.error: str | None = None

    @property
    def fingerprint(self) -> str:
        return corpus_fingerprint(self.chunks, extra=self.model_name)

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / f"emb-{self.fingerprint}.npy"

    def build(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - depends on install
            self.error = f"sentence-transformers unavailable ({exc.__class__.__name__})"
            return False

        try:
            self.model = SentenceTransformer(self.model_name)
            if self.cache_path.exists():
                self.matrix = np.load(self.cache_path)
                if self.matrix.shape[0] != len(self.chunks):
                    self.matrix = None
            if self.matrix is None:
                texts = [c.searchable_text for c in self.chunks]
                self.matrix = self.model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                ).astype(np.float32)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(self.cache_path, self.matrix)
            return True
        except Exception as exc:  # pragma: no cover - network/model failures
            self.error = f"embedding model failed to load ({exc})"
            self.model = None
            self.matrix = None
            return False

    def scores(self, query: str) -> np.ndarray:
        if self.model is None or self.matrix is None:
            return np.zeros(len(self.chunks), dtype=np.float32)
        vector = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)[0]
        return self.matrix @ vector


@dataclass
class Hit:
    chunk: Chunk
    fused_score: float
    lexical_score: float
    dense_score: float
    lexical_rank: int | None = None
    dense_rank: int | None = None

    @property
    def matched_by(self) -> str:
        if self.lexical_rank is not None and self.dense_rank is not None:
            return "both"
        if self.dense_rank is not None:
            return "semantic"
        return "keyword"


@dataclass
class RetrievalResult:
    query: str
    expanded_query: str
    hits: list[Hit] = field(default_factory=list)
    mode: str = "lexical"
    grounded: bool = False
    max_lexical: float = 0.0
    max_dense: float = 0.0
    term_coverage: float = 0.0

    @property
    def doc_ids(self) -> list[str]:
        return [h.chunk.doc_id for h in self.hits]


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.chunks = chunks if chunks is not None else load_chunks()
        self.expansions, self.stopwords = _load_lexicon()

        self.bm25 = BM25Index(
            [tokenize(c.searchable_text, self.stopwords) for c in self.chunks]
        )

        self.dense: DenseIndex | None = None
        self.mode = "lexical"
        if self.settings.use_dense:
            dense = DenseIndex(self.chunks, self.settings.embedding_model)
            if dense.build():
                self.dense = dense
                self.mode = "hybrid"
            else:
                self.dense_error = dense.error

    # -- query preparation -------------------------------------------------

    def expand(self, query: str) -> str:
        """Append clinical vocabulary for any lay term found in the query."""
        lowered = query.lower()
        extra: list[str] = []
        for term, targets in self.expansions.items():
            if term in lowered:
                extra.extend(targets)
        if not extra:
            return query
        # De-duplicate while preserving order.
        seen: set[str] = set()
        unique = [t for t in extra if not (t.lower() in seen or seen.add(t.lower()))]
        return f"{query} {' '.join(unique)}"

    # -- search ------------------------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or self.settings.top_k
        expanded = self.expand(query)

        lexical = self.bm25.scores(tokenize(expanded, self.stopwords))
        lexical_order = np.argsort(-lexical)[:CANDIDATE_DEPTH]
        lexical_rank = {
            int(idx): rank for rank, idx in enumerate(lexical_order) if lexical[idx] > 0
        }

        dense = np.zeros(len(self.chunks), dtype=np.float32)
        dense_rank: dict[int, int] = {}
        if self.dense is not None:
            dense = self.dense.scores(expanded)
            dense_order = np.argsort(-dense)[:CANDIDATE_DEPTH]
            dense_rank = {int(idx): rank for rank, idx in enumerate(dense_order)}

        fused: dict[int, float] = {}
        for idx, rank in lexical_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for idx, rank in dense_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        hits = [
            Hit(
                chunk=self.chunks[idx],
                fused_score=round(score, 6),
                lexical_score=round(float(lexical[idx]), 3),
                dense_score=round(float(dense[idx]), 3),
                lexical_rank=lexical_rank.get(idx),
                dense_rank=dense_rank.get(idx),
            )
            for idx, score in ranked
        ]

        max_lexical = float(lexical.max()) if lexical.size else 0.0
        max_dense = float(dense.max()) if dense.size else 0.0
        coverage = self._term_coverage(expanded, hits)
        grounded = max_dense >= MIN_DENSE_SCORE or (
            max_lexical >= MIN_LEXICAL_SCORE and coverage >= MIN_TERM_COVERAGE
        )

        return RetrievalResult(
            query=query,
            expanded_query=expanded,
            hits=hits,
            mode=self.mode,
            grounded=grounded,
            max_lexical=round(max_lexical, 3),
            max_dense=round(max_dense, 3),
            term_coverage=round(coverage, 3),
        )

    def _term_coverage(self, query: str, hits: list[Hit]) -> float:
        """How much of the query's information content the best hit accounts for.

        A plain word-overlap ratio does not work. "my 3 year old has loose
        motions since morning" is four-sevenths incidental detail, so raw
        overlap scores it as poorly covered even though the one term that
        carries the meaning matched perfectly.

        Weighting each term by IDF fixes that, and it is also what separates an
        out-of-scope question: terms like "python" or "france" are absent from
        the corpus entirely, so they are charged the maximum IDF in the
        denominator and contribute nothing to the numerator.
        """
        terms = set(tokenize(query, self.stopwords))
        if not terms or not hits:
            return 0.0

        max_idf = max(self.bm25.idf.values(), default=1.0)
        matched = terms & set(tokenize(hits[0].chunk.searchable_text, self.stopwords))

        covered = sum(self.bm25.idf.get(t, max_idf) for t in matched)
        total = sum(self.bm25.idf.get(t, max_idf) for t in terms)
        return covered / total if total else 0.0

    # -- introspection -----------------------------------------------------

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "chunks": len(self.chunks),
            "documents": len({c.doc_id for c in self.chunks}),
            "embedding_model": self.settings.embedding_model if self.dense else None,
            "lexicon_terms": len(self.expansions),
        }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    retriever = Retriever()
    print(json.dumps(retriever.describe(), indent=2))
    for q in [
        "loose motion in my 2 year old for two days",
        "seene me dard ho raha hai",
        "my HbA1c is 7.2, is that bad",
        "কাশি দুই সপ্তাহ ধরে",
    ]:
        result = retriever.search(q, top_k=3)
        print(f"\nQ: {q}  [grounded={result.grounded} mode={result.mode}]")
        for hit in result.hits:
            print(f"   {hit.matched_by:9s} {hit.fused_score:.4f}  {hit.chunk.citation}")
