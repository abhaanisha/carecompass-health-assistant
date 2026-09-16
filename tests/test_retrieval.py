from __future__ import annotations

import pytest

from src.config import get_settings
from src.knowledge import corpus_stats, load_chunks, parse_frontmatter, split_sections
from src.retrieval import BM25Index, Retriever, tokenize


@pytest.fixture(scope="module")
def chunks():
    return load_chunks()


@pytest.fixture(scope="module")
def retriever(chunks):
    # Lexical only: the dense half needs a model download, which has no place
    # in a unit test suite.
    return Retriever(chunks=chunks, settings=get_settings().with_(use_dense=False))


def test_corpus_loads_with_metadata(chunks):
    stats = corpus_stats(chunks)
    assert stats["documents"] >= 10
    assert stats["chunks"] > 30
    for chunk in chunks:
        assert chunk.doc_title and chunk.heading and chunk.text
        assert chunk.source, f"{chunk.chunk_id} has no source attribution"


def test_frontmatter_parsing():
    meta, body = parse_frontmatter("---\ndoc_id: x\ntitle: T\n---\n## A\ntext\n")
    assert meta["doc_id"] == "x"
    assert body.strip().startswith("## A")


def test_sections_split_on_headings():
    sections = split_sections("## One\nalpha\n\n## Two\nbeta")
    assert [heading for heading, _ in sections] == ["One", "Two"]


def test_tokenizer_handles_indic_scripts():
    assert "जर" not in tokenize("बुखार है")  # tokens are whole words
    assert tokenize("बुखार है")
    assert tokenize("জ্বর হয়েছে")


def test_bm25_ranks_the_relevant_document_first():
    docs = [["fever", "dengue", "platelet"], ["burn", "water", "cool"], ["salt", "pressure"]]
    index = BM25Index(docs)
    scores = index.scores(["dengue", "platelet"])
    assert scores.argmax() == 0


@pytest.mark.parametrize(
    "query,expected_doc",
    [
        ("how do I prepare ORS for my child", "gastro_and_hydration"),
        ("warning signs of dengue", "fever_and_infection"),
        ("what is the FAST test for stroke", "cardiac_and_stroke"),
        ("how much salt per day", "hypertension"),
        ("what to do for a burn", "first_aid"),
        ("danger signs in pregnancy", "maternal_and_child"),
    ],
)
def test_retrieval_finds_the_right_document(retriever, query, expected_doc):
    result = retriever.search(query, top_k=4)
    assert expected_doc in result.doc_ids
    assert result.grounded


def test_lay_terms_are_expanded(retriever):
    result = retriever.search("loose motion")
    assert "diarrhoea" in result.expanded_query.lower()
    assert "gastro_and_hydration" in result.doc_ids


@pytest.mark.parametrize(
    "query",
    ["what is the capital of France", "write me a python function to sort a list"],
)
def test_out_of_scope_queries_are_not_grounded(retriever, query):
    assert not retriever.search(query).grounded


def test_lexical_only_mode_is_reported(retriever):
    assert retriever.describe()["mode"] == "lexical"
