from __future__ import annotations

import pytest

from filingqa.chunking import ChunkConfig, chunk_filing, verify_spans
from filingqa.models import Block, BlockType
from filingqa.retrieval import HashingEmbedder, HybridRetriever, RetrievalConfig
from filingqa.retrieval.expansion import QueryExpander
from filingqa.retrieval.indexes import BM25Index, reciprocal_rank_fusion
from filingqa.retrieval.rerank import LexicalOverlapReranker, NoOpReranker
from filingqa.retrieval.tokenize import normalise_number, tokenize

# -- tokenisation ----------------------------------------------------------


def test_money_survives_tokenisation():
    """A stock tokeniser turns "$2,755" into "2" and "755"."""
    tokens = tokenize("Revenue was $2,755 million")
    assert "money:2755" in tokens
    assert "2" not in tokens and "755" not in tokens


def test_percentages_survive():
    assert "pct:12.5" in tokenize("Margin improved to 12.5%")


def test_parenthesised_numbers_are_negative():
    """Filings write losses as (1,234). Dropping the parens flips the sign."""
    assert normalise_number("(1,234)") == "num:-1234"
    assert normalise_number("1,234") == "num:1234"


def test_stopwords_are_removed_but_domain_terms_kept():
    tokens = tokenize("the risk of a material adverse effect")
    assert "the" not in tokens
    assert "risk" in tokens and "material" in tokens


# -- query expansion -------------------------------------------------------


def test_revenue_expands_to_filing_vocabulary():
    """Apple says "net sales"; nobody asks for "net sales"."""
    result = QueryExpander().expand("What was revenue in 2024?")
    assert "net sales" in result.expanded


def test_analyst_jargon_expands():
    assert "net sales" in QueryExpander().expand("top-line results").expanded


def test_expansion_keeps_the_original_wording():
    """The synonym table is hand-built and therefore sometimes wrong.

    Additive expansion means a bad mapping costs precision; replacement would
    let it destroy the query.
    """
    result = QueryExpander().expand("What was revenue in 2024?")
    assert "revenue" in result.expanded


def test_noise_phrases_are_stripped():
    result = QueryExpander().expand("What does Apple say according to the filing?")
    assert "according to the filing" not in result.expanded.lower()


def test_expansion_is_bounded():
    expander = QueryExpander(max_expansions=3)
    result = expander.expand("revenue profit margin debt liquidity cash")
    assert len(result.added) <= 3


# -- BM25 ------------------------------------------------------------------


def test_bm25_finds_the_right_chunk(chunks):
    index = BM25Index().build(chunks)
    hits = index.search("supply chain concentration risk", top_k=5)
    assert hits
    assert "Supply chain" in chunks[hits[0][0]].text


def test_bm25_returns_nothing_for_unmatched_terms(chunks):
    index = BM25Index().build(chunks)
    assert index.search("photosynthesis chlorophyll", top_k=5) == []


def test_bm25_roundtrips(chunks, tmp_path):
    index = BM25Index().build(chunks)
    index.save(tmp_path / "bm25.pkl")
    loaded = BM25Index.load(tmp_path / "bm25.pkl")
    assert loaded.search("smartphones", top_k=3) == index.search("smartphones", top_k=3)


# -- fusion ----------------------------------------------------------------


def test_rrf_rewards_agreement():
    """A document both retrievers like should beat one only a single one does."""
    a = [(1, 9.0), (2, 8.0), (3, 7.0)]
    b = [(3, 0.9), (1, 0.8), (4, 0.7)]
    fused = dict(reciprocal_rank_fusion([a, b]))
    assert fused[1] > fused[4]
    assert fused[3] > fused[4]


def test_rrf_ignores_score_scale():
    """Score blending would need the two scales normalised; rank does not."""
    a = [(1, 1000.0), (2, 999.0)]
    b = [(2, 0.01), (1, 0.009)]
    small = reciprocal_rank_fusion([a, b])
    a_scaled = [(1, 1.0), (2, 0.999)]
    assert [i for i, _ in small] == [i for i, _ in reciprocal_rank_fusion([a_scaled, b])]


# -- chunking --------------------------------------------------------------


def test_tables_are_never_split(filing):
    """Splitting a table separates row labels from their numbers."""
    big_rows = [["Segment", "2024"]] + [[f"Row {i}", f"{i},000"] for i in range(400)]
    table_text = "\n".join(" | ".join(r) for r in big_rows)
    filing.text = filing.text + "\n\n" + table_text
    filing.blocks.append(
        Block(
            block_type=BlockType.TABLE,
            text=table_text,
            start=len(filing.text) - len(table_text),
            end=len(filing.text),
            item="Item 8",
            rows=big_rows,
        )
    )

    chunks = chunk_filing(filing, ChunkConfig(target_tokens=100))
    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) == 1
    assert "Row 399" in table_chunks[0].text


def test_chunks_do_not_straddle_item_boundaries(filing):
    """A chunk spanning two items would be labelled with the wrong section."""
    chunks = chunk_filing(filing, ChunkConfig(target_tokens=10_000))
    assert len({c.item for c in chunks}) > 1


def test_chunk_spans_resolve_against_the_filing(filing):
    chunks = chunk_filing(filing)
    assert verify_spans(chunks, [filing]) == []


def test_chunk_carries_its_provenance(filing):
    chunk = chunk_filing(filing)[0]
    assert chunk.company == "AAPL"
    assert chunk.fiscal_year == 2024
    assert "AAPL 2024 10-K" in chunk.citation_label


# -- retriever -------------------------------------------------------------


@pytest.fixture
def retriever(chunks):
    return HybridRetriever(
        chunks,
        embedder=HashingEmbedder(dim=128),
        reranker=NoOpReranker(),
        config=RetrievalConfig(use_bm25=True, use_dense=True),
    ).build(verbose=False)


def test_search_returns_results(retriever):
    assert retriever.search("supply chain risk", top_k=3)


def test_search_requires_a_built_index(chunks):
    with pytest.raises(RuntimeError, match="build"):
        HybridRetriever(chunks, embedder=HashingEmbedder(dim=64)).search("x")


def test_company_filter_applies_for_a_single_company(retriever):
    _results, trace = retriever.search_with_trace("Apple supply chain")
    assert trace.filters.get("company") == "AAPL"


def test_no_company_filter_when_two_are_named(retriever):
    """Filtering a comparison question to one company guarantees a wrong answer."""
    _results, trace = retriever.search_with_trace("Compare Apple and Ford on liquidity")
    assert "company" not in trace.filters


def test_filter_that_removes_everything_is_dropped(chunks):
    """A filter that matches nothing misfired; a bad filter is unrecoverable."""
    retriever = HybridRetriever(
        chunks, embedder=HashingEmbedder(dim=64), reranker=NoOpReranker()
    ).build(verbose=False)
    # Terms that do match the Apple-only fixture, but attributed to Ford - so
    # retrieval finds candidates and the company filter would erase all of them.
    results, trace = retriever.search_with_trace("Ford supply chain concentration risk")
    assert results, "a filter matching nothing must be dropped, not left to empty the results"
    assert trace.filters.get("applied") is False


def test_dense_defaults_off_for_the_offline_embedder(chunks):
    """Measured worse than BM25 alone, so it does not ship on by default."""
    retriever = HybridRetriever(chunks, embedder=HashingEmbedder(dim=64))
    assert retriever.config.use_dense is False


def test_explicit_config_overrides_the_offline_default(chunks):
    retriever = HybridRetriever(
        chunks, embedder=HashingEmbedder(dim=64), config=RetrievalConfig(use_dense=True)
    )
    assert retriever.config.use_dense is True


def test_reranker_receives_idf_from_bm25(chunks):
    reranker = LexicalOverlapReranker()
    HybridRetriever(chunks, embedder=HashingEmbedder(dim=64), reranker=reranker).build(
        verbose=False
    )
    assert reranker.idf


def test_index_roundtrips(retriever, tmp_path):
    retriever.save(tmp_path)
    loaded = HybridRetriever.load(tmp_path, embedder=retriever.embedder, reranker=NoOpReranker())
    assert len(loaded.chunks) == len(retriever.chunks)
    assert loaded.search("smartphones", top_k=3)
