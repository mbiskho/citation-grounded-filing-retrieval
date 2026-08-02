"""Streamlit demo: ask a question, see the answer, click through to the page.

The whole point of the product is the last part. An answer is a means of
finding the span; the span is what goes in the memo. So the citation is not a
footnote in this UI - it is the primary object, and every claim is one click
from the filing text that supports it, with the exact characters highlighted.

    filingqa app
    # or: streamlit run app/demo.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from filingqa.answer import FilingQA, RefusalPolicy  # noqa: E402
from filingqa.chunking import chunk_corpus  # noqa: E402
from filingqa.ingest import load_corpus  # noqa: E402
from filingqa.models import Answer, Filing  # noqa: E402
from filingqa.retrieval import HybridRetriever, get_embedder, get_reranker  # noqa: E402

st.set_page_config(page_title="Filing QA", page_icon="§", layout="wide")

CONTEXT_CHARS = 700

EXAMPLES = [
    "What supplier and manufacturing dependencies does Apple flag as a risk in FY2024?",
    "How does JPMorgan Chase describe its liquidity position in FY2024?",
    "What material legal proceedings was Ford involved in during FY2024?",
    "How sensitive is Apple to a change in interest rates as of FY2024?",
    "What was Tesla's revenue in FY2024?",
    "What price target do analysts have for Apple stock?",
]


@st.cache_resource(show_spinner="Loading filings and building indexes...")
def load_engine(embedder_name: str, reranker_name: str):
    filings = load_corpus(ROOT / "data" / "parsed")
    chunks = chunk_corpus(filings)
    retriever = HybridRetriever(
        chunks,
        embedder=get_embedder(embedder_name),
        reranker=get_reranker(reranker_name),
    ).build(verbose=False)
    engine = FilingQA(retriever, filings, refusal=RefusalPolicy())
    return engine, {f.doc_id: f for f in filings}, len(chunks)


def highlight(filing: Filing, start: int, end: int) -> str:
    """Filing text around a span, with the span marked."""
    left = max(0, start - CONTEXT_CHARS)
    right = min(len(filing.text), end + CONTEXT_CHARS)
    before = html.escape(filing.text[left:start])
    span = html.escape(filing.text[start:end])
    after = html.escape(filing.text[end:right])
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(filing.text) else ""
    return (
        "<div style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
        "font-size:0.82rem;line-height:1.6;white-space:pre-wrap;"
        "background:#fafafa;border:1px solid #e6e6e6;border-radius:6px;"
        "padding:0.85rem;max-height:340px;overflow-y:auto'>"
        f"{prefix}{before}"
        "<mark style='background:#ffe58f;padding:1px 2px;border-radius:2px'>"
        f"{span}</mark>{after}{suffix}</div>"
    )


def render_answer(answer: Answer, filings: dict[str, Filing]) -> None:
    if not answer.answered:
        st.warning(f"**Refused** — {answer.verdict.value}")
        st.caption(answer.refusal_reason)
        st.caption(
            "Refusing is the designed behaviour here, not a failure. An "
            "unsourced number is worse than no number."
        )
        if answer.retrieved:
            with st.expander(f"What it retrieved anyway ({len(answer.retrieved)} passages)"):
                for scored in answer.retrieved[:5]:
                    st.caption(f"`{scored.chunk.chunk_id}` — {scored.chunk.citation_label}")
                    st.text(scored.chunk.text[:280].replace("\n", " "))
        return

    st.markdown(f"#### {answer.text}")

    cols = st.columns(4)
    cols[0].metric("Citations", len(answer.citations))
    cols[1].metric(
        "Verified", f"{sum(1 for c in answer.citations if c.verified)}/{len(answer.citations)}"
    )
    cols[2].metric("Uncited claims", len(answer.uncited_sentences))
    cols[3].metric("Latency", f"{answer.latency_ms:.0f} ms")

    st.divider()
    st.markdown("##### Sources")

    for index, citation in enumerate(answer.citations, start=1):
        filing = filings.get(citation.doc_id)
        badge = "verified" if citation.verified else f"UNVERIFIED — {citation.verification_note}"
        header = f"**[{index}]** {citation.label} · chars {citation.start:,}–{citation.end:,} · {badge}"
        with st.expander(header, expanded=index == 1):
            if filing is None:
                st.error(f"filing {citation.doc_id} is not loaded")
                continue
            st.markdown(highlight(filing, citation.start, citation.end), unsafe_allow_html=True)
            st.caption(f"Source: [{filing.ref.url}]({filing.ref.url})")


def main() -> None:
    st.title("Citation-Grounded Filing Retrieval")
    st.caption("Answers questions about 10-Ks and shows you the page.")

    with st.sidebar:
        st.header("Configuration")
        embedder_name = st.selectbox(
            "Embedder",
            ["auto", "hashing"],
            help=(
                "'auto' uses sentence-transformers when installed and falls back "
                "to the dependency-free hashing embedder otherwise."
            ),
        )
        reranker_name = st.selectbox("Reranker", ["auto", "lexical", "none"])
        st.divider()

    try:
        engine, filings, n_chunks = load_engine(embedder_name, reranker_name)
    except FileNotFoundError:
        st.error("No parsed filings found. Run `filingqa ingest` first.")
        st.stop()

    with st.sidebar:
        st.metric("Filings", len(filings))
        st.metric("Chunks", f"{n_chunks:,}")
        st.caption("Corpus: " + ", ".join(sorted({f.ref.ticker for f in filings.values()})))
        years = sorted({f.ref.fiscal_year for f in filings.values()})
        st.caption(f"Fiscal years {years[0]}–{years[-1]}")
        st.divider()
        st.caption(
            "Questions naming a company or year outside the corpus are refused "
            "before retrieval runs."
        )

    if "question" not in st.session_state:
        st.session_state.question = EXAMPLES[0]

    st.markdown("**Try one:**")
    example_cols = st.columns(3)
    for i, example in enumerate(EXAMPLES):
        if example_cols[i % 3].button(
            example[:52] + ("…" if len(example) > 52 else ""),
            key=f"ex{i}",
            use_container_width=True,
        ):
            st.session_state.question = example

    question = st.text_input("Question", key="question")

    if question:
        with st.spinner("Retrieving..."):
            answer = engine.ask(question)
        render_answer(answer, filings)

        with st.expander("Retrieval detail"):
            if answer.trace:
                st.json(answer.trace)
            st.markdown("**Top passages**")
            for scored in answer.retrieved[:10]:
                st.caption(
                    f"`{scored.chunk.chunk_id}` · {scored.chunk.citation_label} · "
                    f"{'table' if scored.chunk.is_table else 'prose'} · score {scored.score:.3f}"
                )


if __name__ == "__main__":
    main()
