"""Command line interface.

filingqa ingest                 # fetch and parse filings from EDGAR
filingqa index                  # build BM25 + dense indexes
filingqa ask "question"         # answer, with citations
filingqa goldset                # regenerate the evaluation set
filingqa evaluate               # run the ablation study
filingqa calibrate              # sweep refusal thresholds on dev
filingqa app                    # launch the Streamlit demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_PARSED = "data/parsed"
DEFAULT_INDEX = "data/index"
DEFAULT_GOLD = "evaluation/goldset/goldset.jsonl"
DEFAULT_RESULTS = "evaluation/results"


def _load_pipeline(args, *, need_answers: bool = True):
    from .answer import FilingQA, RefusalPolicy
    from .chunking import chunk_corpus
    from .ingest import load_corpus
    from .retrieval import HybridRetriever, get_embedder, get_reranker

    filings = load_corpus(args.parsed)
    chunks = chunk_corpus(filings)
    retriever = HybridRetriever(
        chunks,
        embedder=get_embedder(args.embedder),
        reranker=get_reranker(args.reranker),
    ).build(verbose=not args.quiet)
    if not need_answers:
        return filings, chunks, retriever
    return filings, chunks, FilingQA(retriever, filings, refusal=RefusalPolicy())


def cmd_ingest(args) -> int:
    from .ingest import EdgarClient, default_corpus, ingest

    client = EdgarClient(user_agent=args.user_agent, cache_dir=args.raw)
    refs = default_corpus(client, years=args.years)
    print(f"fetching {len(refs)} filings from EDGAR...")
    ingest(refs, client, parsed_dir=args.parsed)
    return 0


def cmd_index(args) -> int:
    from .chunking import chunk_corpus, verify_spans
    from .ingest import load_corpus
    from .retrieval import HybridRetriever, get_embedder, get_reranker

    filings = load_corpus(args.parsed)
    chunks = chunk_corpus(filings)
    problems = verify_spans(chunks, filings)
    if problems:
        print(f"{len(problems)} span problems; refusing to index", file=sys.stderr)
        for problem in problems[:5]:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"{len(filings)} filings, {len(chunks)} chunks")
    retriever = HybridRetriever(
        chunks, embedder=get_embedder(args.embedder), reranker=get_reranker(args.reranker)
    ).build()
    retriever.save(args.index)
    print(f"wrote index to {args.index}")
    return 0


def cmd_ask(args) -> int:
    _filings, _chunks, engine = _load_pipeline(args)
    answer = engine.ask(args.question)

    print()
    print(f"Q: {answer.question}")
    print()
    if not answer.answered:
        print(f"REFUSED ({answer.verdict.value})")
        print(f"  {answer.refusal_reason}")
        return 0

    print(answer.text)
    print()
    print("Sources:")
    for index, citation in enumerate(answer.citations, start=1):
        mark = "verified" if citation.verified else f"UNVERIFIED: {citation.verification_note}"
        print(f"  [{index}] {citation.label}  chars {citation.start}-{citation.end}  ({mark})")
    print()
    print(
        f"confidence {answer.confidence:.2f} | {answer.latency_ms:.0f}ms | "
        f"uncited claims: {len(answer.uncited_sentences)}"
    )
    return 0


def cmd_goldset(args) -> int:
    from .chunking import chunk_corpus
    from .eval import GoldSetBuilder, goldset_summary, save_goldset
    from .ingest import load_corpus

    filings = load_corpus(args.parsed)
    chunks = chunk_corpus(filings)
    questions = GoldSetBuilder(chunks, seed=args.seed).build()
    save_goldset(questions, args.gold)
    print(json.dumps(goldset_summary(questions), indent=2))
    print(f"\nwrote {len(questions)} questions to {args.gold}")
    return 0


def cmd_evaluate(args) -> int:
    from .chunking import chunk_corpus
    from .eval import load_goldset, run_ablations
    from .eval.report import write_report
    from .eval.runner import save_results
    from .ingest import load_corpus

    filings = load_corpus(args.parsed)
    chunks = chunk_corpus(filings)
    questions = load_goldset(args.gold)
    if args.split != "all":
        questions = [q for q in questions if q.split == args.split]

    print(f"evaluating on {len(questions)} questions (split={args.split})")
    results, answers = run_ablations(chunks, filings, questions, embedder_name=args.embedder)

    out = Path(args.results)
    save_results(
        results,
        out / f"ablations-{args.split}.json",
        extra={"split": args.split, "questions": len(questions), "embedder": args.embedder},
    )
    write_report(results, questions, answers, out / f"report-{args.split}.md")
    print(f"\nwrote {out}/ablations-{args.split}.json and report-{args.split}.md")
    return 0


def cmd_calibrate(args) -> int:
    from .chunking import chunk_corpus
    from .eval import load_goldset
    from .eval.tuning import save_tuning, sweep_bm25_b, sweep_refusal_thresholds
    from .ingest import load_corpus
    from .retrieval import HybridRetriever, get_embedder, get_reranker
    from .retrieval.rerank import NoOpReranker

    filings = load_corpus(args.parsed)
    chunks = chunk_corpus(filings)
    dev = [q for q in load_goldset(args.gold) if q.split == "dev"]
    print(f"calibrating on {len(dev)} dev questions (test split untouched)")

    embedder = get_embedder(args.embedder)
    bm25 = sweep_bm25_b(chunks, dev, embedder=embedder, reranker=NoOpReranker())
    print(f"  bm25 b: best {bm25.best_value} (recall@10 {bm25.best_score:.3f})")

    retriever = HybridRetriever(
        chunks, embedder=embedder, reranker=get_reranker(args.reranker)
    ).build(verbose=False)
    refusal = sweep_refusal_thresholds(retriever, filings, dev)
    print(f"  refusal: {json.dumps(refusal['best'], indent=4)}")

    save_tuning(
        {"bm25_b": bm25.to_dict(), "refusal": refusal},
        Path(args.results) / "calibration.json",
    )
    return 0


def cmd_app(args) -> int:
    import subprocess

    app = Path(__file__).resolve().parent.parent.parent / "app" / "demo.py"
    if not app.exists():
        print(f"demo app not found at {app}", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="filingqa",
        description="Answers questions about 10-Ks and shows you the page.",
    )
    parser.add_argument("--parsed", default=DEFAULT_PARSED)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument(
        "--embedder",
        default="auto",
        help="auto | hashing | openai | a sentence-transformers model name",
    )
    parser.add_argument("--reranker", default="auto", help="auto | lexical | none")
    parser.add_argument("-q", "--quiet", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="fetch and parse filings from EDGAR")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--years", type=int, default=3)
    p.add_argument(
        "--user-agent",
        default="filingqa-research contact@example.com",
        help="EDGAR requires a contact email; requests without one get a 403",
    )
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("index", help="build the retrieval indexes")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("ask", help="answer a question with citations")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("goldset", help="regenerate the evaluation set")
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(func=cmd_goldset)

    p = sub.add_parser("evaluate", help="run the ablation study")
    p.add_argument("--split", default="test", choices=["dev", "test", "all"])
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("calibrate", help="sweep parameters on the dev split")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("app", help="launch the Streamlit demo")
    p.set_defaults(func=cmd_app)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
