"""Turns evaluation results into the markdown report the README links to."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..models import Answer
from .goldset import GoldQuestion
from .metrics import recall_at_k


def _category_family(question: GoldQuestion) -> str:
    if not question.answerable:
        return question.category
    if question.category.endswith("_table"):
        return "numeric / table"
    if question.category.startswith("compare"):
        return "cross-filing comparison"
    return "prose / topic"


def write_report(
    results: list[Any],
    questions: list[GoldQuestion],
    answers: dict[str, Answer],
    path: Path | str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    full = next((r for r in results if r.is_full_system), results[-1] if results else None)
    lines: list[str] = ["# Retrieval quality report", ""]

    answerable = [q for q in questions if q.answerable]
    lines += [
        f"Questions: **{len(questions)}** ({len(answerable)} answerable, "
        f"{len(questions) - len(answerable)} unanswerable).",
        "",
    ]

    if full:
        lines += [
            "## Headline",
            "",
            f"- **recall@10** {full.retrieval['recall_at_k'].get('10', 0):.3f}",
            f"- **MRR** {full.retrieval['mrr']:.3f}",
            f"- **nDCG@10** {full.retrieval['ndcg_at_k'].get('10', 0):.3f}",
            f"- **refusal recall** {full.refusal['refusal_recall']:.3f} "
            f"(precision {full.refusal['refusal_precision']:.3f})",
            f"- **answer rate on answerable questions** "
            f"{full.refusal['answer_rate_on_answerable']:.3f}",
            f"- **citation accuracy** {full.citations['citation_accuracy']:.3f} "
            f"over {full.citations['total_citations']} citations",
            f"- **answers containing an uncited claim** "
            f"{full.citations['answers_with_uncited_claims']}",
            f"- **latency** p50 {full.latency_ms['p50']:.0f}ms / "
            f"p95 {full.latency_ms['p95']:.0f}ms",
            "",
        ]

    # -- ablation table ----------------------------------------------------
    lines += [
        "## Ablations",
        "",
        "| arm | recall@10 | recall@5 | MRR | nDCG@10 | refusal recall | p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        label = f"**{result.label}**" if result.is_full_system else result.label
        lines.append(
            f"| {label} | {result.retrieval['recall_at_k'].get('10', 0):.3f} "
            f"| {result.retrieval['recall_at_k'].get('5', 0):.3f} "
            f"| {result.retrieval['mrr']:.3f} "
            f"| {result.retrieval['ndcg_at_k'].get('10', 0):.3f} "
            f"| {result.refusal['refusal_recall']:.3f} "
            f"| {result.latency_ms['p95']:.0f} ms |"
        )
    lines.append("")

    for result in results:
        lines.append(f"- **{result.label}** — {result.description}")
    lines.append("")

    # -- per-category ------------------------------------------------------
    if answers:
        buckets: dict[str, list[float]] = defaultdict(list)
        for question in answerable:
            answer = answers.get(question.question_id)
            if answer is None:
                continue
            hit = recall_at_k([s.chunk.chunk_id for s in answer.retrieved], question.relevance, 10)
            buckets[_category_family(question)].append(hit)

        lines += [
            "## Where it works and where it does not",
            "",
            "| question type | n | recall@10 |",
            "|---|---:|---:|",
        ]
        for family, hits in sorted(buckets.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            lines.append(f"| {family} | {len(hits)} | {sum(hits) / len(hits):.3f} |")
        lines.append("")

        refused_by_reason: dict[str, int] = defaultdict(int)
        for question in questions:
            answer = answers.get(question.question_id)
            if answer and not answer.answered:
                refused_by_reason[answer.verdict.value] += 1
        if refused_by_reason:
            lines += ["## Refusals by type", "", "| verdict | count |", "|---|---:|"]
            for verdict, count in sorted(refused_by_reason.items(), key=lambda kv: -kv[1]):
                lines.append(f"| {verdict} | {count} |")
            lines.append("")

        wrongly_refused = [
            q
            for q in answerable
            if (a := answers.get(q.question_id)) is not None and not a.answered
        ]
        if wrongly_refused:
            lines += [
                "## Answerable questions the system refused",
                "",
                "These are the false refusals - the cost of the calibration.",
                "",
            ]
            for question in wrongly_refused[:10]:
                answer = answers[question.question_id]
                lines.append(
                    f"- *{question.question}* — {answer.verdict.value}: "
                    f"{answer.refusal_reason[:140]}"
                )
            lines.append("")

    path.write_text("\n".join(lines) + "\n")
    return path
