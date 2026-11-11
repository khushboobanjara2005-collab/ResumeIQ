# Expected labels were written by hand; review them before reporting results.

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.job_matching import compare_resume_to_job


EVALUATION_FILE = Path(__file__).with_name("eval_pairs.json")
RESULTS_FILE = Path(__file__).with_name("results.md")


def _score_pair(
    predicted: set[str], expected: set[str]
) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1 for the matched keyword labels."""
    correct = len(predicted & expected)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(expected) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def evaluate_pairs() -> list[dict[str, Any]]:
    """Compare every labeled pair and return its measured keyword scores."""
    with EVALUATION_FILE.open(encoding="utf-8") as pairs_file:
        pairs = json.load(pairs_file)
    if not isinstance(pairs, list):
        raise ValueError("Evaluation data must be a JSON list of pairs.")

    results = []
    for pair in pairs:
        comparison = compare_resume_to_job(pair["resume"], pair["job_description"])
        expected = set(pair["expected_matched"])
        predicted = set(comparison["matched_keywords"])
        precision, recall, f1 = _score_pair(predicted, expected)
        results.append(
            {
                "id": pair["id"],
                "role": pair["role"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted": sorted(predicted),
            }
        )
    return results


def _format_results(results: list[dict[str, Any]]) -> str:
    """Build the same readable Markdown table for the terminal and report."""
    lines = [
        "| Pair | Role | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['id']} | {result['role']} | "
            f"{result['precision']:.3f} | {result['recall']:.3f} | "
            f"{result['f1']:.3f} |"
        )

    count = len(results)
    averages = {
        metric: sum(result[metric] for result in results) / count if count else 0.0
        for metric in ("precision", "recall", "f1")
    }
    lines.append(
        f"| **Average** | **{count} pairs** | "
        f"**{averages['precision']:.3f}** | **{averages['recall']:.3f}** | "
        f"**{averages['f1']:.3f}** |"
    )
    return "\n".join(lines)


def main() -> None:
    results = evaluate_pairs()
    table = _format_results(results)
    report = (
        "# Evaluation Results\n\n"
        "Precision, recall, and F1 compare extracted matched keywords with the "
        "hand-written `expected_matched` labels. Results describe this small "
        "fictional sample only.\n\n"
        f"{table}\n"
    )
    RESULTS_FILE.write_text(report, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
