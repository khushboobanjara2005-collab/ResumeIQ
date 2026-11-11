"""Downloadable result-report helpers."""

from __future__ import annotations

import pandas as pd


def build_results_csv(analysis: dict, match_result: dict | None) -> bytes:
    """Create a CSV containing the current analysis and optional comparison."""
    rows: list[dict[str, str | float | int]] = [
        {"section": "Summary", "keyword": "analysis_status", "value": "Complete"},
        {
            "section": "Summary",
            "keyword": "total_keywords_extracted",
            "value": len(analysis["keywords"]),
        },
    ]
    for entry in analysis["keyword_scores"]:
        rows.append(
            {"section": "Resume keyword", "keyword": entry["keyword"], "value": entry["score"]}
        )

    if match_result:
        rows.extend(
            [
                {
                    "section": "Summary",
                    "keyword": "matched_keyword_count",
                    "value": len(match_result["matched_keywords"]),
                },
                {
                    "section": "Summary",
                    "keyword": "missing_keyword_count",
                    "value": len(match_result["missing_keywords"]),
                },
                {
                    "section": "Summary",
                    "keyword": "keyword_match_percentage",
                    "value": (
                        match_result["keyword_coverage"]
                        if match_result["keyword_coverage"] is not None
                        else "Not available"
                    ),
                },
                {
                    "section": "Summary",
                    "keyword": "cosine_similarity_percentage",
                    "value": match_result["cosine_similarity"],
                },
            ]
        )
        rows.extend(
            {"section": "Matched keyword", "keyword": word, "value": ""}
            for word in match_result["matched_keywords"]
        )
        rows.extend(
            {"section": "Missing keyword", "keyword": word, "value": ""}
            for word in match_result["missing_keywords"]
        )

    return pd.DataFrame(rows, columns=["section", "keyword", "value"]).to_csv(
        index=False
    ).encode("utf-8")
