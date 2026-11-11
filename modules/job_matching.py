"""Compare resume keywords and text with a job description."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.alias_loader import canonicalize_aliases
from modules.keyword_extraction import extract_keywords, is_relevant_keyword
from modules.text_preprocessing import (
    TOKEN_PATTERN,
    get_english_stopwords,
    normalize_keyword,
    tokenize_text,
)

JOB_KEYWORD_LIMIT = 25


def _input_fingerprint(text: str) -> str:
    """Create a non-secret SHA-256 identifier for an exact comparison input."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def comparison_inputs_match(
    result: dict[str, Any] | None,
    resume_text: str,
    job_description: str,
) -> bool:
    """Return whether a saved comparison belongs to the current input texts."""
    if result is None:
        return False
    fingerprints = result.get("input_fingerprints")
    if not isinstance(fingerprints, dict):
        return False
    return fingerprints == {
        "resume": _input_fingerprint(resume_text),
        "job_description": _input_fingerprint(job_description.strip()),
    }


def comparison_input_diagnostics(
    result: dict[str, Any],
    resume_text: str,
    job_description: str,
    active_resume_text: str,
    current_job_description: str,
) -> dict[str, bool]:
    """Report input equality and source checks without exposing input contents."""
    fingerprints = result.get("input_fingerprints", {})
    return {
        "inputs_identical": resume_text == job_description,
        "fingerprints_match": (
            isinstance(fingerprints, dict)
            and fingerprints.get("resume") == _input_fingerprint(resume_text)
            and fingerprints.get("job_description")
            == _input_fingerprint(job_description)
        ),
        "resume_from_active_analysis": resume_text == active_resume_text,
        "job_from_current_text_area": (
            job_description == current_job_description.strip()
        ),
    }


def _unique_keywords(keywords: list[str]) -> list[str]:
    """Preserve keyword order while removing equivalent normalized spellings."""
    unique: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        normalized = normalize_keyword(keyword)
        if normalized and is_relevant_keyword(normalized) and normalized not in seen:
            unique.append(normalized)
            seen.add(normalized)
    return unique


def _contains_keyword(resume_tokens: list[str], keyword: str) -> bool:
    """Check for a normalized keyword as a whole token or contiguous phrase."""
    keyword_tokens = tokenize_text(keyword)
    width = len(keyword_tokens)
    return bool(width) and any(
        resume_tokens[index : index + width] == keyword_tokens
        for index in range(len(resume_tokens) - width + 1)
    )


def compare_resume_to_job(resume_text: str, job_description: str) -> dict[str, Any]:
    """Return keyword coverage and TF-IDF cosine similarity as separate metrics."""
    if not isinstance(resume_text, str) or not isinstance(job_description, str):
        raise TypeError("Resume and job description must both be strings.")
    if not resume_text.strip():
        raise ValueError("Resume text must not be empty.")
    if not job_description.strip():
        raise ValueError("Job description must not be empty.")

    job_keywords = _unique_keywords(
        extract_keywords(job_description, max_keywords=JOB_KEYWORD_LIMIT)[
            "keyword"
        ].tolist()
    )
    canonical_resume = canonicalize_aliases(resume_text)
    canonical_job_description = canonicalize_aliases(job_description)
    resume_tokens = tokenize_text(canonical_resume)
    matched = [
        keyword for keyword in job_keywords if _contains_keyword(resume_tokens, keyword)
    ]
    missing = [
        keyword for keyword in job_keywords if not _contains_keyword(resume_tokens, keyword)
    ]

    # Coverage uses only unique, relevant keywords extracted from the job description.
    coverage = len(matched) / len(job_keywords) * 100 if job_keywords else None

    similarity = 0.0
    vectorizer = TfidfVectorizer(
        token_pattern=TOKEN_PATTERN,
        lowercase=True,
        stop_words=list(get_english_stopwords()),
        ngram_range=(1, 1),
    )
    try:
        vectors = vectorizer.fit_transform(
            [canonical_resume, canonical_job_description]
        )
        resume_vector = vectors[0:1]
        job_description_vector = vectors[1:2]
        similarity = float(
            cosine_similarity(resume_vector, job_description_vector)[0, 0]
        )
    except ValueError:
        # Non-empty inputs can still contain no usable tokens.
        similarity = 0.0

    if math.isclose(similarity, 1.0, rel_tol=0.0, abs_tol=1e-12):
        similarity_percentage = 100.0
    else:
        similarity_percentage = min(round(similarity * 100, 1), 99.9)

    result = {
        "job_keywords": job_keywords,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "keyword_coverage": round(coverage, 1) if coverage is not None else None,
        "cosine_similarity": similarity_percentage,
        "input_fingerprints": {
            "resume": _input_fingerprint(resume_text),
            "job_description": _input_fingerprint(job_description),
        },
        "job_keyword_limit": JOB_KEYWORD_LIMIT,
    }
    if not job_keywords:
        result["message"] = (
            "No relevant job-description keywords were found, so keyword match "
            "coverage is not available."
        )
    return result
