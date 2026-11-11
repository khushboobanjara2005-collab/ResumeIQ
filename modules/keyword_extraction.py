"""TF-IDF keyword extraction with support for common multiword skills."""

from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from modules.alias_loader import canonicalize_aliases
from modules.keyword_data import load_extra_stopwords, load_skill_terms
from modules.text_preprocessing import (
    TOKEN_PATTERN,
    get_english_stopwords,
    normalize_keyword,
    normalize_text,
    preprocess_text,
)


# These are filtered as standalone terms; configured skill phrases such as
# "computer vision" remain valid because filtering applies to the whole keyword.
CONFIGURABLE_KEYWORD_STOP_WORDS = frozenset(
    {"computer", "applications", "coursework", "undergraduate"}
)

_RESUME_HEADINGS = frozenset(
    {
        "summary",
        "professional summary",
        "education",
        "experience",
        "work experience",
        "technical skills",
        "skills",
        "projects",
        "project",
        "sample",
        "test",
        "contact",
        "objective",
    }
)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)
_URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)")
_NAME_LINE_PATTERN = re.compile(
    r"[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?){1,2}"
)


def is_relevant_keyword(keyword: str) -> bool:
    """Return false for generic job wording without rejecting multiword skills."""
    normalized = normalize_keyword(keyword)
    return (
        bool(normalized)
        and not normalized.isdigit()
        and normalized not in load_extra_stopwords()
        and normalized not in CONFIGURABLE_KEYWORD_STOP_WORDS
    )


def _keyword_source_text(text: str) -> str:
    """Remove resume metadata from keyword scoring without changing the stored text."""
    lines = text.splitlines()
    nonempty_positions = [index for index, line in enumerate(lines) if line.strip()]
    contact_lines = {
        index
        for index, line in enumerate(lines)
        if _EMAIL_PATTERN.search(line)
        or _PHONE_PATTERN.search(line)
        or _URL_PATTERN.search(line)
    }
    next_nonempty_line = {
        index: next(
            (position for position in nonempty_positions if position > index), None
        )
        for index in nonempty_positions
    }

    cleaned_lines = []
    for index, line in enumerate(lines):
        # Contact headers are document metadata, not resume skills.
        if index in contact_lines:
            continue
        line = _EMAIL_PATTERN.sub(" ", line)
        line = _URL_PATTERN.sub(" ", line)
        line = _PHONE_PATTERN.sub(" ", line)
        line = re.sub(r"\s+", " ", line).strip(" \t|,;")
        if not line or normalize_keyword(line) in _RESUME_HEADINGS:
            continue

        if _NAME_LINE_PATTERN.fullmatch(line) and (
            index in contact_lines
            or next_nonempty_line.get(index) in contact_lines
        ):
            continue

        # Keep content on the same line as a section label, but discard the label.
        line = re.sub(
            r"^(?:technical\s+)?skills?\s*[:\-]\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _text_segments(text: str) -> list[str]:
    """Split text into sentence-like documents for TF-IDF weighting."""
    segments = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [segment for segment in segments if preprocess_text(segment)]


def extract_keywords(text: str, max_keywords: int = 30) -> pd.DataFrame:
    """Return useful phrases and TF-IDF-ranked terms as unique keyword/score rows."""
    if not isinstance(text, str):
        raise TypeError("Resume text must be a string.")
    if max_keywords < 1:
        raise ValueError("max_keywords must be at least 1.")

    keyword_text = canonicalize_aliases(_keyword_source_text(text))
    normalized = normalize_text(keyword_text)
    if not normalized:
        return pd.DataFrame(columns=["keyword", "score"])

    # Curated phrases and known tools are retained even when they occur once.
    detected_phrases = []
    seen_phrases: set[str] = set()
    for phrase in load_skill_terms():
        canonical_phrase = normalize_keyword(phrase)
        # A standalone C should not also be detected inside the C++ skill name.
        right_boundary = r"(?![a-z0-9+#])" if canonical_phrase == "c" else r"(?![a-z0-9])"
        phrase_pattern = (
            rf"(?<![a-z0-9]){re.escape(canonical_phrase)}{right_boundary}"
        )
        if (
            canonical_phrase not in seen_phrases
            and re.search(phrase_pattern, normalized)
        ):
            detected_phrases.append(canonical_phrase)
            seen_phrases.add(canonical_phrase)

    segments = _text_segments(keyword_text)
    # Suppress fragments of compounds such as "front-end" and "entry-level".
    hyphenated_components = {
        component
        for match in re.finditer(
            r"(?<![a-z0-9])([a-z0-9]+)[\-‐‑‒–—−]([a-z0-9]+)(?![a-z0-9])",
            keyword_text.lower(),
        )
        for component in match.groups()
    }
    cleaned_segments = [" ".join(preprocess_text(segment)) for segment in segments]
    ranked_terms: list[tuple[str, float]] = []
    if cleaned_segments:
        vectorizer = TfidfVectorizer(
            token_pattern=TOKEN_PATTERN,
            lowercase=False,
            stop_words=list(get_english_stopwords()),
            ngram_range=(1, 1),
            max_features=500,
            sublinear_tf=True,
        )
        try:
            matrix = vectorizer.fit_transform(cleaned_segments)
            terms = vectorizer.get_feature_names_out()
            scores = matrix.sum(axis=0).A1
            ranked_terms = sorted(
                (
                    (term, float(score))
                    for term, score in zip(terms, scores)
                    if is_relevant_keyword(term)
                ),
                key=lambda item: (-item[1], item[0]),
            )
        except ValueError:
            # Empty vocabularies can occur for punctuation-only or stopword-only text.
            ranked_terms = []

    keywords: list[tuple[str, float]] = []
    seen: set[str] = set()
    scores_by_term = dict(ranked_terms)
    phrase_components = {
        component
        for phrase in detected_phrases
        if " " in phrase
        for component in phrase.split()
    }

    for phrase in detected_phrases:
        phrase_score = sum(
            scores_by_term.get(word, 0.0) for word in phrase.split()
        ) / len(phrase.split())
        keywords.append((phrase, phrase_score or 1.0))
        seen.add(phrase)

    for term, score in ranked_terms:
        canonical_term = normalize_keyword(term)
        if (
            not canonical_term
            or canonical_term in seen
            or not is_relevant_keyword(canonical_term)
            or canonical_term in phrase_components
            or canonical_term in hyphenated_components
        ):
            continue
        keywords.append((canonical_term, score))
        seen.add(canonical_term)
        if len(keywords) >= max_keywords:
            break

    return pd.DataFrame(keywords[:max_keywords], columns=["keyword", "score"])
