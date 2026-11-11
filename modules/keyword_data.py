"""Small JSON data loaders for skill vocabulary and generic wording."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


_DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=1)
def load_skill_terms() -> tuple[str, ...]:
    """Return all curated skills and phrases in their data-file order."""
    with (_DATA_DIRECTORY / "skills.json").open(encoding="utf-8") as skills_file:
        skills = json.load(skills_file)
    if not isinstance(skills, dict) or not all(
        isinstance(category, str)
        and isinstance(terms, list)
        and all(isinstance(term, str) for term in terms)
        for category, terms in skills.items()
    ):
        raise ValueError("Skill data must group lists of text terms by category.")
    return tuple(term for terms in skills.values() for term in terms)


@lru_cache(maxsize=1)
def load_extra_stopwords() -> frozenset[str]:
    """Load generic job wording that should not be suggested as a skill."""
    with (_DATA_DIRECTORY / "stopwords_extra.json").open(
        encoding="utf-8"
    ) as stopwords_file:
        stopwords = json.load(stopwords_file)
    if not isinstance(stopwords, list) or not all(
        isinstance(word, str) for word in stopwords
    ):
        raise ValueError("Extra stopwords must be a JSON list of text terms.")
    return frozenset(word.lower() for word in stopwords)
