"""Load and apply the project's explainable skill-name aliases."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_aliases() -> dict[str, str]:
    """Read alias-to-canonical mappings from the project's JSON data file."""
    path = Path(__file__).resolve().parent.parent / "data" / "aliases.json"
    with path.open(encoding="utf-8") as aliases_file:
        aliases = json.load(aliases_file)
    if not isinstance(aliases, dict) or not all(
        isinstance(alias, str) and isinstance(canonical, str)
        for alias, canonical in aliases.items()
    ):
        raise ValueError("Alias data must be a JSON object of text mappings.")
    return {alias.lower(): canonical.lower() for alias, canonical in aliases.items()}


def canonicalize_aliases(text: str) -> str:
    """Replace whole-word skill variants without changing parts of larger words."""
    if not isinstance(text, str):
        raise TypeError("Text to canonicalize must be a string.")

    canonical_text = text.lower()
    # Longer aliases are replaced first so short aliases cannot cut up compounds.
    aliases = load_aliases()
    for alias in sorted(aliases, key=len, reverse=True):
        left_boundary = r"(?<![a-z0-9.])" if alias == "js" else r"(?<![a-z0-9])"
        pattern = rf"{left_boundary}{re.escape(alias)}(?![a-z0-9])"
        canonical_text = re.sub(pattern, aliases[alias], canonical_text)
    return canonical_text
