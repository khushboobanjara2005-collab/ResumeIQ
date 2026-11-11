"""Text cleanup and tokenization helpers used by ResumeIQ."""

from __future__ import annotations

import re
from pathlib import Path

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 10

# Keep common technical spellings intact while still tokenizing ordinary words.
TOKEN_PATTERN = (
    r"(?:scikit-learn|tf-idf|ci/cd|c\+\+|c#|f#|\.net|"
    r"[a-z0-9]+(?:[.+#][a-z0-9]+)*[+#]?)"
)
_tokenizer = RegexpTokenizer(TOKEN_PATTERN, flags=re.IGNORECASE)
_PRESERVED_TERMS = {
    "scikitlearn": "scikit-learn",
    "tfidf": "tf-idf",
}
_FALLBACK_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have he in is it its of on or that the
    to was were will with this these those you your i we they their our
    """.split()
)


def normalize_text(text: str) -> str:
    """Lowercase text and remove punctuation that is not useful in technical terms."""
    if not isinstance(text, str):
        raise TypeError("Text to normalize must be a string.")

    normalized = text.lower()
    # Temporarily protect technical names while punctuation is cleaned up.
    hyphens = r"\s.\-‐‑‒–—−"
    normalized = re.sub(rf"scikit[{hyphens}]+learn", "scikitlearn", normalized)
    normalized = re.sub(rf"tf[{hyphens}]+idf", "tfidf", normalized)
    normalized = re.sub(r"[^a-z0-9+#./\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for placeholder, original in _PRESERVED_TERMS.items():
        normalized = normalized.replace(placeholder, original)
    return normalized


def normalize_keyword(keyword: str) -> str:
    """Return one canonical, lowercase form for matching an extracted keyword."""
    return " ".join(tokenize_text(keyword))


def get_english_stopwords() -> frozenset[str]:
    """Load NLTK stopwords, using a small local fallback if its corpus is absent."""
    try:
        return frozenset(word.lower() for word in stopwords.words("english"))
    except LookupError:
        # Keep analysis usable without downloading the NLTK stopwords corpus.
        return frozenset(ENGLISH_STOP_WORDS) | _FALLBACK_STOPWORDS


def tokenize_text(text: str) -> list[str]:
    """Tokenize normalized text with NLTK's resource-free regular-expression tokenizer."""
    return _tokenizer.tokenize(normalize_text(text))


def preprocess_text(text: str) -> list[str]:
    """Normalize, tokenize, and remove English stopwords from text."""
    blocked_words = get_english_stopwords()
    return [
        token
        for token in tokenize_text(text)
        if token not in blocked_words and token.strip(".+#")
    ]


def extract_text_from_upload(filename: str, file_bytes: bytes) -> str:
    """Read a supported text or PDF upload without saving it to disk."""
    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError("This file is larger than 5 MB. Upload a smaller file.")

    suffix = Path(filename).suffix.lower()

    if suffix == ".txt":
        try:
            return file_bytes.decode("utf-8-sig").strip()
        except UnicodeDecodeError as error:
            raise ValueError("This text file is not valid UTF-8.") from error

    if suffix == ".pdf":
        import pymupdf as fitz

        try:
            document = fitz.open(stream=file_bytes, filetype="pdf")
        except (fitz.FileDataError, RuntimeError, ValueError) as error:
            raise ValueError("The PDF could not be read. Try another PDF file.") from error

        with document:
            if len(document) > MAX_PDF_PAGES:
                raise ValueError(
                    "This PDF has more than 10 pages. Upload a PDF with 10 pages or fewer."
                )
            try:
                return "\n".join(page.get_text() for page in document).strip()
            except (fitz.FileDataError, RuntimeError, ValueError) as error:
                raise ValueError(
                    "The PDF could not be read. Try another PDF file."
                ) from error

    raise ValueError("Unsupported file type. Upload a .txt or .pdf file.")
