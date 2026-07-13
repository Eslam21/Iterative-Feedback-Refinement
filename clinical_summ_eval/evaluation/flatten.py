"""
Shared helpers for turning a model output into evaluable units.

Two consumers:
  - SelfCheckGPT needs a list of sentence-like units (for BERTScore) and a
    single candidate string (for MQAG).
  - DeepEval needs one declarative string passed as actual_output.

For "text" outputs we split on sentence boundaries with spaCy.
For "json" outputs we flatten the schema into "key > subkey: value" statements.
"""

from __future__ import annotations

import json
from typing import Iterable, List

# spaCy is heavy; load lazily so importing this module is cheap.
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# ─────────────────────────────────────────────────────────────
# JSON flattening
# ─────────────────────────────────────────────────────────────
def flatten_json(obj, prefix: str = "") -> List[str]:
    """Recursively flatten nested JSON into 'key > subkey: value' statements.

    - Skips None / empty-string / empty-list leaves.
    - Single-item lists are unwrapped (no '[0]' suffix).
    - Multi-item lists are indexed.
    """
    results: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix} > {key}" if prefix else key
            results.extend(flatten_json(value, prefix=full_key))
    elif isinstance(obj, list):
        items = [v for v in obj if v not in (None, "", [])]
        if len(items) == 1:
            results.extend(flatten_json(items[0], prefix=prefix))
        else:
            for i, item in enumerate(items):
                results.extend(flatten_json(item, prefix=f"{prefix}[{i}]"))
    else:
        if obj not in (None, ""):
            label = prefix.replace("_", " ")
            results.append(f"{label}: {obj}")
    return results


def _try_parse_json(text):
    """Return parsed JSON or None if the string isn't valid JSON."""
    if not isinstance(text, str):
        return text  # already a dict/list
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def to_sentences(output: str, fmt: str, min_token_count: int = 3) -> List[str]:
    """Convert a model output into a list of scorable units.

    fmt='text' → spaCy sentence split, dropping ultra-short fragments.
    fmt='json' → flattened field statements (empty list if JSON is malformed).
    """
    if fmt == "json":
        obj = _try_parse_json(output)
        if obj is None:
            return []
        return flatten_json(obj)

    if not output or not str(output).strip():
        return []
    doc = _get_nlp()(str(output))
    return [s.text.strip() for s in doc.sents if len(s) > min_token_count]


def to_candidate_text(output: str, fmt: str) -> str:
    """Convert a model output into a single string for metrics that want one blob.

    For JSON, joins flattened statements with periods.
    For text, returns the output unchanged.
    """
    if fmt == "json":
        obj = _try_parse_json(output)
        if obj is None:
            return ""
        stmts = flatten_json(obj)
        return ". ".join(stmts) + "." if stmts else ""
    return "" if output is None else str(output)
