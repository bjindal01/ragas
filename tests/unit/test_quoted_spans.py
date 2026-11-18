"""
Unit tests for the quoted spans alignment metric.

These tests are written using pytest and cover several common cases:
    - A perfect match where the quoted span appears in the sources.
    - A mismatch where the quoted span does not appear in the sources.
    - Case and whitespace variations to verify normalization logic.
    - Answers with no quoted spans to ensure the score is zero and total is zero.

To run these tests, install pytest and run `pytest` in the repository root.
"""

import pytest

from ragas.metrics.quoted_spans import quoted_spans_alignment


def test_perfect_match():
    """Quoted span matches exactly in the source."""
    answers = ['Paris is "the capital of France".']
    sources = [["The capital of France is Paris."]]
    result = quoted_spans_alignment(answers, sources)
    assert result["citation_alignment_quoted_spans"] == 1.0
    assert result["matched"] == 1.0
    assert result["total"] == 1.0


def test_mismatch_detected():
    """Quoted span does not appear in the sources."""
    answers = ['GDP was "$2.9T" in 2023.']
    sources = [["…GDP was $2.7T in 2023 per WB…"]]
    result = quoted_spans_alignment(answers, sources, min_len=1)
    assert result["citation_alignment_quoted_spans"] == 0.0
    assert result["matched"] == 0.0
    assert result["total"] == 1.0


def test_mixed_case_and_whitespace():
    """Matching should be case-insensitive and handle extra whitespace."""
    answers = ['Result: "Delta   E    = mc  ^ 2".']
    sources = [["…delta e = mc ^ 2 holds…"]]
    result = quoted_spans_alignment(answers, sources)
    assert result["citation_alignment_quoted_spans"] == 1.0


def test_no_quotes_returns_zero_with_zero_denominator():
    """An answer with no quoted spans should yield score 0.0 and total 0."""
    answers = ["No quotes here."]
    sources = [["Irrelevant."]]
    result = quoted_spans_alignment(answers, sources)
    assert result["citation_alignment_quoted_spans"] == 0.0
    assert result["total"] == 0.0


def test_length_mismatch_raises():
    """answers and sources must be aligned one-to-one."""
    with pytest.raises(ValueError, match="same length"):
        quoted_spans_alignment(["a"], [])


def test_casefold_disabled_is_case_sensitive():
    answers = ['He said "The Quick Brown Fox".']
    sources = [["the quick brown fox jumps"]]
    assert (
        quoted_spans_alignment(answers, sources)["citation_alignment_quoted_spans"]
        == 1.0
    )
    result = quoted_spans_alignment(answers, sources, casefold=False)
    assert result["matched"] == 0.0
    assert result["total"] == 1.0


def test_short_spans_are_ignored_by_min_len():
    answers = ['The value was "42" and "a much longer quote".']
    sources = [["here is a much longer quote"]]
    result = quoted_spans_alignment(answers, sources)
    assert result["total"] == 1.0
    assert result["matched"] == 1.0
    result = quoted_spans_alignment(answers, sources, min_len=1)
    assert result["total"] == 2.0
    assert result["matched"] == 1.0


def test_aggregates_across_answers_and_joins_sources():
    answers = [
        'First "alpha beta gamma" then "delta epsilon zeta".',
        'Second "not in sources at all".',
    ]
    sources = [["alpha beta gamma", "delta epsilon zeta"], ["unrelated text"]]
    result = quoted_spans_alignment(answers, sources)
    assert result["matched"] == 2.0
    assert result["total"] == 3.0
    assert result["citation_alignment_quoted_spans"] == pytest.approx(2 / 3)
