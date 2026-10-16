"""
Simple pytest checks for src/scoring.py.

These tests use tiny fake tables, not the full FIRMS CSV.
"""

import sys
from pathlib import Path

import pandas as pd

# Make the project root importable: from src.scoring import ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring import (
    add_priority_scores,
    combine_acq_datetime,
    frp_score,
    persistence_score,
    recency_score,
    size_score,
)


def test_combine_acq_datetime_reads_hhmm():
    """3 means 00:03. 1430 means 14:30."""
    df = pd.DataFrame(
        {
            "acq_date": ["2023-07-12", "2023-07-12"],
            "acq_time": [3, 1430],
        }
    )

    stamps = combine_acq_datetime(df)

    assert stamps.iloc[0] == pd.Timestamp("2023-07-12 00:03:00")
    assert stamps.iloc[1] == pd.Timestamp("2023-07-12 14:30:00")


def test_combine_acq_datetime_invalid_time_is_nat():
    """Hours must be 0-23 and minutes 0-59. Bad times become NaT."""
    df = pd.DataFrame(
        {
            "acq_date": ["2023-07-12", "2023-07-12"],
            "acq_time": [99, 2500],  # 00:99 and 25:00 are invalid
        }
    )

    stamps = combine_acq_datetime(df)

    assert pd.isna(stamps.iloc[0])
    assert pd.isna(stamps.iloc[1])


def test_recency_score_newest_is_one_and_12h_is_half():
    """Newest row scores 1. A row 12 hours older scores about 0.5."""
    df = pd.DataFrame(
        {
            "acq_date": ["2023-07-12", "2023-07-12"],
            "acq_time": [0, 1200],  # 00:00 and 12:00
        }
    )

    scores = recency_score(df)

    assert scores.iloc[1] == 1.0
    assert scores.iloc[0] == 0.5


def test_frp_score_higher_is_not_lower_and_missing_is_zero():
    """Larger valid FRP should not rank below smaller FRP. Missing -> 0."""
    frp = pd.Series([1.0, 10.0, None])
    scores = frp_score(frp)

    assert scores.iloc[1] >= scores.iloc[0]
    assert scores.iloc[2] == 0.0


def test_persistence_score_higher_is_not_lower_and_missing_is_zero():
    """More distinct dates should not score lower. Missing/invalid -> 0."""
    persistence = pd.Series([1, 5, None, -2])
    scores = persistence_score(persistence)

    assert scores.iloc[1] >= scores.iloc[0]
    assert scores.iloc[2] == 0.0
    assert scores.iloc[3] == 0.0


def test_size_score_higher_count_is_not_lower():
    """A larger observation_count should not score lower."""
    counts = pd.Series([1, 8])
    scores = size_score(counts)

    assert scores.iloc[1] >= scores.iloc[0]


def test_add_priority_scores_columns_range_and_sum():
    """Output columns exist, score is 0-100, contributions add up."""
    df = pd.DataFrame(
        {
            "acq_date": ["2023-07-12", "2023-07-12", "2023-07-12"],
            "acq_time": [3, 1200, 1430],
            "frp": [0.5, 4.0, None],
            "persistence": [1, 3, 2],
            "observation_count": [1, 4, 2],
        }
    )

    scored = add_priority_scores(df)

    expected_columns = [
        "recency_score",
        "frp_score",
        "persistence_score",
        "size_score",
        "recency_contribution",
        "frp_contribution",
        "persistence_contribution",
        "size_contribution",
        "priority_score",
    ]
    for column_name in expected_columns:
        assert column_name in scored.columns

    assert scored["priority_score"].between(0, 100).all()

    contribution_sum = (
        scored["recency_contribution"]
        + scored["frp_contribution"]
        + scored["persistence_contribution"]
        + scored["size_contribution"]
    )
    # Tiny float differences are OK.
    pd.testing.assert_series_equal(
        contribution_sum,
        scored["priority_score"],
        check_names=False,
        atol=1e-10,
    )
