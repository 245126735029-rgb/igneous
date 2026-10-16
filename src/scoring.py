"""
Priority scoring for Igneous triage.

The score ranks satellite observations for human attention only.
It is NOT a fire-risk score. A higher number does not mean a more
dangerous fire, and it does not confirm that a fire is present.
"""

import numpy as np
import pandas as pd

# Change these later without rewriting the formulas.
WEIGHTS = {
    "recency": 0.30,
    "frp": 0.30,
    "persistence": 0.25,
    "size": 0.15,
}

# Placeholder columns until clustering is connected.
PERSISTENCE_COLUMN = "persistence"
SIZE_COLUMN = "observation_count"


def _safe_numeric(series):
    """Turn a column into numbers. Invalid text becomes NaN."""
    return pd.to_numeric(series, errors="coerce")


def _percentile_rank(series):
    """
    Convert values to a 0-1 rank inside this dataset.

    Example: the largest value gets a rank near 1.
    Missing values are not ranked and later score 0.
    """
    return series.rank(method="average", pct=True)


def combine_acq_datetime(df):
    """
    Build one timestamp from FIRMS acq_date and acq_time.

    acq_time in FIRMS is HHMM as an integer, so 3 means 00:03
    and 1430 means 14:30. Invalid dates/times become NaT.
    """
    dates = pd.to_datetime(df["acq_date"], errors="coerce")
    times = _safe_numeric(df["acq_time"])

    hours = times // 100
    minutes = times % 100

    # Hours must be 0-23 and minutes 0-59.
    invalid_clock = (hours < 0) | (hours > 23) | (minutes < 0) | (minutes > 59)
    hours = hours.mask(invalid_clock)
    minutes = minutes.mask(invalid_clock)

    return dates + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")


def recency_score(df):
    """
    Recency vs the newest observation in THIS dataset (not vs today).

    Formula: 0.5 ** (hours_old / 12)
    Newest row -> 1. Missing timestamps -> 0.
    """
    acquired_at = combine_acq_datetime(df)
    newest = acquired_at.max()

    hours_old = (newest - acquired_at).dt.total_seconds() / 3600.0
    # Negative gaps should not happen; clip them just in case.
    hours_old = hours_old.clip(lower=0)

    score = 0.5 ** (hours_old / 12.0)
    return score.fillna(0.0)


def frp_score(frp):
    """
    FRP intensity: log1p reduces extreme FRP, then percentile rank.

    Negative or missing FRP is invalid and scores 0.
    """
    frp_values = _safe_numeric(frp)
    frp_values = frp_values.where(frp_values >= 0)

    # log1p(x) = log(1 + x). Keeps 0 FRP at 0.
    logged = np.log1p(frp_values)
    return _percentile_rank(logged).fillna(0.0)


def persistence_score(persistence):
    """
    Persistence = distinct acquisition dates in a cluster (later).

    For now this is an input column. It is percentile-ranked to 0-1.
    Missing or negative values score 0.
    """
    values = _safe_numeric(persistence)
    values = values.where(values >= 0)
    return _percentile_rank(values).fillna(0.0)


def size_score(observation_count):
    """
    Observation size = how many detections belong together (later).

    For now this is the observation_count column, percentile-ranked.
    Missing or negative values score 0.
    """
    values = _safe_numeric(observation_count)
    values = values.where(values >= 0)
    return _percentile_rank(values).fillna(0.0)


def add_priority_scores(df, weights=None):
    """
    Add factor scores, contributions, and a 0-100 priority_score.

    If persistence or observation_count is missing, each row is treated
    as a single observation (value = 1) until clustering is added.
    """
    if weights is None:
        weights = WEIGHTS

    scored = df.copy()

    # Temporary defaults until DBSCAN fills these in.
    if PERSISTENCE_COLUMN not in scored.columns:
        scored[PERSISTENCE_COLUMN] = 1
    if SIZE_COLUMN not in scored.columns:
        scored[SIZE_COLUMN] = 1

    scored["recency_score"] = recency_score(scored)
    scored["frp_score"] = frp_score(scored["frp"])
    scored["persistence_score"] = persistence_score(scored[PERSISTENCE_COLUMN])
    scored["size_score"] = size_score(scored[SIZE_COLUMN])

    # Contributions are the weighted points that add up to priority_score.
    scored["recency_contribution"] = 100 * weights["recency"] * scored["recency_score"]
    scored["frp_contribution"] = 100 * weights["frp"] * scored["frp_score"]
    scored["persistence_contribution"] = (
        100 * weights["persistence"] * scored["persistence_score"]
    )
    scored["size_contribution"] = 100 * weights["size"] * scored["size_score"]

    scored["priority_score"] = (
        scored["recency_contribution"]
        + scored["frp_contribution"]
        + scored["persistence_contribution"]
        + scored["size_contribution"]
    )

    return scored
