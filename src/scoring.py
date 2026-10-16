"""
Priority scoring for Igneous triage.

The score ranks satellite observations for human attention only.
It is NOT a fire-risk score. A higher number does not mean a more
dangerous fire, and it does not confirm that a fire is present.
"""

import numpy as np
import pandas as pd

# Default weights (Balanced preset)
WEIGHTS = {
    "recency": 0.30,
    "frp": 0.30,
    "persistence": 0.25,
    "size": 0.15,
}

# Analyst persona presets — transparent assumptions, not different models.
PERSONAS = {
    "Balanced": {
        "recency": 0.30,
        "frp": 0.30,
        "persistence": 0.25,
        "size": 0.15,
    },
    "Newest-first": {
        "recency": 0.50,
        "frp": 0.25,
        "persistence": 0.15,
        "size": 0.10,
    },
    "Most-intense": {
        "recency": 0.20,
        "frp": 0.50,
        "persistence": 0.15,
        "size": 0.15,
    },
    "Persistence-first": {
        "recency": 0.15,
        "frp": 0.20,
        "persistence": 0.45,
        "size": 0.20,
    },
}


def normalize_weights(weights):
    """
    Ensure weights sum to 1.0.

    Raises ValueError if weights are empty, all-zero, or have negative values.
    """
    if not weights:
        raise ValueError("Weights dictionary cannot be empty.")

    total = sum(float(w) for w in weights.values())
    if total <= 0:
        raise ValueError("Sum of weights must be positive.")

    for k, v in weights.items():
        if float(v) < 0:
            raise ValueError(f"Weight for '{k}' cannot be negative.")

    return {k: float(v) / total for k, v in weights.items()}


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

    Weights are automatically normalized to sum to 1.0.
    If persistence or observation_count is missing, each row is treated
    as a single observation (value = 1) until clustering is added.
    """
    if weights is None:
        weights = WEIGHTS
    else:
        weights = normalize_weights(weights)

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


def compute_persona_consensus(df, consensus_cutoff=25):
    """
    Evaluate whether items in the queue remain prioritized across all analyst
    personas (Consensus) or depend on the chosen persona assumptions (Contested).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with observations or clusters.
    consensus_cutoff : int
        Max rank across all personas to qualify as consensus.

    Returns
    -------
    pd.Series
        'Consensus' or 'Contested' for each row.
    """
    if df.empty:
        return pd.Series(dtype=str)

    ranks = {}
    for persona_name, persona_weights in PERSONAS.items():
        scored_temp = add_priority_scores(df, weights=persona_weights)
        ranks[persona_name] = scored_temp["priority_score"].rank(
            ascending=False, method="min"
        )

    rank_df = pd.DataFrame(ranks, index=df.index)
    max_rank = rank_df.max(axis=1)
    rank_span = rank_df.max(axis=1) - rank_df.min(axis=1)

    is_consensus = (max_rank <= consensus_cutoff) | (rank_span <= 10)
    return is_consensus.map({True: "Consensus", False: "Contested"})

