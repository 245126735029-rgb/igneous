"""
Screening helper for recurrent / static thermal sources.

A static or persistent thermal source exhibits repeat observations across
multiple calendar days at the same geographical location.

Scientific Safeguards:
- Requires sufficient temporal span (minimum 7+ days) to avoid false conclusions.
- If insufficient temporal span: reports "Static-source screening unavailable: insufficient temporal coverage."
- Does NOT claim:
    * industrial facility detected
    * gas flare detected
    * volcano detected
    * false positive / non-fire
- Does NOT output wildfire probability.
- Clearly states that persistent thermal activity has multiple possible causes
  and requires analyst / ground verification.
"""

from typing import Dict, Any, Set
import pandas as pd


def screen_static_sources(df: pd.DataFrame, min_days: int = 7) -> Dict[str, Any]:
    """
    Screen for recurrent thermal detections across distinct calendar dates.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing observations or clusters with 'acq_date'.
    min_days : int
        Minimum calendar day span required to activate screening (default 7).

    Returns
    -------
    dict
        Screening result containing availability status, explanation, and
        flagged cluster IDs if temporal threshold is met.
    """
    if df.empty or "acq_date" not in df.columns:
        return {
            "available": False,
            "status": "Static-source screening unavailable: insufficient temporal coverage.",
            "days_span": 0,
            "required_days": min_days,
            "explanation": (
                "Insufficient temporal data to screen for static sources. "
                "At least 7+ days of multi-day satellite observations are required."
            ),
            "screened_clusters": set(),
        }

    parsed_dates = pd.to_datetime(df["acq_date"], errors="coerce").dropna()
    if parsed_dates.empty:
        return {
            "available": False,
            "status": "Static-source screening unavailable: insufficient temporal coverage.",
            "days_span": 0,
            "required_days": min_days,
            "explanation": "No valid acquisition dates found in dataset.",
            "screened_clusters": set(),
        }

    min_date = parsed_dates.min().date()
    max_date = parsed_dates.max().date()
    days_span = (max_date - min_date).days + 1
    unique_dates_count = parsed_dates.dt.date.nunique()

    # Guard: requires at least min_days of temporal span
    if days_span < min_days or unique_dates_count < min_days:
        return {
            "available": False,
            "status": "Static-source screening unavailable: insufficient temporal coverage.",
            "days_span": int(days_span),
            "unique_dates_count": int(unique_dates_count),
            "required_days": min_days,
            "explanation": (
                f"Dataset covers only {unique_dates_count} distinct date(s) across {days_span} day(s). "
                f"Reliable static-source screening requires multi-day satellite observation (target: {min_days}+ days) "
                "to distinguish recurrent stationary thermal activity from episodic fire progression. "
                "Persistent thermal activity can have multiple causes (e.g., controlled agricultural burning, "
                "active wildfires, industrial processes, or geographic thermal anomalies) and requires external verification."
            ),
            "screened_clusters": set(),
        }

    # Temporal threshold met: identify clusters observed across multiple dates
    flagged: Dict[int, Dict[str, Any]] = {}
    if "cluster_id" in df.columns:
        clustered = df[df["cluster_id"] >= 0]
        date_counts = clustered.groupby("cluster_id")["acq_date"].nunique()
        # Detections on >= 50% of observed dates or >= 4 distinct dates
        recurrence_threshold = max(4, unique_dates_count // 2)
        persistent_cids = date_counts[date_counts >= recurrence_threshold].index

        for cid in persistent_cids:
            observed_dates = date_counts[cid]
            flagged[int(cid)] = {
                "label": "Possible persistent/static thermal source",
                "explanation": (
                    f"Detected on {observed_dates} of {unique_dates_count} observed dates near this location. "
                    "May represent persistent thermal activity; requires external verification."
                ),
                "observed_dates": int(observed_dates),
                "total_dates": int(unique_dates_count),
            }

    return {
        "available": True,
        "status": f"Screening active across {unique_dates_count} observation dates ({days_span} days).",
        "days_span": int(days_span),
        "unique_dates_count": int(unique_dates_count),
        "required_days": min_days,
        "explanation": (
            "Screening flags thermal anomalies observed recurrently across multiple distinct dates. "
            "Does not confirm non-fire or industrial origin. External verification required."
        ),
        "screened_clusters": flagged,
    }


def get_item_screening_info(item: pd.Series, screening: Dict[str, Any]) -> Dict[str, str]:
    """
    Get human-readable screening status for a single selected observation or cluster.
    """
    if not screening.get("available", False):
        return {
            "status": screening.get("status", "Static-source screening unavailable: insufficient temporal coverage."),
            "detail": screening.get("explanation", ""),
            "is_flagged": False,
        }

    cid = item.get("cluster_id", -1)
    screened = screening.get("screened_clusters", {})
    if cid in screened:
        entry = screened[cid]
        return {
            "status": entry["label"],
            "detail": entry["explanation"],
            "is_flagged": True,
        }

    return {
        "status": "Not flagged as recurrent static source",
        "detail": f"Observed on fewer than required threshold dates out of {screening.get('unique_dates_count', 0)}.",
        "is_flagged": False,
    }
