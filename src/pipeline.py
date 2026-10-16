"""
Igneous data pipeline:

FIRMS CSV → validation → spatial clustering → cluster aggregation →
priority scoring → evidence scoring → robustness analysis → analyst dashboard

Orchestrates ingestion, data quality reporting, spatial DBSCAN clustering,
cluster-level aggregation, multi-factor priority scoring, and evidence tagging.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from src.data_validation import (
    find_missing_values,
    flag_invalid_latitude,
    flag_invalid_longitude,
    flag_negative_frp,
    validate_required_columns,
)
from src.evidence import add_evidence_columns
from src.scoring import add_priority_scores
from src.screening import screen_static_sources


@dataclass
class PipelineResult:
    """Everything the dashboard needs after one cached pipeline run."""

    data: pd.DataFrame
    quality: dict
    clustering_available: bool
    source_path: str
    warnings: list
    cluster_data: Optional[pd.DataFrame] = None
    static_screening: Optional[dict] = None


def try_get_clustering_fn(candidates=None):
    """
    Return a clustering function if a teammate module is present.

    Expected optional APIs (first match wins):
      - src.clustering.add_cluster_features(df) -> DataFrame
      - src.clustering.cluster_observations(df) -> DataFrame
      - src.aggregation.add_cluster_features(df) -> DataFrame
    """
    if candidates is None:
        candidates = [
            ("src.clustering", "add_cluster_features"),
            ("src.clustering", "cluster_observations"),
            ("src.aggregation", "add_cluster_features"),
        ]
    for module_name, fn_name in candidates:
        try:
            module = __import__(module_name, fromlist=[fn_name])
            fn = getattr(module, fn_name, None)
            if callable(fn):
                return fn
        except (ImportError, AttributeError):
            continue
    return None


def build_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize validation flags and dataset completeness for KPI cards.

    Preserves legacy keys for compatibility while exposing expanded
    data-quality metrics (time span, valid counts, satellites, clusters/noise).
    """
    missing = find_missing_values(df)
    inv_lat = flag_invalid_latitude(df)
    inv_lon = flag_invalid_longitude(df)
    neg_frp = flag_negative_frp(df)

    has_issue = inv_lat | inv_lon | neg_frp
    for col in missing.index:
        if missing[col] > 0 and col in df.columns:
            has_issue = has_issue | df[col].isna()

    total_obs = len(df)
    invalid_count = int(has_issue.sum())
    valid_count = max(0, total_obs - invalid_count)

    # Time span
    acq_dates = pd.to_datetime(df.get("acq_date"), errors="coerce").dropna()
    if not acq_dates.empty:
        min_date_str = acq_dates.min().strftime("%Y-%m-%d")
        max_date_str = acq_dates.max().strftime("%Y-%m-%d")
        days_span = (acq_dates.max().date() - acq_dates.min().date()).days + 1
        time_span_str = f"{min_date_str} to {max_date_str} ({days_span} day{'s' if days_span > 1 else ''})"
    else:
        time_span_str = "Unavailable"

    # Satellites represented
    sat_series = df.get("satellite", pd.Series(dtype=str)).dropna().astype(str)
    sat_counts = sat_series.value_counts().to_dict()
    sat_summary = ", ".join(f"{k}: {v:,}" for k, v in sat_counts.items()) if sat_counts else "Unknown"

    report = {
        "missing_required": int(missing.sum()),
        "missing_by_column": missing.to_dict(),
        "invalid_latitude": int(inv_lat.sum()),
        "invalid_longitude": int(inv_lon.sum()),
        "negative_frp": int(neg_frp.sum()),
        # Expanded metrics for Phase 9
        "total_observations": total_obs,
        "valid_observations": valid_count,
        "invalid_observations": invalid_count,
        "time_span": time_span_str,
        "satellites": sat_summary,
        "satellite_count": len(sat_counts),
        "cluster_count": 0,
        "noise_count": 0,
    }

    if "cluster_id" in df.columns:
        report["cluster_count"] = int(df.loc[df["cluster_id"] >= 0, "cluster_id"].nunique())
        report["noise_count"] = int((df["cluster_id"] == -1).sum())

    return report


def run_pipeline(csv_path: Any, enable_clustering: bool = True) -> PipelineResult:
    """
    Run the full Igneous pipeline once and return a PipelineResult.

    Raises FileNotFoundError / ValueError with clear messages for the UI.
    """
    path = Path(csv_path)
    warnings: List[str] = []

    if not path.exists():
        raise FileNotFoundError(f"FIRMS CSV not found: {path}")

    df = pd.read_csv(path)

    missing_columns = validate_required_columns(df)
    if missing_columns:
        raise ValueError(
            "Required columns are missing from the CSV: "
            + ", ".join(missing_columns)
        )

    # Spatial clustering (Phase 1)
    clustering_fn = try_get_clustering_fn() if enable_clustering else None
    clustering_available = False
    cluster_df = None

    if clustering_fn is not None:
        try:
            clustered = clustering_fn(df)
            if isinstance(clustered, pd.DataFrame):
                df = clustered
                clustering_available = True

                # Phase 2: Cluster aggregation
                try:
                    from src.clustering import aggregate_clusters

                    cluster_df = aggregate_clusters(df)
                    if not cluster_df.empty:
                        cluster_df = add_priority_scores(cluster_df)
                        cluster_df = add_evidence_columns(cluster_df, clustering_available=True)
                        cluster_df["item_id"] = cluster_df["cluster_id"]
                except Exception as agg_exc:
                    warnings.append(f"Cluster aggregation skipped: {agg_exc}")
            else:
                warnings.append("Clustering function returned a non-DataFrame; skipped.")
        except Exception as exc:
            warnings.append(f"Clustering failed and was skipped: {exc}")
            clustering_available = False
    else:
        warnings.append(
            "Clustering module not found. Persistence and observation_count "
            "default to 1 (provisional)."
        )

    # Data quality report
    quality = build_quality_report(df)

    # Priority scoring for observation-level records (Phase 3)
    scored = add_priority_scores(df)

    # Evidence scoring for observation-level records (Phase 4)
    scored = add_evidence_columns(scored, clustering_available=clustering_available)

    # Static / persistent thermal source screening (Phase 8)
    static_screening = screen_static_sources(df)

    # Assign stable row identifier
    scored = scored.reset_index(drop=True)
    scored.insert(0, "obs_id", scored.index.astype(int))

    return PipelineResult(
        data=scored,
        quality=quality,
        clustering_available=clustering_available,
        source_path=str(path.resolve()),
        warnings=warnings,
        cluster_data=cluster_df,
        static_screening=static_screening,
    )
