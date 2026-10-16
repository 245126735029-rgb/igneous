"""
Spatial clustering and aggregation for NASA FIRMS VIIRS hotspot observations.

Uses scikit-learn DBSCAN with:
- Haversine metric on radian coordinates
- BallTree spatial indexing to avoid O(N^2) memory bottlenecks
- Earth radius ≈ 6,371 km
- Spatial neighborhood radius ≈ 1.5 km

Noise points receive cluster_id = -1.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_KM = 6371.0
DEFAULT_RADIUS_KM = 1.5
DEFAULT_MIN_SAMPLES = 2


def spatial_cluster(
    df: pd.DataFrame,
    eps_km: float = DEFAULT_RADIUS_KM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> pd.DataFrame:
    """
    Assign spatial cluster IDs using DBSCAN with haversine distance.

    Parameters
    ----------
    df : pd.DataFrame
        Input observations with 'latitude' and 'longitude' columns.
    eps_km : float
        Clustering distance threshold in kilometers (default 1.5 km).
    min_samples : int
        Minimum points to form a core cluster (default 2).

    Returns
    -------
    pd.DataFrame
        Copy of input with integer 'cluster_id' column added.
        Noise points (no neighbors within eps_km) receive -1.
    """
    out = df.copy()

    # Identify rows with valid numeric coordinates within Earth bounds
    lat = pd.to_numeric(out.get("latitude"), errors="coerce")
    lon = pd.to_numeric(out.get("longitude"), errors="coerce")
    valid_coords = (
        lat.notna()
        & lon.notna()
        & (lat >= -90.0)
        & (lat <= 90.0)
        & (lon >= -180.0)
        & (lon <= 180.0)
    )

    cluster_labels = np.full(len(out), -1, dtype=int)

    if valid_coords.sum() > 0:
        valid_indices = np.where(valid_coords.values)[0]
        valid_lat_rad = np.radians(lat.iloc[valid_indices].to_numpy(dtype=float))
        valid_lon_rad = np.radians(lon.iloc[valid_indices].to_numpy(dtype=float))
        coords_rad = np.column_stack([valid_lat_rad, valid_lon_rad])

        eps_rad = eps_km / EARTH_RADIUS_KM
        db = DBSCAN(
            eps=eps_rad,
            min_samples=min_samples,
            metric="haversine",
            algorithm="ball_tree",
        )
        labels = db.fit_predict(coords_rad)
        cluster_labels[valid_indices] = labels

    out["cluster_id"] = cluster_labels
    return out


def add_cluster_features(
    df: pd.DataFrame,
    eps_km: float = DEFAULT_RADIUS_KM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> pd.DataFrame:
    """
    Run spatial clustering and map cluster-derived features back to observations.

    Adds:
      - 'cluster_id': cluster identifier (-1 for isolated/noise observations)
      - 'observation_count': number of detections in this cluster (1 for noise)
      - 'persistence': repeat-observation metric across acquisition periods (1 for noise)

    Expected by src.pipeline.try_get_clustering_fn.
    """
    clustered = spatial_cluster(df, eps_km=eps_km, min_samples=min_samples)

    # Initialize defaults for all rows (including noise)
    clustered["observation_count"] = 1
    clustered["persistence"] = 1

    # Compute cluster-level metrics for valid clusters (cluster_id >= 0)
    in_cluster = clustered["cluster_id"] >= 0
    if in_cluster.any():
        cluster_groups = clustered[in_cluster].groupby("cluster_id")

        # Observation count per cluster
        counts = cluster_groups.size()

        # Distinct acquisition dates and distinct acquisition times
        if "acq_date" in clustered.columns and "acq_time" in clustered.columns:
            distinct_dates = cluster_groups["acq_date"].nunique()
            distinct_times = cluster_groups["acq_time"].nunique()
            # If multi-day dataset, persistence = distinct dates; if single-day, distinct times
            persistence = np.maximum(distinct_dates, distinct_times)
        elif "acq_date" in clustered.columns:
            persistence = cluster_groups["acq_date"].nunique()
        else:
            persistence = counts.clip(upper=1)

        # Map back to observation rows
        clustered.loc[in_cluster, "observation_count"] = (
            clustered.loc[in_cluster, "cluster_id"].map(counts).fillna(1).astype(int)
        )
        clustered.loc[in_cluster, "persistence"] = (
            clustered.loc[in_cluster, "cluster_id"].map(persistence).fillna(1).astype(int)
        )

    return clustered


# Alias for pipeline API compatibility
cluster_observations = add_cluster_features


def aggregate_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate clustered observations into cluster-level summary records.

    Each row represents a spatial cluster (cluster_id >= 0).
    Scientifically precise: distinguishes repeat observations / distinct
    acquisition periods from multi-satellite passes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'cluster_id' and FIRMS observation columns.

    Returns
    -------
    pd.DataFrame
        Cluster-level summary records sorted by observation_count descending.
    """
    if "cluster_id" not in df.columns:
        raise ValueError("DataFrame must contain 'cluster_id' column")

    clusters = df[df["cluster_id"] >= 0].copy()
    if clusters.empty:
        return pd.DataFrame()

    # Pre-sort by cluster_id, acq_date, acq_time so 'last' reflects newest observation
    sort_cols = ["cluster_id"]
    if "acq_date" in clusters.columns:
        sort_cols.append("acq_date")
    if "acq_time" in clusters.columns:
        sort_cols.append("acq_time")
    clusters = clusters.sort_values(sort_cols, ascending=True)

    # Ensure numeric FRP
    if "frp" in clusters.columns:
        clusters["_frp"] = pd.to_numeric(clusters["frp"], errors="coerce").fillna(0.0)
    else:
        clusters["_frp"] = 0.0

    g = clusters.groupby("cluster_id")

    lat_mean = g["latitude"].mean().round(5)
    lon_mean = g["longitude"].mean().round(5)
    obs_count = g.size()

    max_frp = g["_frp"].max().round(2)
    latest_frp = g["_frp"].last().round(2)
    mean_frp = g["_frp"].mean().round(2)

    has_date = "acq_date" in clusters.columns
    has_time = "acq_time" in clusters.columns

    earliest_date = g["acq_date"].min() if has_date else "Unknown"
    latest_date = g["acq_date"].max() if has_date else "Unknown"
    earliest_time = g["acq_time"].min() if has_time else 0
    latest_time = g["acq_time"].last() if has_time else 0

    distinct_dates = g["acq_date"].nunique() if has_date else 1
    distinct_times = g["acq_time"].nunique() if has_time else 1

    has_sat = "satellite" in clusters.columns
    sat_count = g["satellite"].nunique() if has_sat else 0
    sat_last = g["satellite"].last().astype(str) if has_sat else "Unknown"

    has_conf = "confidence" in clusters.columns
    conf_last = g["confidence"].last().astype(str) if has_conf else "nominal"

    persistence = np.maximum(distinct_dates, distinct_times)

    cluster_df = pd.DataFrame(
        {
            "cluster_id": g["cluster_id"].first(),
            "latitude": lat_mean,
            "longitude": lon_mean,
            "centroid_latitude": lat_mean,
            "centroid_longitude": lon_mean,
            "observation_count": obs_count,
            "acq_date": latest_date,
            "acq_time": latest_time,
            "earliest_date": earliest_date,
            "latest_date": latest_date,
            "earliest_time": earliest_time,
            "latest_time": latest_time,
            "frp": max_frp,
            "max_frp": max_frp,
            "latest_frp": latest_frp,
            "mean_frp": mean_frp,
            "distinct_dates": distinct_dates,
            "distinct_times": distinct_times,
            "persistence": persistence,
            "satellite_count": sat_count,
            "satellite": sat_last,
            "confidence": conf_last,
            "confidence_summary": conf_last,
        }
    ).reset_index(drop=True)

    return cluster_df.sort_values("observation_count", ascending=False).reset_index(drop=True)
