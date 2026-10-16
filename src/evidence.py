"""
Evidence helpers for Igneous satellite fire triage.

Evidence answers: "How much supporting information do we have?"
Priority answers: "How much attention should an analyst give this?"

These are distinct concepts:
- Priority is NOT fire probability or danger.
- Evidence is NOT fire confirmation.
- Satellite observations are not ground truth.
"""

import numpy as np
import pandas as pd


def add_evidence_columns(df: pd.DataFrame, clustering_available: bool = False) -> pd.DataFrame:
    """
    Add transparent evidence fields derived strictly from available data.

    Never invents multi-satellite agreement or sensor confirmation.
    If multiple satellites are not present, explicitly marks multi-satellite agreement
    as 'Unavailable in this dataset'.

    Parameters
    ----------
    df : pd.DataFrame
        Input observations or clusters.
    clustering_available : bool
        Whether spatial clustering features are present.

    Returns
    -------
    pd.DataFrame
        DataFrame with evidence scores and qualitative assessment fields.
    """
    out = df.copy()

    # Field completeness checks (supporting data completeness)
    has_coords = out.get("latitude", pd.Series(index=out.index)).notna() & out.get(
        "longitude", pd.Series(index=out.index)
    ).notna()
    has_frp = pd.to_numeric(out.get("frp", pd.Series(index=out.index)), errors="coerce").notna()
    has_confidence = out.get("confidence", pd.Series(index=out.index)).notna()
    has_time = out.get("acq_date", pd.Series(index=out.index)).notna() & out.get(
        "acq_time", pd.Series(index=out.index)
    ).notna()
    has_satellite = out.get("satellite", pd.Series(index=out.index)).notna()
    has_daynight = (
        out.get("daynight", pd.Series(index=out.index)).notna()
        if "daynight" in out.columns
        else pd.Series(False, index=out.index)
    )

    support_count = (
        has_coords.astype(int)
        + has_frp.astype(int)
        + has_confidence.astype(int)
        + has_time.astype(int)
        + has_satellite.astype(int)
        + has_daynight.astype(int)
    )
    completeness_pts = (40.0 * support_count / 6.0).round(1)
    out["supporting_field_count"] = support_count

    # Sensor confidence score (from FIRMS algorithm: 'h'=high, 'n'=nominal, 'l'=low)
    conf_series = out.get("confidence", pd.Series(dtype=str)).astype(str).str.lower()
    conf_pts = pd.Series(10.0, index=out.index)  # default for missing / unranked
    conf_pts = conf_pts.mask(conf_series.str.startswith("h"), 25.0)
    conf_pts = conf_pts.mask(conf_series.str.startswith("n"), 18.0)
    conf_pts = conf_pts.mask(conf_series.str.startswith("l"), 8.0)

    # Multi-satellite agreement check (strictly verified from data)
    sat_col = out.get("satellite", pd.Series(index=out.index)).dropna().astype(str)
    unique_sats = sat_col.unique()
    if len(unique_sats) > 1:
        out["multi_satellite_agreement"] = f"Supported ({len(unique_sats)} satellites observed)"
        multi_sat_pts = 15.0
    else:
        out["multi_satellite_agreement"] = "Unavailable in this dataset"
        multi_sat_pts = 0.0

    # Repeat observations & temporal persistence evidence
    if clustering_available and "observation_count" in out.columns:
        obs_count = pd.to_numeric(out["observation_count"], errors="coerce").fillna(1)
        persistence = pd.to_numeric(out.get("persistence", 1), errors="coerce").fillna(1)

        # Multi-observation evidence (up to 20 pts)
        obs_pts = np.where(obs_count >= 5, 20.0, np.where(obs_count > 1, 12.0, 5.0))
        # Temporal persistence evidence (up to 15 pts)
        persist_pts = np.where(persistence >= 3, 15.0, np.where(persistence >= 2, 10.0, 5.0))

        total_evidence = completeness_pts + conf_pts + obs_pts + persist_pts
        out["evidence_score"] = np.clip(total_evidence, 0.0, 100.0).round(1)

        out["sensor_pass_agreement"] = np.where(
            persistence >= 2,
            "Repeat observations recorded across distinct acquisition periods",
            "Single observation period recorded",
        )
        out["evidence_notes"] = (
            "Evidence combines field completeness, sensor confidence, and repeat observations. "
            "High evidence does not confirm a fire."
        )
    else:
        # Fallback when clustering is not present
        out["evidence_score"] = (100.0 * support_count / 6.0).round(1)
        out["sensor_pass_agreement"] = "Unavailable (no clustering yet)"
        out["evidence_notes"] = (
            "Observation-level field completeness only. "
            "Persistence / repeat observation evidence unavailable until clustering is integrated."
        )

    return out
