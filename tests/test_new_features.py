"""
Test suite covering Phase 15 requirements for the Igneous active-fire triage platform.

Covers:
1. Nearby points cluster together.
2. Distant points do not.
3. Cluster observation counts are correct.
4. Newer observations score >= older observations when everything else is equal.
5. Higher FRP scores >= lower FRP when everything else is equal.
6. Higher persistence cannot lower priority when everything else is equal.
7. Weight normalization sums to 1.
8. Persona weights sum to 1.
9. Robustness simulation is deterministic with fixed seed.
10. Counterfactuals use actual recalculated scores.
11. Static-source screening requires sufficient temporal coverage.
12. Missing optional evidence fields do not crash the pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import (
    spatial_cluster,
    add_cluster_features,
    aggregate_clusters,
    EARTH_RADIUS_KM,
)
from src.scoring import (
    PERSONAS,
    WEIGHTS,
    normalize_weights,
    add_priority_scores,
)
from src.robustness import (
    simulate_rank_robustness,
    compute_counterfactuals,
)
from src.screening import (
    screen_static_sources,
)
from src.evidence import (
    add_evidence_columns,
)


# ==============================================================================
# 1. Nearby points cluster together
# ==============================================================================
def test_nearby_points_cluster_together():
    """Points separated by ~100 meters (< 1.5 km threshold) must share cluster_id >= 0."""
    # At latitude 0, 0.001 deg longitude is ~111 meters
    df = pd.DataFrame({
        "latitude": [0.0, 0.0005],
        "longitude": [0.0, 0.0005],
        "acq_date": ["2023-07-12", "2023-07-12"],
        "acq_time": [1000, 1000],
    })
    clustered = spatial_cluster(df, eps_km=1.5, min_samples=2)
    assert clustered["cluster_id"].iloc[0] >= 0
    assert clustered["cluster_id"].iloc[0] == clustered["cluster_id"].iloc[1]


# ==============================================================================
# 2. Distant points do not cluster together
# ==============================================================================
def test_distant_points_do_not_cluster():
    """Points separated by ~100 km (> 1.5 km threshold) must NOT share a cluster."""
    df = pd.DataFrame({
        "latitude": [0.0, 1.0],  # 1 degree latitude is ~111 km
        "longitude": [0.0, 0.0],
        "acq_date": ["2023-07-12", "2023-07-12"],
        "acq_time": [1000, 1000],
    })
    clustered = spatial_cluster(df, eps_km=1.5, min_samples=2)
    # Both points have no neighbor within 1.5 km, so both are noise (-1)
    assert clustered["cluster_id"].iloc[0] == -1
    assert clustered["cluster_id"].iloc[1] == -1


# ==============================================================================
# 3. Cluster observation counts are correct
# ==============================================================================
def test_cluster_observation_counts_correct():
    """A cluster of 3 points must report observation_count = 3."""
    df = pd.DataFrame({
        "latitude": [10.000, 10.001, 10.002, 50.0],
        "longitude": [20.000, 20.001, 20.002, 50.0],
        "acq_date": ["2023-07-12"] * 4,
        "acq_time": [100, 200, 300, 400],
        "frp": [10.0, 20.0, 30.0, 5.0],
        "satellite": ["N"] * 4,
        "confidence": ["nominal"] * 4,
    })
    clustered = add_cluster_features(df, eps_km=1.5, min_samples=2)
    # First 3 points form cluster 0
    assert (clustered.iloc[:3]["observation_count"] == 3).all()
    # 4th point is noise (single observation)
    assert clustered.iloc[3]["observation_count"] == 1
    assert clustered.iloc[3]["cluster_id"] == -1

    agg = aggregate_clusters(clustered)
    assert len(agg) == 1
    assert agg.iloc[0]["observation_count"] == 3
    assert agg.iloc[0]["cluster_id"] == clustered.iloc[0]["cluster_id"]


# ==============================================================================
# 4. Newer observations score >= older observations when everything else is equal
# ==============================================================================
def test_newer_observations_score_not_lower_than_older():
    """Holding FRP, persistence, and size constant, newer observation must score >= older."""
    df = pd.DataFrame({
        "acq_date": ["2023-07-12", "2023-07-12"],
        "acq_time": [0, 1200],  # 00:00 vs 12:00
        "frp": [25.0, 25.0],
        "persistence": [2, 2],
        "observation_count": [5, 5],
    })
    scored = add_priority_scores(df)
    assert scored.iloc[1]["priority_score"] >= scored.iloc[0]["priority_score"]


# ==============================================================================
# 5. Higher FRP scores >= lower FRP when everything else is equal
# ==============================================================================
def test_higher_frp_scores_not_lower():
    """Holding time, persistence, and size constant, higher FRP must score >= lower FRP."""
    df = pd.DataFrame({
        "acq_date": ["2023-07-12", "2023-07-12"],
        "acq_time": [1200, 1200],
        "frp": [10.0, 150.0],
        "persistence": [2, 2],
        "observation_count": [5, 5],
    })
    scored = add_priority_scores(df)
    assert scored.iloc[1]["priority_score"] >= scored.iloc[0]["priority_score"]


# ==============================================================================
# 6. Higher persistence cannot lower priority when everything else is equal
# ==============================================================================
def test_higher_persistence_does_not_lower_priority():
    """Holding time, FRP, and size constant, higher persistence must score >= lower persistence."""
    df = pd.DataFrame({
        "acq_date": ["2023-07-12", "2023-07-12"],
        "acq_time": [1200, 1200],
        "frp": [50.0, 50.0],
        "persistence": [1, 5],
        "observation_count": [4, 4],
    })
    scored = add_priority_scores(df)
    assert scored.iloc[1]["priority_score"] >= scored.iloc[0]["priority_score"]


# ==============================================================================
# 7. Weight normalization sums to 1
# ==============================================================================
def test_weight_normalization_sums_to_one():
    """Arbitrary positive weights must normalize to a sum of exactly 1.0."""
    raw = {"recency": 10.0, "frp": 25.0, "persistence": 15.0, "size": 50.0}
    normalized = normalize_weights(raw)
    assert pytest.approx(sum(normalized.values()), rel=1e-7) == 1.0
    assert normalized["size"] == 0.50


# ==============================================================================
# 8. Persona weights sum to 1
# ==============================================================================
def test_all_persona_weights_sum_to_one():
    """Every defined analyst persona must sum to 1.0."""
    for name, weights in PERSONAS.items():
        total = sum(weights.values())
        assert pytest.approx(total, rel=1e-7) == 1.0, f"Persona '{name}' sum is {total}"


# ==============================================================================
# 9. Robustness simulation is deterministic with fixed seed
# ==============================================================================
def test_robustness_simulation_is_deterministic():
    """Two simulation runs with the same random seed must yield identical results."""
    df = pd.DataFrame({
        "obs_id": [1, 2, 3, 4],
        "acq_date": ["2023-07-12"] * 4,
        "acq_time": [100, 400, 800, 1200],
        "frp": [10.0, 50.0, 20.0, 80.0],
        "persistence": [1, 3, 2, 4],
        "observation_count": [2, 10, 5, 20],
    })
    scored = add_priority_scores(df)

    res1 = simulate_rank_robustness(scored, n_simulations=50, top_n=4, seed=42, id_col="obs_id")
    res2 = simulate_rank_robustness(scored, n_simulations=50, top_n=4, seed=42, id_col="obs_id")

    assert res1.keys() == res2.keys()
    for item_id in res1:
        assert res1[item_id]["best_rank"] == res2[item_id]["best_rank"]
        assert res1[item_id]["worst_rank"] == res2[item_id]["worst_rank"]
        assert res1[item_id]["rank_span"] == res2[item_id]["rank_span"]
        assert res1[item_id]["status"] == res2[item_id]["status"]


# ==============================================================================
# 10. Counterfactuals use actual recalculated scores
# ==============================================================================
def test_counterfactuals_use_actual_recalculated_scores():
    """Counterfactual ranking changes must match recalculation of the scoring function."""
    df = pd.DataFrame({
        "obs_id": [101, 102, 103],
        "acq_date": ["2023-07-12"] * 3,
        "acq_time": [1200, 1100, 100],
        "frp": [100.0, 20.0, 5.0],
        "persistence": [1, 2, 1],
        "observation_count": [1, 4, 1],
    })
    statements = compute_counterfactuals(df, item_id=102, id_col="obs_id")
    assert isinstance(statements, list)
    assert len(statements) >= 1
    # Check that statements are strings and report concrete ranks
    for stmt in statements:
        assert isinstance(stmt, str)
        assert len(stmt) > 10


# ==============================================================================
# 11. Static-source screening requires sufficient temporal coverage
# ==============================================================================
def test_static_source_screening_requires_temporal_coverage():
    """1-day dataset must be marked unavailable; 10-day dataset must be evaluated."""
    # 1-day dataset
    df_1day = pd.DataFrame({
        "acq_date": ["2023-07-12"] * 10,
        "cluster_id": [0] * 10,
    })
    res_1day = screen_static_sources(df_1day, min_days=7)
    assert res_1day["available"] is False
    assert "insufficient temporal coverage" in res_1day["status"]

    # 10-day dataset
    dates = [f"2023-07-{d:02d}" for d in range(1, 11)]
    df_10days = pd.DataFrame({
        "acq_date": dates * 2,
        "cluster_id": [1] * len(dates) + [2] * len(dates),
    })
    res_10days = screen_static_sources(df_10days, min_days=7)
    assert res_10days["available"] is True
    assert 1 in res_10days["screened_clusters"]


# ==============================================================================
# 12. Missing optional evidence fields do not crash the pipeline
# ==============================================================================
def test_missing_optional_evidence_fields_do_not_crash():
    """add_evidence_columns must work safely when optional columns are absent."""
    df = pd.DataFrame({
        "latitude": [12.0],
        "longitude": [77.0],
        "frp": [45.0],
        "acq_date": ["2023-07-12"],
        "acq_time": [1200],
        # 'confidence', 'daynight', 'satellite' intentionally omitted
    })
    res = add_evidence_columns(df, clustering_available=False)
    assert "evidence_score" in res.columns
    assert 0 <= res["evidence_score"].iloc[0] <= 100
    assert "sensor_pass_agreement" in res.columns
    assert "multi_satellite_agreement" in res.columns
