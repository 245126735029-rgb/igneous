"""Smoke tests for the Igneous pipeline (tiny frames, not the full CSV)."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence import add_evidence_columns
from src.pipeline import build_quality_report, run_pipeline, try_get_clustering_fn
from src.robustness import explain_row_contributions


def _tiny_firms_frame():
    return pd.DataFrame(
        {
            "latitude": [1.0, 2.0],
            "longitude": [10.0, 20.0],
            "acq_date": ["2023-07-12", "2023-07-12"],
            "acq_time": [3, 1200],
            "satellite": ["N", "N"],
            "confidence": ["n", "h"],
            "frp": [1.0, 8.0],
            "daynight": ["N", "D"],
        }
    )


def test_try_get_clustering_fn_handles_missing_module():
    # Gracefully returns None when candidates module is missing
    assert try_get_clustering_fn(candidates=[("src.nonexistent_clustering", "fn")]) is None
    # Returns the clustering function when present
    assert callable(try_get_clustering_fn())


def test_build_quality_report_counts():
    df = _tiny_firms_frame()
    report = build_quality_report(df)
    assert report["missing_required"] == 0
    assert report["invalid_latitude"] == 0
    assert report["negative_frp"] == 0


def test_add_evidence_columns_labels_unavailable_agreement():
    df = add_evidence_columns(_tiny_firms_frame(), clustering_available=False)
    assert "evidence_score" in df.columns
    assert df["sensor_pass_agreement"].iloc[0].startswith("Unavailable")
    assert df["evidence_score"].between(0, 100).all()


def test_run_pipeline_on_sample_head(tmp_path):
    sample = PROJECT_ROOT / "sample_viirs_snpp_071223.csv"
    tiny = pd.read_csv(sample, nrows=20)
    path = tmp_path / "tiny_firms.csv"
    tiny.to_csv(path, index=False)

    result = run_pipeline(path)
    assert len(result.data) == 20
    assert "priority_score" in result.data.columns
    assert "evidence_score" in result.data.columns
    assert "obs_id" in result.data.columns
    assert result.clustering_available is True
    assert result.data["priority_score"].between(0, 100).all()


def test_explain_row_contributions_has_drivers():
    from src.scoring import add_priority_scores

    scored = add_priority_scores(_tiny_firms_frame())
    scored = add_evidence_columns(scored, clustering_available=False)
    explanation = explain_row_contributions(scored.iloc[1])
    assert "priority_score" in explanation
    assert len(explanation["top_drivers"]) == 4
