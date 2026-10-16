# Igneous

**Satellite-Derived Active Fire Triage**

> *A triage tool that shows its work, and shows how fragile its answers are.*

Igneous is a human decision-support dashboard for prioritizing NASA FIRMS / VIIRS satellite thermal observations for **review**.

It does **not** confirm fires, predict fire risk, or replace emergency-response systems.

---

## Conceptual Foundations

- **Priority ≠ Fire Probability:** The priority score (0–100) measures how urgently a human analyst should review an observation or cluster based on transparent operational criteria (recency, fire radiative power, repeat detections, cluster size). A high score does not mean high fire danger.
- **Evidence ≠ Fire Confirmation:** Evidence quantifies observational completeness, sensor confidence, and repeat detections across overpasses. Satellite thermal anomalies are not ground truth.
- **Robustness = Ranking Sensitivity:** Evaluates how much an item's ranking shifts when scoring weights vary (Monte Carlo weight perturbation with 200 trials). It is a test of mathematical scoring fragility, **not** real-world fire uncertainty.

---

## Architecture & Pipeline

```
NASA FIRMS CSV
  → Data Validation (`src/data_validation.py`)
  → Spatial DBSCAN Clustering (`src/clustering.py`)
  → Cluster Aggregation (`src/clustering.py`)
  → Priority Scoring (`src/scoring.py`)
  → Evidence Assessment (`src/evidence.py`)
  → Static-Source Screening (`src/screening.py`)
  → Robustness & Counterfactuals (`src/robustness.py`)
  → Analyst Dashboard & Review (`app/dashboard.py`)
```

Orchestrated end-to-end by `src/pipeline.py`.

---

## How to Run

From the repository root (`igneous`):

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

If setting up the virtual environment from scratch:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

---

## Running Tests

Execute the complete pytest suite (24 tests):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

---

## Key Features

1. **Spatial DBSCAN Clustering:** Haversine distance with BallTree spatial indexing (1.5 km radius, Earth $R \approx 6,371$ km) executed on 74k+ points in < 1 second.
2. **Cluster Aggregation:** Derives centroid coordinates, observation counts, repeat observations, and intensity metrics without fabricating multi-satellite passes.
3. **Analyst Persona Presets:** Switchable prioritization assumptions (`Balanced`, `Newest-first`, `Most-intense`, `Persistence-first`) with automatic weight normalization and `Consensus` vs `Contested` tagging.
4. **Priority vs Evidence Quadrants:** 2D visualization distinguishing review priority from supporting observational evidence.
5. **Ranking Robustness (Monte Carlo):** 200 weight-perturbation trials with fixed random seed (42) determining rank ranges and stability badges (`Stable`, `Moderately sensitive`, `Fragile`).
6. **Recalculated Counterfactuals:** Concrete "What could change this ranking?" explanations strictly recomputed from the scoring function.
7. **Static / Persistent Thermal Source Screening:** Safeguarded multi-day persistence detector that safely disengages with a clear explanation when temporal coverage is insufficient (< 7 days).
8. **Analyst Workflow:** Session-state tracking of review status (`Unreviewed`, `Under Review`, `Reviewed`, `Dismissed`) and custom notes.

---

## Data

Uses the sample NASA FIRMS VIIRS (Suomi NPP) CSV already present in this repository:
- `sample_viirs_snpp_071223.csv` (74,605 observations from 2023-07-12)
- No external network calls, no downloads, and no fabricated observations.

---

## Scientific Limitations

- **Satellite Revisit Gaps:** Polar-orbiting satellites provide periodic overpass snapshots. Revisit intervals create temporal blind spots.
- **Atmospheric & Canopy Obscuration:** Cloud cover, dense smoke, and forest canopy can attenuate thermal infrared radiation or mask ground heat.
- **Pixel Geolocation:** Coordinates represent nominal pixel centers (≈375 m at nadir), not exact ignition locations or perimeters.
- **Sensor View-Angle Distortion:** Pixel footprints expand away from nadir toward scan edges (bow-tie effect).
- **Persistent Non-Wildfire Heat Sources:** Industrial flaring, refineries, agricultural burns, and geothermal activity can trigger thermal anomalies.
- **Absence of Detection:** A lack of detection in a region is **never** proof that no fire exists.

---

## Project Layout

| Path | Role |
| --- | --- |
| `app/dashboard.py` | Streamlit analyst decision-support triage dashboard |
| `src/data_validation.py` | Schema validation and coordinate/FRP bounds checks |
| `src/clustering.py` | Haversine DBSCAN spatial clustering and cluster aggregation |
| `src/scoring.py` | Multi-factor priority scoring, personas, and weight normalization |
| `src/evidence.py` | Observational completeness and repeat detection evidence |
| `src/screening.py` | Recurrent static thermal source screening with temporal guard |
| `src/robustness.py` | Monte Carlo rank robustness and recalculated counterfactuals |
| `src/pipeline.py` | End-to-end ingestion and analysis orchestrator |
| `tests/` | Comprehensive 24-test pytest suite |
| `notebooks/` | Exploration notebook (untouched) |
