"""
Igneous Streamlit dashboard — satellite fire triage (human decision support).

Ranks NASA FIRMS / VIIRS thermal observations for human review attention.
Does NOT confirm fires, predict fire risk, or replace emergency dispatch systems.

Tagline: "A triage tool that shows its work, and shows how fragile its answers are."
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Repository root on sys.path so `from src...` works when launched via Streamlit.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import run_pipeline, PipelineResult
from src.robustness import (
    explain_row_contributions,
    simulate_rank_robustness,
    compute_counterfactuals,
    sensitivity_with_alt_weights,
)
from src.scoring import (
    WEIGHTS,
    PERSONAS,
    normalize_weights,
    add_priority_scores,
    compute_persona_consensus,
)
from src.screening import get_item_screening_info

SAMPLE_CSV = REPO_ROOT / "sample_viirs_snpp_071223.csv"
HIGH_PRIORITY_THRESHOLD = 70.0


def inject_styles():
    """Clean, professional decision-support styling."""
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
        h1 { letter-spacing: -0.02em; margin-bottom: 0.2rem !important; }
        .igneous-tagline {
            color: #4a5568; font-size: 1.05rem; margin-bottom: 0.75rem;
        }
        .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 0.75rem; }
        .kpi-grid-6 { display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.65rem; margin-bottom: 0.75rem; }
        .kpi-card {
            background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
            border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.75rem 0.9rem;
        }
        .kpi-label { font-size: 0.75rem; color: #4a5568; text-transform: uppercase; letter-spacing: 0.04em; }
        .kpi-value { font-size: 1.45rem; font-weight: 700; color: #1a202c; margin-top: 0.15rem; }
        .disclaimer {
            background: #fffaf0; border-left: 4px solid #dd6b20; padding: 0.85rem 1rem;
            border-radius: 0 8px 8px 0; margin: 0.75rem 0 1.25rem 0; color: #7b341e; font-size: 0.92rem;
        }
        .concept-box {
            border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.9rem 1rem;
            background: #ffffff; height: 100%;
        }
        .concept-title { font-weight: 700; color: #2d3748; margin-bottom: 0.35rem; }
        .muted { color: #718096; font-size: 0.88rem; }
        .badge-stable { background: #c6f6d5; color: #22543d; padding: 3px 8px; border-radius: 12px; font-weight: 600; font-size: 0.8rem; }
        .badge-moderate { background: #feebc8; color: #7b341e; padding: 3px 8px; border-radius: 12px; font-weight: 600; font-size: 0.8rem; }
        .badge-fragile { background: #fed7d7; color: #742a2a; padding: 3px 8px; border-radius: 12px; font-weight: 600; font-size: 0.8rem; }
        .badge-consensus { background: #ebf8ff; color: #2b6cb0; padding: 3px 8px; border-radius: 12px; font-weight: 600; font-size: 0.8rem; }
        .badge-contested { background: #edf2f7; color: #4a5568; padding: 3px 8px; border-radius: 12px; font-weight: 600; font-size: 0.8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Running Igneous pipeline (validate → cluster → aggregate → score → evidence)…")
def load_pipeline() -> PipelineResult:
    """Load and process the sample once. Cached deterministically."""
    return run_pipeline(SAMPLE_CSV)


@st.cache_data(show_spinner="Running Monte Carlo robustness simulation (200 trials)…")
def get_cached_robustness(
    df_subset: pd.DataFrame,
    weights_tuple: Tuple[Tuple[str, float], ...],
    top_n: int = 50,
    seed: int = 42,
    id_col: str = "item_id",
) -> Dict[Any, Dict[str, Any]]:
    """Compute 200 weight perturbation trials for the top items."""
    weights_dict = dict(weights_tuple)
    return simulate_rank_robustness(
        df_subset,
        n_simulations=200,
        top_n=top_n,
        seed=seed,
        baseline_weights=weights_dict,
        id_col=id_col,
    )


def ensure_review_state():
    if "review_status" not in st.session_state:
        st.session_state.review_status = {}
    if "review_notes" not in st.session_state:
        st.session_state.review_notes = {}


def score_to_rgba(score: float):
    """Map priority 0–100 to amber / fire triage spectrum."""
    t = max(0.0, min(1.0, float(score) / 100.0))
    r = int(90 + t * (230 - 90))
    g = int(140 + t * (85 - 140))
    b = int(190 + t * (30 - 190))
    a = int(140 + t * (210 - 140))
    return [r, g, b, a]


def render_map(display_df: pd.DataFrame, map_point_limit: int, id_col: str = "item_id"):
    """Interactive PyDeck map scaled by priority score with robust fallback."""
    map_df = (
        display_df.head(map_point_limit)
        .dropna(subset=["latitude", "longitude"])
        .copy()
    )
    if map_df.empty:
        st.warning("No valid coordinates to plot for the current filters.")
        return

    map_df["color"] = map_df["priority_score"].apply(score_to_rgba)
    map_df["radius"] = 120 + (map_df["priority_score"] / 100.0) * 350

    try:
        import pydeck as pdk

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[longitude, latitude]",
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
        )
        view = pdk.ViewState(
            latitude=float(map_df["latitude"].mean()),
            longitude=float(map_df["longitude"].mean()),
            zoom=2.2,
            pitch=0,
        )
        tooltip = {
            "html": (
                "<b>ID:</b> {" + id_col + "}<br/>"
                "<b>Priority Score:</b> {priority_score}<br/>"
                "<b>Evidence Score:</b> {evidence_score}<br/>"
                "<b>Observations:</b> {observation_count}<br/>"
                "<b>Lat/Lon:</b> {latitude}, {longitude}"
            ),
            "style": {"backgroundColor": "#1a202c", "color": "white", "fontSize": "12px"},
        }
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view,
                tooltip=tooltip,
                map_style=None,
            ),
            use_container_width=True,
        )
    except Exception:
        st.map(map_df, latitude="latitude", longitude="longitude", size=25)

    if len(display_df) > map_point_limit:
        st.caption(
            f"Map displaying top {len(map_df):,} of {len(display_df):,} records "
            f"by review priority (performance budget limit)."
        )
    else:
        st.caption(f"Map displaying {len(map_df):,} records.")


def render_priority_vs_evidence(df: pd.DataFrame, id_col: str = "item_id"):
    """Render 2D Quadrant Priority vs Evidence visualization."""
    plot_df = df.head(150).copy()
    if plot_df.empty:
        return

    try:
        import plotly.express as px

        fig = px.scatter(
            plot_df,
            x="evidence_score",
            y="priority_score",
            color="priority_score",
            size=plot_df["observation_count"].clip(lower=1, upper=50),
            hover_data=[id_col, "observation_count", "persistence", "frp"],
            color_continuous_scale="Viridis",
            labels={
                "evidence_score": "Evidence Score (Supporting Information Completeness)",
                "priority_score": "Review Priority Score (Attention Allocation)",
            },
            range_x=[0, 105],
            range_y=[0, 105],
        )

        # Quadrant dividing lines
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)

        # Quadrant annotations
        fig.add_annotation(
            x=80, y=95, text="High Priority / High Evidence", showarrow=False,
            font=dict(size=10, color="#2b6cb0")
        )
        fig.add_annotation(
            x=20, y=95, text="High Priority / Lower Evidence", showarrow=False,
            font=dict(size=10, color="#dd6b20")
        )
        fig.add_annotation(
            x=80, y=10, text="Lower Priority / High Evidence", showarrow=False,
            font=dict(size=10, color="#4a5568")
        )
        fig.add_annotation(
            x=20, y=10, text="Lower Priority / Lower Evidence", showarrow=False,
            font=dict(size=10, color="#718096")
        )

        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=380,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.scatter_chart(plot_df, x="evidence_score", y="priority_score", color="priority_score")


def main():
    st.set_page_config(
        page_title="Igneous — Satellite Active Fire Triage",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    ensure_review_state()

    # ==================================================
    # 1. HEADER + TAGLINE + DISCLAIMERS
    # ==================================================
    st.markdown("# Igneous")
    st.markdown("### Satellite-Derived Active Fire Triage Dashboard")
    st.markdown(
        '<p class="igneous-tagline">'
        "<em>A triage tool that shows its work, and shows how fragile its answers are.</em>"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer">'
        "<strong>Human Decision-Support Disclaimers:</strong><br/>"
        "• <strong>Priority is NOT fire probability or risk:</strong> A higher priority score reflects analyst review urgency, not dangerousness.<br/>"
        "• <strong>Evidence does NOT confirm a fire:</strong> Evidence quantifies observational completeness and repeat detection, not ground truth.<br/>"
        "• <strong>Robustness is NOT fire uncertainty:</strong> Sensitivity simulation measures stability to scoring weight assumptions.<br/>"
        "• Satellite thermal detections are subject to revisit gaps, cloud/smoke obscuration, and non-fire thermal sources."
        "</div>",
        unsafe_allow_html=True,
    )

    # Load Pipeline
    if not SAMPLE_CSV.exists():
        st.error(f"Sample FIRMS CSV not found at `{SAMPLE_CSV}`.")
        return

    try:
        pipeline_res = load_pipeline()
    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        return

    raw_scored = pipeline_res.data
    quality = pipeline_res.quality
    cluster_df = pipeline_res.cluster_data
    static_screening = pipeline_res.static_screening

    for w in pipeline_res.warnings:
        st.sidebar.info(w)

    # ==================================================
    # 2. DATASET / DATA-QUALITY OVERVIEW (PHASE 9)
    # ==================================================
    st.subheader("Dataset & Data-Quality Overview")
    st.markdown(
        f"""
        <div class="kpi-grid-6">
          <div class="kpi-card"><div class="kpi-label">Total Observations</div><div class="kpi-value">{quality['total_observations']:,}</div></div>
          <div class="kpi-card"><div class="kpi-label">Valid Observations</div><div class="kpi-value">{quality['valid_observations']:,}</div></div>
          <div class="kpi-card"><div class="kpi-label">Quality Issues</div><div class="kpi-value">{quality['invalid_observations']:,}</div></div>
          <div class="kpi-card"><div class="kpi-label">Spatial Clusters</div><div class="kpi-value">{quality['cluster_count']:,}</div></div>
          <div class="kpi-card"><div class="kpi-label">Noise / Singletons</div><div class="kpi-value">{quality['noise_count']:,}</div></div>
          <div class="kpi-card"><div class="kpi-label">Satellites</div><div class="kpi-value">{quality['satellite_count']}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Temporal Coverage: **{quality['time_span']}** · "
        f"Satellites Represented: **{quality['satellites']}** · "
        f"Clustering: DBSCAN 1.5 km haversine radius (Earth R ≈ 6,371 km)."
    )

    # ==================================================
    # 3. ANALYST PRIORITY PRESET / WEIGHT CONTROLS (PHASE 6)
    # ==================================================
    st.sidebar.header("Prioritization Presets & Weights")
    persona_choice = st.sidebar.selectbox(
        "Analyst Prioritization Preset",
        options=list(PERSONAS.keys()) + ["Custom Weights"],
        index=0,
        help="Analyst prioritization assumptions — NOT different predictive models.",
    )

    if persona_choice in PERSONAS:
        active_weights = PERSONAS[persona_choice]
    else:
        st.sidebar.caption("Adjust factor weights (automatically normalized to sum to 100%):")
        w_rec = st.sidebar.slider("Recency Weight", 0.0, 1.0, 0.30, 0.05)
        w_frp = st.sidebar.slider("FRP (Intensity) Weight", 0.0, 1.0, 0.30, 0.05)
        w_pers = st.sidebar.slider("Persistence Weight", 0.0, 1.0, 0.25, 0.05)
        w_size = st.sidebar.slider("Cluster Size Weight", 0.0, 1.0, 0.15, 0.05)
        active_weights = normalize_weights(
            {"recency": w_rec, "frp": w_frp, "persistence": w_pers, "size": w_size}
        )

    st.sidebar.markdown(
        f"**Active Weights ({persona_choice}):**<br/>"
        f"• Recency: `{active_weights['recency']:.0%}`<br/>"
        f"• FRP: `{active_weights['frp']:.0%}`<br/>"
        f"• Persistence: `{active_weights['persistence']:.0%}`<br/>"
        f"• Size: `{active_weights['size']:.0%}`",
        unsafe_allow_html=True,
    )

    # Queue view selection
    st.sidebar.divider()
    st.sidebar.header("Queue & Filters")
    use_cluster_view = st.sidebar.radio(
        "Triage Queue Mode",
        options=["Cluster Queue (Recommended)", "Observation Queue"],
        index=0,
    ) == "Cluster Queue (Recommended)" and (cluster_df is not None and not cluster_df.empty)

    working_df = cluster_df.copy() if use_cluster_view else raw_scored.copy()
    id_col = "cluster_id" if use_cluster_view else "obs_id"

    # Re-score with active weights
    working_df = add_priority_scores(working_df, weights=active_weights)
    if "item_id" not in working_df.columns:
        working_df["item_id"] = working_df[id_col]

    # Filters
    min_priority = st.sidebar.slider("Minimum Priority Score", 0.0, 100.0, 0.0, 5.0)
    min_frp = st.sidebar.slider("Minimum FRP (MW)", 0.0, float(working_df["frp"].max()), 0.0, 5.0)
    map_limit = st.sidebar.slider("Map Display Budget", 500, 5000, 2000, 500)

    # Filter
    filtered = working_df[
        (working_df["priority_score"] >= min_priority) & (working_df["frp"] >= min_frp)
    ].copy()

    if filtered.empty:
        st.warning("No observations or clusters match the current filters.")
        return

    filtered = filtered.sort_values("priority_score", ascending=False).reset_index(drop=True)
    filtered["current_rank"] = np.arange(1, len(filtered) + 1)

    # Consensus vs Contested tagging (Phase 6)
    consensus_series = compute_persona_consensus(filtered.head(100), consensus_cutoff=25)
    filtered["persona_alignment"] = filtered.index.map(consensus_series).fillna("Contested")

    # ==================================================
    # 4. RANKED QUEUE (PHASE 3, 6)
    # ==================================================
    st.subheader(f"Ranked Triage Queue ({'Spatial Clusters' if use_cluster_view else 'Individual Observations'})")
    st.caption("Sorted strictly by explainable priority score. Download queue for external review.")

    cols_to_show = [
        "current_rank",
        "item_id",
        "priority_score",
        "evidence_score",
        "observation_count",
        "persistence",
        "frp",
        "acq_date",
        "acq_time",
        "persona_alignment",
    ]
    if "confidence" in filtered.columns:
        cols_to_show.append("confidence")

    st.dataframe(
        filtered[cols_to_show].head(100),
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    csv_data = filtered[cols_to_show].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered & Ranked CSV Queue",
        data=csv_data,
        file_name="igneous_ranked_triage_queue.csv",
        mime="text/csv",
    )

    # ==================================================
    # 5. MAP (PHASE 11)
    # ==================================================
    st.subheader("Geospatial Review Map")
    st.caption(
        f"Plotting {'cluster centroids' if use_cluster_view else 'observations'}. "
        "Marker color and radius scale with review priority score."
    )
    render_map(filtered, map_limit, id_col=id_col)

    # ==================================================
    # 6. PRIORITY VS EVIDENCE (PHASE 4)
    # ==================================================
    st.subheader("Priority vs Evidence Quadrants")
    col_p, col_e = st.columns(2)
    with col_p:
        st.markdown(
            '<div class="concept-box"><div class="concept-title">Review Priority (Y-Axis)</div>'
            "<p>Answers: <em>How urgently should an analyst review this?</em></p>"
            '<p class="muted">Derived from recency decay, FRP intensity percentile, persistence, and cluster size. '
            "<strong>High priority does not mean high fire probability.</strong></p></div>",
            unsafe_allow_html=True,
        )
    with col_e:
        st.markdown(
            '<div class="concept-box"><div class="concept-title">Supporting Evidence (X-Axis)</div>'
            "<p>Answers: <em>How much supporting satellite information is present?</em></p>"
            '<p class="muted">Derived from data completeness, sensor confidence, and repeat observations. '
            "<strong>Evidence does not confirm a fire.</strong></p></div>",
            unsafe_allow_html=True,
        )

    render_priority_vs_evidence(filtered, id_col=id_col)

    # ==================================================
    # 7. SELECTED-ITEM DETAIL PANEL (PHASE 10)
    # ==================================================
    st.subheader("Selected Item Inspection & Audit")
    st.caption("Deep-dive inspection for human decision-support.")

    top_choices = filtered.head(50)
    options_labels = [
        f"Rank #{int(r.current_rank)} | ID: {r.item_id} | Priority: {r.priority_score:.1f} | "
        f"FRP: {r.frp:.1f} MW | Obs: {int(r.observation_count)} | ({r.latitude:.3f}, {r.longitude:.3f})"
        for r in top_choices.itertuples(index=False)
    ]
    selected_label = st.selectbox("Select Item to Inspect (Top 50)", options=options_labels)
    selected_rank = int(selected_label.split("|")[0].replace("Rank #", "").strip())
    selected_row = filtered.loc[filtered["current_rank"] == selected_rank].iloc[0]
    selected_id = selected_row["item_id"]

    # Compute/fetch robustness for top subset
    weights_tuple = tuple(sorted(active_weights.items()))
    robustness_map = get_cached_robustness(
        filtered.head(100),
        weights_tuple=weights_tuple,
        top_n=50,
        seed=42,
        id_col="item_id",
    )
    rob_info = robustness_map.get(
        selected_id,
        {
            "baseline_rank": selected_rank,
            "best_rank": selected_rank,
            "worst_rank": selected_rank,
            "rank_span": 0,
            "status": "Stable",
            "summary_text": f"Rank #{selected_rank} · Stable",
        },
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Rank & Priority", f"#{selected_rank} ({selected_row['priority_score']:.1f})")
    c2.metric("Evidence Score", f"{selected_row['evidence_score']:.1f} / 100")
    c3.metric("Observation Count", f"{int(selected_row['observation_count']):,}")
    c4.metric("Robustness Status", f"{rob_info['status']} (Span: {rob_info['rank_span']})")

    # ==================================================
    # 8. WHY RANKED HIGH (PHASE 3)
    # ==================================================
    st.markdown("#### Why is this ranked high?")
    explanation = explain_row_contributions(selected_row)

    contrib_data = [
        {"Factor": k.capitalize(), "Contribution (pts)": round(v, 2)}
        for k, v in explanation["top_drivers"]
    ]
    contrib_df = pd.DataFrame(contrib_data).set_index("Factor")

    col_bar, col_txt = st.columns([1, 1])
    with col_bar:
        st.bar_chart(contrib_df)
    with col_txt:
        top_driver_str = ", ".join(
            f"**{name.capitalize()}** ({val:.1f} pts)"
            for name, val in explanation["top_drivers"][:2]
        )
        st.markdown(f"**Primary Drivers:** {top_driver_str}")
        st.markdown(
            f"• **Recency Score:** `{explanation['factor_scores']['recency_score']:.2f}` (12h half-life decay)<br/>"
            f"• **FRP Percentile:** `{explanation['factor_scores']['frp_score']:.2f}` (log1p transformed)<br/>"
            f"• **Persistence Score:** `{explanation['factor_scores']['persistence_score']:.2f}` (repeat observations)<br/>"
            f"• **Cluster Size Score:** `{explanation['factor_scores']['size_score']:.2f}`",
            unsafe_allow_html=True,
        )
        if explanation["missing_information"]:
            st.markdown("**Data Caveats:**")
            for m in explanation["missing_information"]:
                st.caption(f"• {m}")

    # ==================================================
    # 9. ROBUSTNESS ANALYSIS (PHASE 5)
    # ==================================================
    st.markdown("#### Ranking Robustness & Sensitivity (Monte Carlo)")
    st.caption(
        "Evaluates ranking stability across 200 random weight perturbation simulations (fixed seed = 42). "
        "<strong>Robustness measures scoring sensitivity — NOT fire uncertainty.</strong>",
        unsafe_allow_html=True,
    )
    r1, r2, r3 = st.columns(3)
    r1.metric("Simulated Rank Range", f"#{rob_info['best_rank']} – #{rob_info['worst_rank']}")
    r2.metric("Rank Shift Span", f"{rob_info['rank_span']} positions")
    r3.metric("Stability Classification", rob_info["status"])

    # ==================================================
    # 10. COUNTERFACTUAL EXPLANATIONS (PHASE 7)
    # ==================================================
    st.markdown("#### What could change this ranking? (Counterfactuals)")
    cf_statements = compute_counterfactuals(
        filtered.head(100),
        item_id=selected_id,
        id_col="item_id",
        weights=active_weights,
    )
    for stmt in cf_statements:
        st.info(f"💡 {stmt}")

    # ==================================================
    # 8B. STATIC / PERSISTENT SOURCE SCREENING (PHASE 8)
    # ==================================================
    st.markdown("#### Static / Persistent Thermal Source Screening")
    screening_info = get_item_screening_info(selected_row, static_screening)
    if not static_screening.get("available", False):
        st.warning(f"⚠️ {screening_info['status']}")
        st.caption(screening_info["detail"])
    else:
        st.success(f"{screening_info['status']}: {screening_info['detail']}")

    # Underlying Observations expander (for clusters)
    if use_cluster_view:
        with st.expander(f"View Underlying Observations for Cluster #{selected_id}", expanded=False):
            underlying = raw_scored[raw_scored["cluster_id"] == selected_id]
            st.dataframe(
                underlying[
                    [
                        "obs_id",
                        "latitude",
                        "longitude",
                        "acq_date",
                        "acq_time",
                        "frp",
                        "confidence",
                        "satellite",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    # ==================================================
    # 11. ANALYST REVIEW WORKFLOW (PHASE 10)
    # ==================================================
    st.markdown("#### Analyst Review & Triage Notes")
    st.caption("Session-state review tracking. Preserved during active session.")

    status_options = ["Unreviewed", "Under Review", "Reviewed", "Dismissed"]
    curr_status = st.session_state.review_status.get(selected_id, "Unreviewed")
    curr_note = st.session_state.review_notes.get(selected_id, "")

    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        new_status = st.selectbox(
            "Review Status",
            options=status_options,
            index=status_options.index(curr_status) if curr_status in status_options else 0,
            key=f"status_select_{selected_id}",
        )
    with col_s2:
        new_note = st.text_area(
            "Analyst Notes",
            value=curr_note,
            key=f"note_area_{selected_id}",
            height=70,
        )

    if st.button("Save Review for Item", key=f"btn_save_{selected_id}"):
        st.session_state.review_status[selected_id] = new_status
        st.session_state.review_notes[selected_id] = new_note
        st.success(f"Review updated for Item #{selected_id}: {new_status}")

    reviewed_count = sum(1 for s in st.session_state.review_status.values() if s != "Unreviewed")
    st.caption(f"Total session items reviewed: **{reviewed_count}**")

    # ==================================================
    # 12. LIMITATIONS & SCIENTIFIC CAVEATS (PHASE 14, 16)
    # ==================================================
    st.markdown("---")
    with st.expander("Scientific Caveats & Operational Limitations", expanded=False):
        st.markdown(
            """
            - **Satellite Revisit Gaps:** Polar-orbiting satellites (VIIRS / Suomi NPP) provide periodic temporal snapshots. Temporal gaps between overpasses do NOT mean a fire has stopped.
            - **Occlusion & Distortions:** Clouds, thick smoke plumes, and dense forest canopy can obscure thermal radiation or lower measured Fire Radiative Power (FRP).
            - **Pixel Geolocation:** FIRMS coordinate locations reflect nominal pixel centers (≈375 m at nadir for VIIRS I-band), not the exact ignition point or fire perimeter.
            - **Persistent Non-Wildfire Sources:** Industrial flares, refinery heat, agricultural burn cycles, and geothermal anomalies register as thermal detections and require verification.
            - **No Ground-Truth Confirmation:** Igneous prioritizes satellite observations for human analyst attention. It does not replace on-scene emergency reconnaissance.
            - **Absence of Detection:** A lack of detection in a region is **never** proof that no fire exists.
            """
        )

    st.caption(
        f"Igneous Triage MVP · Source: `{Path(pipeline_res.source_path).name}` · "
        "Pipeline: Validation → Clustering → Aggregation → Priority → Evidence → Robustness → Review"
    )


if __name__ == "__main__":
    main()
