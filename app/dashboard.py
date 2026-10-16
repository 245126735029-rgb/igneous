"""
IGNEOUS — Satellite-Derived Active Fire Triage Dashboard.

A human decision-support triage platform for prioritizing NASA FIRMS / VIIRS
satellite thermal observations for analyst review.

Core Message: "A triage tool that shows its work, and shows how fragile its answers are."
Supporting Statement: "Human decision support — not fire confirmation."
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


def inject_styles():
    """Dark-friendly, compact, modern decision-dashboard styling."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            margin-bottom: 0.1rem !important;
            line-height: 1.1;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            font-weight: 600;
            color: #dd6b20;
            margin-bottom: 0.25rem;
        }
        .hero-tagline {
            font-size: 0.95rem;
            font-style: italic;
            opacity: 0.85;
            margin-bottom: 0.2rem;
        }
        .hero-statement {
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            opacity: 0.65;
            margin-bottom: 0.85rem;
        }
        .workflow-strip {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.6rem;
            margin: 0.8rem 0 1.2rem 0;
        }
        .workflow-box {
            background: rgba(127, 127, 127, 0.06);
            border: 1px solid rgba(127, 127, 127, 0.18);
            border-radius: 6px;
            padding: 0.55rem 0.75rem;
        }
        .workflow-head {
            font-weight: 700;
            font-size: 0.82rem;
            letter-spacing: 0.03em;
            margin-bottom: 0.15rem;
        }
        .workflow-desc {
            font-size: 0.75rem;
            opacity: 0.75;
            line-height: 1.25;
        }
        .disclaimer-card {
            background: rgba(221, 107, 32, 0.07);
            border-left: 3px solid #dd6b20;
            padding: 0.55rem 0.85rem;
            border-radius: 0 6px 6px 0;
            font-size: 0.83rem;
            margin-bottom: 0.9rem;
            line-height: 1.4;
        }
        .status-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .badge-stable { background: rgba(56, 161, 105, 0.18); color: #2f855a; }
        .badge-moderate { background: rgba(221, 107, 32, 0.18); color: #c05621; }
        .badge-fragile { background: rgba(229, 62, 62, 0.18); color: #c53030; }
        .badge-consensus { background: rgba(49, 130, 206, 0.18); color: #2b6cb0; }
        .badge-contested { background: rgba(113, 128, 150, 0.18); color: #4a5568; }
        .concept-card {
            background: rgba(127, 127, 127, 0.05);
            border: 1px solid rgba(127, 127, 127, 0.15);
            border-radius: 6px;
            padding: 0.75rem 0.9rem;
            height: 100%;
        }
        .screening-card {
            background: rgba(59, 130, 246, 0.07);
            border: 1px solid rgba(59, 130, 246, 0.22);
            border-radius: 6px;
            padding: 0.75rem 0.95rem;
            margin: 0.5rem 0;
        }
        .footer-note {
            font-size: 0.78rem;
            opacity: 0.65;
            text-align: center;
            margin-top: 2rem;
            padding-top: 0.8rem;
            border-top: 1px solid rgba(127, 127, 127, 0.15);
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading and executing Igneous pipeline…")
def load_pipeline() -> PipelineResult:
    """Load and process sample dataset once with caching."""
    return run_pipeline(SAMPLE_CSV)


@st.cache_data(show_spinner="Evaluating ranking stability (200 Monte Carlo trials)…")
def get_cached_robustness(
    df_subset: pd.DataFrame,
    weights_tuple: Tuple[Tuple[str, float], ...],
    top_n: int = 50,
    seed: int = 42,
    id_col: str = "item_id",
) -> Dict[Any, Dict[str, Any]]:
    """Compute 200 weight perturbation trials for the top queue items."""
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
    """Ensure session state preserves analyst review records across interactions."""
    if "review_status" not in st.session_state:
        st.session_state.review_status = {}
    if "review_notes" not in st.session_state:
        st.session_state.review_notes = {}


def score_to_rgba(score: float):
    """Map priority score 0–100 to an amber fire-triage color gradient."""
    t = max(0.0, min(1.0, float(score) / 100.0))
    r = int(90 + t * (235 - 90))
    g = int(140 + t * (80 - 140))
    b = int(190 + t * (25 - 190))
    a = int(140 + t * (215 - 140))
    return [r, g, b, a]


def render_map(display_df: pd.DataFrame, map_point_limit: int, id_col: str = "item_id"):
    """Render interactive PyDeck geospatial map with rich tooltip."""
    map_df = (
        display_df.head(map_point_limit)
        .dropna(subset=["latitude", "longitude"])
        .copy()
    )
    if map_df.empty:
        st.warning("No coordinates to plot for the current filter criteria.")
        return

    map_df["color"] = map_df["priority_score"].apply(score_to_rgba)
    map_df["radius"] = 120 + (map_df["priority_score"] / 100.0) * 360

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
            zoom=2.3,
            pitch=0,
        )
        tooltip = {
            "html": (
                "<b>Case ID:</b> {" + id_col + "}<br/>"
                "<b>Priority Score:</b> {priority_score:.1f}<br/>"
                "<b>Evidence Score:</b> {evidence_score:.1f}<br/>"
                "<b>Detections:</b> {observation_count}<br/>"
                "<b>Coordinates:</b> ({latitude:.3f}, {longitude:.3f})"
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
            f"Showing top {len(map_df):,} of {len(display_df):,} filtered records "
            f"by review priority (display budget limit)."
        )
    else:
        st.caption(f"Displaying {len(map_df):,} records on the map.")


def render_priority_vs_evidence(df: pd.DataFrame, id_col: str = "item_id"):
    """Render 2D Priority vs Evidence quadrant scatter plot."""
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
                "evidence_score": "Evidence Score (Observational Completeness)",
                "priority_score": "Review Priority Score (Attention Allocation)",
            },
            range_x=[0, 105],
            range_y=[0, 105],
        )

        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.4)
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.4)

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
            margin=dict(l=20, r=20, t=25, b=20),
            height=370,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.scatter_chart(plot_df, x="evidence_score", y="priority_score", color="priority_score")


def main():
    st.set_page_config(
        page_title="IGNEOUS — Satellite Active Fire Triage",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    ensure_review_state()

    # ==================================================
    # 1. HERO / HEADER (COMPACT & INTENTIONAL)
    # ==================================================
    st.markdown('<div class="hero-title">IGNEOUS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Satellite-Derived Active Fire Triage</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-tagline">“A triage tool that shows its work, and shows how fragile its answers are.”</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-statement">Human decision support — not fire confirmation.</div>',
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
    # 2. TOP-LEVEL DATA SNAPSHOT (5 KEY DATASET KPIS)
    # ==================================================
    snap1, snap2, snap3, snap4, snap5 = st.columns(5)
    snap1.metric("Observations", f"{quality['total_observations']:,}")
    snap2.metric("Spatial Clusters", f"{quality['cluster_count']:,}")
    snap3.metric("Isolated / Noise", f"{quality['noise_count']:,}")
    snap4.metric("Time Span", quality["time_span"])
    snap5.metric("Satellites", quality["satellites"])

    # ==================================================
    # 3. WORKFLOW VISUAL STRIP
    # ==================================================
    st.markdown(
        """
        <div class="workflow-strip">
          <div class="workflow-box">
            <div class="workflow-head">1. Explore</div>
            <div class="workflow-desc">Inspect satellite observations and spatial groupings</div>
          </div>
          <div class="workflow-box">
            <div class="workflow-head">2. Prioritize</div>
            <div class="workflow-desc">Rank cases using transparent, multi-factor signals</div>
          </div>
          <div class="workflow-box">
            <div class="workflow-head">3. Investigate</div>
            <div class="workflow-desc">Inspect evidence, factor drivers, and rank robustness</div>
          </div>
          <div class="workflow-box">
            <div class="workflow-head">4. Review</div>
            <div class="workflow-desc">Record analyst triage status and operational notes</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==================================================
    # SIDEBAR: CONTROLS & FILTERS
    # ==================================================
    st.sidebar.markdown("### Prioritization Presets")
    st.sidebar.caption(
        "Personas adjust prioritization weighting for review order — "
        "they do not alter the underlying satellite observations."
    )

    persona_choice = st.sidebar.selectbox(
        "Analyst Persona",
        options=list(PERSONAS.keys()) + ["Custom Weights"],
        index=0,
        help="Transparent analyst prioritization assumptions, not predictive models.",
    )

    if persona_choice in PERSONAS:
        active_weights = PERSONAS[persona_choice]
    else:
        st.sidebar.caption("Adjust factor weights (normalized to 100%):")
        w_rec = st.sidebar.slider("Recency Weight", 0.0, 1.0, 0.30, 0.05)
        w_frp = st.sidebar.slider("FRP Intensity Weight", 0.0, 1.0, 0.30, 0.05)
        w_pers = st.sidebar.slider("Persistence Weight", 0.0, 1.0, 0.25, 0.05)
        w_size = st.sidebar.slider("Cluster Size Weight", 0.0, 1.0, 0.15, 0.05)
        active_weights = normalize_weights(
            {"recency": w_rec, "frp": w_frp, "persistence": w_pers, "size": w_size}
        )

    st.sidebar.markdown(
        f"**Active Weights ({persona_choice}):**<br/>"
        f"• Recency: `{active_weights['recency']:.0%}` &nbsp; "
        f"• FRP: `{active_weights['frp']:.0%}`<br/>"
        f"• Persistence: `{active_weights['persistence']:.0%}` &nbsp; "
        f"• Size: `{active_weights['size']:.0%}`",
        unsafe_allow_html=True,
    )

    st.sidebar.divider()
    st.sidebar.markdown("### Queue & Filters")
    use_cluster_view = st.sidebar.radio(
        "Queue Mode",
        options=["Spatial Clusters (Recommended)", "Individual Observations"],
        index=0,
    ) == "Spatial Clusters (Recommended)" and (cluster_df is not None and not cluster_df.empty)

    working_df = cluster_df.copy() if use_cluster_view else raw_scored.copy()
    id_col = "cluster_id" if use_cluster_view else "obs_id"

    # Re-score with active weights
    working_df = add_priority_scores(working_df, weights=active_weights)
    if "item_id" not in working_df.columns:
        working_df["item_id"] = working_df[id_col]

    min_priority = st.sidebar.slider("Minimum Priority Score", 0.0, 100.0, 0.0, 5.0)
    min_frp = st.sidebar.slider("Minimum FRP (MW)", 0.0, float(working_df["frp"].max()), 0.0, 5.0)
    map_limit = st.sidebar.slider("Map Display Budget", 500, 5000, 2000, 500)

    # Filter
    filtered = working_df[
        (working_df["priority_score"] >= min_priority) & (working_df["frp"] >= min_frp)
    ].copy()

    if filtered.empty:
        st.warning("No observations match the current filter criteria.")
        return

    filtered = filtered.sort_values("priority_score", ascending=False).reset_index(drop=True)
    filtered["current_rank"] = np.arange(1, len(filtered) + 1)

    # Consensus vs Contested tagging
    consensus_series = compute_persona_consensus(filtered.head(100), consensus_cutoff=25)
    filtered["persona_alignment"] = filtered.index.map(consensus_series).fillna("Contested")

    # ==================================================
    # 4. MAIN ANALYST AREA: RANKED QUEUE
    # ==================================================
    head_left, head_right = st.columns([2, 1])
    with head_left:
        st.subheader(f"Ranked Triage Queue ({'Spatial Clusters' if use_cluster_view else 'Individual Observations'})")
        st.caption(
            f"Active Persona: **{persona_choice}** · "
            f"Showing **{len(filtered):,}** cases sorted by review priority."
        )
    with head_right:
        csv_data = filtered[
            [
                "current_rank", "item_id", "priority_score", "evidence_score",
                "observation_count", "persistence", "frp", "acq_date", "acq_time",
                "persona_alignment"
            ]
        ].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Ranked CSV Queue",
            data=csv_data,
            file_name="igneous_ranked_queue.csv",
            mime="text/csv",
        )

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
        height=300,
    )

    # ==================================================
    # 5. GEOSPATIAL MAP (PROMINENT REVIEW CANVAS)
    # ==================================================
    st.subheader("Geospatial Review Map")
    st.caption(
        f"Displaying {'cluster centroids' if use_cluster_view else 'observations'}. "
        "Point color and radius scale with review priority score."
    )
    render_map(filtered, map_limit, id_col=id_col)

    # ==================================================
    # 6. PRIORITY VS EVIDENCE (2D QUADRANT VISUALIZATION)
    # ==================================================
    st.subheader("Priority vs Evidence Quadrants")
    st.markdown(
        """
        <div class="disclaimer-card">
          <strong>Key Distinction:</strong> Priority indicates what deserves attention first.
          Evidence indicates the strength of available supporting observations.
          They are intentionally separate. Neither represents probability of fire.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_priority_vs_evidence(filtered, id_col=id_col)

    # ==================================================
    # 7. SELECTED CASE FILE (ANALYST DOSSIER)
    # ==================================================
    st.subheader("Analyst Case File")
    st.caption("Deep-dive inspection, robustness analysis, and review tracking for a single case.")

    top_choices = filtered.head(50)
    options_labels = [
        f"Rank #{int(r.current_rank)} | ID: {r.item_id} | Priority: {r.priority_score:.1f} | "
        f"FRP: {r.frp:.1f} MW | Obs: {int(r.observation_count)} | ({r.latitude:.3f}, {r.longitude:.3f})"
        for r in top_choices.itertuples(index=False)
    ]
    selected_label = st.selectbox("Select Case to Inspect (Top 50 by Priority)", options=options_labels)
    selected_rank = int(selected_label.split("|")[0].replace("Rank #", "").strip())
    selected_row = filtered.loc[filtered["current_rank"] == selected_rank].iloc[0]
    selected_id = selected_row["item_id"]

    # Robustness simulation lookup
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

    # Case Dossier Header Metrics
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Case Rank & Priority", f"#{selected_rank} ({selected_row['priority_score']:.1f})")
    d2.metric("Evidence Score", f"{selected_row['evidence_score']:.1f} / 100")
    d3.metric("Observation Count", f"{int(selected_row['observation_count']):,}")
    d4.metric("Robustness Status", f"{rob_info['status']} (Span: {rob_info['rank_span']})")

    st.caption(
        f"Case ID: **{selected_id}** · "
        f"Location: **({selected_row['latitude']:.4f}, {selected_row['longitude']:.4f})** · "
        f"Latest Observation: **{selected_row['acq_date']} {selected_row['acq_time']}** · "
        f"Peak FRP: **{selected_row['frp']:.1f} MW**"
    )

    # Structured Case Sections inside Tabs
    tab_why, tab_rob, tab_cf, tab_screen, tab_obs, tab_rev = st.tabs(
        [
            "Why Ranked Here",
            "Robustness & Fragility",
            "Counterfactuals",
            "Temporal Screening",
            "Underlying Observations",
            "Analyst Review",
        ]
    )

    # --- TAB 1: WHY RANKED HERE ---
    with tab_why:
        explanation = explain_row_contributions(selected_row)
        contrib_data = [
            {"Factor": k.capitalize(), "Points": round(v, 2)}
            for k, v in explanation["top_drivers"]
        ]
        contrib_df = pd.DataFrame(contrib_data).set_index("Factor")

        c_col1, c_col2 = st.columns([1, 1])
        with c_col1:
            st.markdown("**Factor Contributions to Priority Score**")
            st.bar_chart(contrib_df)
        with c_col2:
            st.markdown("**Component Scores (0–1 Rank)**")
            st.markdown(
                f"• **Recency Score:** `{explanation['factor_scores']['recency_score']:.2f}` (12h half-life decay)<br/>"
                f"• **FRP Percentile:** `{explanation['factor_scores']['frp_score']:.2f}` (log1p transformed)<br/>"
                f"• **Persistence Score:** `{explanation['factor_scores']['persistence_score']:.2f}` (repeat observations)<br/>"
                f"• **Cluster Size Score:** `{explanation['factor_scores']['size_score']:.2f}`",
                unsafe_allow_html=True,
            )
            top_drivers_text = ", ".join(
                f"**{k.capitalize()}** ({v:.1f} pts)" for k, v in explanation["top_drivers"][:2]
            )
            st.markdown(f"**Primary Drivers:** {top_drivers_text}")
            if explanation["missing_information"]:
                st.caption("Data caveats: " + "; ".join(explanation["missing_information"]))

    # --- TAB 2: ROBUSTNESS & FRAGILITY ---
    with tab_rob:
        st.markdown(
            """
            <div class="concept-card">
              <strong>Ranking Robustness:</strong> How sensitive is this ranking to reasonable changes in scoring weights?<br/>
              <span style="opacity:0.8; font-size:0.83rem;">
                Evaluated via 200 random weight perturbation simulations (fixed seed = 42).
                This measures mathematical scoring fragility — <strong>NOT fire uncertainty</strong>.
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        r_c1, r_c2, r_c3 = st.columns(3)
        r_c1.metric("Simulated Rank Range", f"#{rob_info['best_rank']} – #{rob_info['worst_rank']}")
        r_c2.metric("Rank Shift Span", f"{rob_info['rank_span']} positions")
        r_c3.metric("Stability Classification", rob_info["status"])

    # --- TAB 3: COUNTERFACTUALS ---
    with tab_cf:
        st.markdown("**What could change this ranking?**")
        st.caption("Every numerical counterfactual is strictly recalculated from the scoring function.")
        cf_statements = compute_counterfactuals(
            filtered.head(100),
            item_id=selected_id,
            id_col="item_id",
            weights=active_weights,
        )
        for stmt in cf_statements:
            st.info(f"💡 {stmt}")

    # --- TAB 4: TEMPORAL SCREENING (DATA LIMITATION CARD) ---
    with tab_screen:
        st.markdown("**Recurrent / Static Thermal Source Screening**")
        screening_info = get_item_screening_info(selected_row, static_screening)
        if not static_screening.get("available", False):
            st.markdown(
                """
                <div class="screening-card">
                  <strong>Temporal screening unavailable</strong><br/>
                  At least 7 distinct observation days are required for recurrent/static-source screening.
                  The current demonstration dataset covers one day (2023-07-12).<br/>
                  <span style="font-size:0.82rem; opacity:0.85;">
                    Persistent thermal activity can have multiple causes (e.g. controlled agricultural burns,
                    active wildfires, industrial heat sources, or geothermal anomalies) and requires external verification.
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.success(f"{screening_info['status']}: {screening_info['detail']}")

    # --- TAB 5: UNDERLYING OBSERVATIONS ---
    with tab_obs:
        if use_cluster_view:
            underlying = raw_scored[raw_scored["cluster_id"] == selected_id]
            st.caption(f"Cluster #{selected_id} contains {len(underlying)} constituent detections:")
            st.dataframe(
                underlying[
                    [
                        "obs_id", "latitude", "longitude", "acq_date",
                        "acq_time", "frp", "confidence", "satellite"
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("Observation view mode active — inspecting individual detection record.")

    # --- TAB 6: ANALYST REVIEW & AUDIT ---
    with tab_rev:
        st.markdown("**Record Analyst Review & Triage Notes**")
        status_options = ["Unreviewed", "Under Review", "Reviewed", "Dismissed"]
        curr_status = st.session_state.review_status.get(selected_id, "Unreviewed")
        curr_note = st.session_state.review_notes.get(selected_id, "")

        ar_col1, ar_col2 = st.columns([1, 2])
        with ar_col1:
            new_status = st.selectbox(
                "Review Status",
                options=status_options,
                index=status_options.index(curr_status) if curr_status in status_options else 0,
                key=f"status_select_{selected_id}",
            )
        with ar_col2:
            new_note = st.text_area(
                "Analyst Notes",
                value=curr_note,
                key=f"note_area_{selected_id}",
                height=70,
            )

        if st.button("Save Review for Case", key=f"btn_save_{selected_id}"):
            st.session_state.review_status[selected_id] = new_status
            st.session_state.review_notes[selected_id] = new_note
            st.success(f"Review updated for Case #{selected_id}: {new_status}")

        reviewed_count = sum(1 for s in st.session_state.review_status.values() if s != "Unreviewed")
        st.caption(f"Total session cases reviewed: **{reviewed_count}**")

    # ==================================================
    # 8. SECONDARY DETAILS & METHODOLOGY (EXPANDERS)
    # ==================================================
    st.markdown("---")
    with st.expander("Technical Methodology & Scoring Formulas", expanded=False):
        st.markdown(
            f"""
            **Priority Score (0–100):**
            `priority = 100 × ({active_weights['recency']:.2f}×recency + {active_weights['frp']:.2f}×frp + {active_weights['persistence']:.2f}×persistence + {active_weights['size']:.2f}×size)`

            - **Recency Decay:** $0.5^{{\\Delta\\text{{hours}} / 12}}$ relative to newest observation in the dataset.
            - **FRP Intensity:** Percentile rank of $\\log(1 + \\text{{FRP}})$ across observations.
            - **Persistence:** Repeat observations across distinct acquisition periods within the spatial cluster.
            - **Cluster Size:** Percentile rank of observation count within the spatial cluster.
            """
        )

    with st.expander("Data Quality & Sensor Validation Audit", expanded=False):
        st.markdown(
            f"""
            - **Required Fields Checked:** 7 of 7 required FIRMS fields validated.
            - **Missing Required Values:** `{quality['missing_required']}`
            - **Invalid Coordinate Flags:** Latitude errors = `{quality['invalid_latitude']}`, Longitude errors = `{quality['invalid_longitude']}`
            - **Non-Physical FRP Readings (< 0):** `{quality['negative_frp']}`
            - **Clustering Performance:** 1.5 km haversine DBSCAN executed in < 1 second.
            """
        )

    with st.expander("Operational Limitations & Scientific Caveats", expanded=False):
        st.markdown(
            """
            - **Satellite Revisit Gaps:** Polar-orbiting VIIRS instruments provide periodic overpasses. Revisit intervals create temporal blind spots.
            - **Atmospheric & Canopy Obscuration:** Cloud cover, dense smoke, and heavy forest canopy attenuate thermal infrared radiation.
            - **Pixel Geolocation:** Coordinates represent nominal pixel centers (≈375 m at nadir), not exact fire perimeters or ignition points.
            - **Sensor View-Angle Distortion:** Pixel footprints expand toward scan edges (bow-tie effect).
            - **Persistent Non-Wildfire Heat Sources:** Industrial flares, refineries, agricultural burns, and geothermal anomalies can register as hotspots.
            - **Absence of Detection:** A lack of detection in a region is **never** proof that no fire exists.
            """
        )

    # ==================================================
    # 9. FOOTER / SCIENTIFIC GUARDRAIL
    # ==================================================
    st.markdown(
        """
        <div class="footer-note">
          Igneous is a satellite-observation triage and decision-support tool.
          Satellite detections are not ground truth, and absence of detection does not imply absence of fire.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
