"""
Robustness and ranking sensitivity analysis for Igneous satellite fire triage.

Quantifies how fragile a priority score and ranking are to scoring assumptions.
Scientific Note:
- Ranking sensitivity to scoring assumptions is NOT real-world fire uncertainty.
- Simulates weight perturbations with a fixed NumPy seed for determinism.
- Counterfactual explanations are strictly derived from recalculating the scoring function.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.scoring import (
    WEIGHTS,
    PERSONAS,
    add_priority_scores,
    normalize_weights,
)

# Explicit robustness thresholds for rank shift span
ROBUSTNESS_STABLE_SPAN = 5  # span <= 5 ranks -> Stable
ROBUSTNESS_MODERATE_SPAN = 15  # 6 <= span <= 15 ranks -> Moderately sensitive
# span > 15 ranks -> Fragile


def explain_row_contributions(row: pd.Series) -> Dict[str, Any]:
    """
    Build a human-readable explanation for one scored observation or cluster.

    Returns a dict with scores, contributions, drivers, and caveats.
    """
    contributions = {
        "recency": float(row.get("recency_contribution", 0.0)),
        "frp": float(row.get("frp_contribution", 0.0)),
        "persistence": float(row.get("persistence_contribution", 0.0)),
        "size": float(row.get("size_contribution", 0.0)),
    }
    ranked = sorted(contributions.items(), key=lambda item: item[1], reverse=True)

    missing = []
    persistence = row.get("persistence", 1)
    obs_count = row.get("observation_count", 1)
    if float(persistence) == 1 and float(obs_count) == 1 and "cluster_id" not in row:
        missing.append(
            "Cluster persistence and size are provisional defaults (both = 1) "
            "until clustering is integrated."
        )
    if str(row.get("sensor_pass_agreement", "")).startswith("Unavailable"):
        missing.append("Sensor/pass agreement is unavailable without clustering.")
    if pd.isna(row.get("frp")):
        missing.append("FRP is missing, so the FRP factor scores 0.")

    return {
        "priority_score": float(row.get("priority_score", 0.0)),
        "contributions": contributions,
        "top_drivers": ranked,
        "missing_information": missing,
        "factor_scores": {
            "recency_score": float(row.get("recency_score", 0.0)),
            "frp_score": float(row.get("frp_score", 0.0)),
            "persistence_score": float(row.get("persistence_score", 0.0)),
            "size_score": float(row.get("size_score", 0.0)),
        },
    }


def sensitivity_with_alt_weights(
    df: pd.DataFrame, obs_id: Any, weights: Optional[Dict[str, float]] = None
) -> Optional[Dict[str, Any]]:
    """
    Re-score with alternate weights and compare one observation or cluster.
    """
    if weights is None:
        weights = {
            "recency": 0.20,
            "frp": 0.40,
            "persistence": 0.25,
            "size": 0.15,
        }

    needed = [
        "acq_date",
        "acq_time",
        "frp",
        "persistence",
        "observation_count",
    ]
    base_cols = [c for c in needed if c in df.columns]
    rescored = add_priority_scores(df[base_cols].copy(), weights=weights)

    id_col = "obs_id" if "obs_id" in df.columns else ("cluster_id" if "cluster_id" in df.columns else None)
    if id_col is None:
        return None

    rescored[id_col] = df[id_col].values

    original = df.loc[df[id_col] == obs_id]
    alternate = rescored.loc[rescored[id_col] == obs_id]
    if original.empty or alternate.empty:
        return None

    original_score = float(original.iloc[0]["priority_score"])
    alt_score = float(alternate.iloc[0]["priority_score"])
    return {
        "default_weights": WEIGHTS,
        "alternate_weights": weights,
        "original_priority": original_score,
        "alternate_priority": alt_score,
        "absolute_difference": abs(alt_score - original_score),
        "note": (
            "If this difference is large, the ranking is sensitive to weight choices. "
            "That is expected for triage scoring — not a fire-danger estimate."
        ),
    }


def simulate_rank_robustness(
    df: pd.DataFrame,
    n_simulations: int = 200,
    top_n: int = 50,
    seed: int = 42,
    baseline_weights: Optional[Dict[str, float]] = None,
    id_col: Optional[str] = None,
) -> Dict[Any, Dict[str, Any]]:
    """
    Simulate weight perturbation across 200 runs to evaluate ranking sensitivity.

    Uses a fixed NumPy random seed for deterministic reproducibility.
    Computes baseline rank, best rank, worst rank, median rank, and rank span.
    Labels items as Stable, Moderately sensitive, or Fragile.

    Parameters
    ----------
    df : pd.DataFrame
        Scored DataFrame containing factor scores or raw fields.
    n_simulations : int
        Number of perturbation trials (default 200).
    top_n : int
        Number of top items to evaluate.
    seed : int
        Fixed random seed for reproducible Monte Carlo simulations.
    baseline_weights : dict, optional
        Starting weights (default WEIGHTS).
    id_col : str, optional
        Identifier column name ('obs_id' or 'cluster_id').

    Returns
    -------
    dict
        Mapping of item ID -> robustness statistics.
    """
    if df.empty:
        return {}

    if baseline_weights is None:
        baseline_weights = WEIGHTS
    base_w = normalize_weights(baseline_weights)

    if id_col is None:
        id_col = "obs_id" if "obs_id" in df.columns else ("cluster_id" if "cluster_id" in df.columns else None)

    # Ensure factor scores exist
    scored = df.copy()
    factor_cols = ["recency_score", "frp_score", "persistence_score", "size_score"]
    if not all(col in scored.columns for col in factor_cols):
        scored = add_priority_scores(scored, weights=base_w)

    # Sort by baseline priority
    scored = scored.sort_values("priority_score", ascending=False).reset_index(drop=True)
    candidate_pool_size = min(len(scored), max(top_n * 3, 100))
    candidates = scored.head(candidate_pool_size).copy()

    factors = candidates[factor_cols].to_numpy(dtype=float)  # shape (M, 4)

    # Generate 200 deterministic weight perturbations
    rng = np.random.RandomState(seed)
    w_vec = np.array([base_w["recency"], base_w["frp"], base_w["persistence"], base_w["size"]])

    # Perturbations around base weights
    noise = rng.uniform(-0.12, 0.12, size=(n_simulations, 4))
    sim_weights = np.clip(w_vec + noise, 0.05, 0.70)
    # Normalize rows to sum to 1.0
    sim_weights /= sim_weights.sum(axis=1, keepdims=True)
    # Ensure trial 0 is exactly baseline
    sim_weights[0] = w_vec

    # Fast matrix multiplication: (M, 4) @ (4, n_simulations) -> (M, n_simulations)
    sim_scores = factors @ sim_weights.T * 100.0

    # Ranks across simulations: shape (M, n_simulations), 1-indexed
    # Note: argsort of negative scores yields ranks
    order = np.argsort(-sim_scores, axis=0)
    sim_ranks = np.empty_like(order)
    for col_idx in range(n_simulations):
        sim_ranks[order[:, col_idx], col_idx] = np.arange(1, len(candidates) + 1)

    eval_count = min(top_n, len(candidates))
    results = {}

    for i in range(eval_count):
        item_id = candidates.iloc[i][id_col] if id_col in candidates.columns else i
        ranks_i = sim_ranks[i, :]

        baseline_rank = int(ranks_i[0])
        best_rank = int(np.min(ranks_i))
        worst_rank = int(np.max(ranks_i))
        median_rank = float(np.median(ranks_i))
        rank_span = worst_rank - best_rank

        if rank_span <= ROBUSTNESS_STABLE_SPAN:
            status = "Stable"
        elif rank_span <= ROBUSTNESS_MODERATE_SPAN:
            status = "Moderately sensitive"
        else:
            status = "Fragile"

        results[item_id] = {
            "baseline_rank": baseline_rank,
            "best_rank": best_rank,
            "worst_rank": worst_rank,
            "median_rank": median_rank,
            "rank_span": rank_span,
            "status": status,
            "summary_text": f"Rank #{baseline_rank} (Range #{best_rank}–#{worst_rank}) · {status}",
        }

    return results


def compute_counterfactuals(
    df: pd.DataFrame,
    item_id: Any,
    id_col: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Generate concrete counterfactual statements for an item by RECALCULATING the scoring function.

    Every number and rank reported is derived from actual recomputation.
    Never invents numbers or rank shifts.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset of observations or clusters.
    item_id : Any
        Identifier of the target item.
    id_col : str, optional
        ID column ('obs_id' or 'cluster_id').
    weights : dict, optional
        Baseline weights.

    Returns
    -------
    list of str
        Concrete counterfactual explanation statements.
    """
    if df.empty:
        return ["No observations available for counterfactual evaluation."]

    if id_col is None:
        id_col = "obs_id" if "obs_id" in df.columns else ("cluster_id" if "cluster_id" in df.columns else None)
    if id_col is None or id_col not in df.columns:
        return ["Counterfactual evaluation requires a valid identifier column."]

    if weights is None:
        weights = WEIGHTS
    base_w = normalize_weights(weights)

    # Base scoring
    scored = add_priority_scores(df.copy(), weights=base_w)
    scored["_rank"] = scored["priority_score"].rank(ascending=False, method="min").astype(int)

    target_match = scored[scored[id_col] == item_id]
    if target_match.empty:
        return ["Selected item not found in dataset."]

    base_rank = int(target_match.iloc[0]["_rank"])
    explanations: List[str] = []

    # --- Counterfactual 1: +1 Repeat Observation / Persistence ---
    alt_df_persist = df.copy()
    match_idx = alt_df_persist[alt_df_persist[id_col] == item_id].index
    if not match_idx.empty:
        idx = match_idx[0]
        cur_persist = alt_df_persist.at[idx, "persistence"] if "persistence" in alt_df_persist.columns else 1
        cur_count = alt_df_persist.at[idx, "observation_count"] if "observation_count" in alt_df_persist.columns else 1

        alt_df_persist.at[idx, "persistence"] = cur_persist + 1
        alt_df_persist.at[idx, "observation_count"] = cur_count + 1

        re_scored = add_priority_scores(alt_df_persist, weights=base_w)
        re_rank = int(re_scored["priority_score"].rank(ascending=False, method="min").loc[idx])

        if re_rank < base_rank:
            explanations.append(
                f"One additional repeat observation would move this item approximately "
                f"from rank #{base_rank} to rank #{re_rank}."
            )
        elif re_rank > base_rank:
            explanations.append(
                f"With adjusted persistence, this item would rank at #{re_rank} (currently #{base_rank})."
            )

    # --- Counterfactual 2: Stronger Recency Weighting (Newest-first persona) ---
    recency_preset = PERSONAS.get("Newest-first")
    if recency_preset:
        recency_scored = add_priority_scores(df.copy(), weights=recency_preset)
        recency_rank = int(
            recency_scored.loc[recency_scored[id_col] == item_id, "priority_score"]
            .rank(ascending=False, method="min")
            .iloc[0]
        )
        if recency_rank > 10 and base_rank <= 10:
            explanations.append(
                f"With stronger recency weighting (50%), this item would fall from "
                f"rank #{base_rank} to rank #{recency_rank} (outside the top 10)."
            )
        elif abs(recency_rank - base_rank) >= 2:
            direction = "improve" if recency_rank < base_rank else "drop"
            explanations.append(
                f"With stronger recency weighting (50%), this item would {direction} "
                f"from rank #{base_rank} to rank #{recency_rank}."
            )

    # --- Counterfactual 3: Stronger FRP Weighting (Most-intense persona) ---
    frp_preset = PERSONAS.get("Most-intense")
    if frp_preset:
        frp_scored = add_priority_scores(df.copy(), weights=frp_preset)
        frp_rank = int(
            frp_scored.loc[frp_scored[id_col] == item_id, "priority_score"]
            .rank(ascending=False, method="min")
            .iloc[0]
        )
        if abs(frp_rank - base_rank) >= 2:
            support_desc = "strongly supported" if frp_rank <= base_rank else "sensitive to"
            explanations.append(
                f"This item's ranking is {support_desc} FRP weighting: under 50% FRP weight, "
                f"it shifts from rank #{base_rank} to rank #{frp_rank}."
            )

    # --- Counterfactual 4: Stronger Persistence Weighting (Persistence-first persona) ---
    persist_preset = PERSONAS.get("Persistence-first")
    if persist_preset:
        persist_scored = add_priority_scores(df.copy(), weights=persist_preset)
        persist_rank = int(
            persist_scored.loc[persist_scored[id_col] == item_id, "priority_score"]
            .rank(ascending=False, method="min")
            .iloc[0]
        )
        if abs(persist_rank - base_rank) >= 2:
            explanations.append(
                f"This item is sensitive to persistence weighting: under 45% persistence weight, "
                f"it shifts from rank #{base_rank} to rank #{persist_rank}."
            )

    if not explanations:
        return ["No simple one-factor change materially alters this ranking."]

    return explanations
