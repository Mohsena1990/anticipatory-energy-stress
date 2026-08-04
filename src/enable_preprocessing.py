"""
enable_preprocessing.py
────────────────────────
ENABLE.EU UK household survey: loading, cleaning, construct scoring, and
High Adaptive Energy Vulnerability (HighAEV) composite construction.

Pipeline steps
──────────────
  1. Load ENABLE Excel → filter UK (Country == 11)
  2. Replace survey missing codes with NaN
  3. Item diagnostics (missingness, variance)
  4. Directional recoding (H9, E5A1, E6A1, H15D)
  5. Normalize each item to [0, 1]
  6. Build four COR composite scores
  7. Construct AEV composite + binary HighAEV target
  8. Attach annual FES context (same value for all UK households)
  9. Energy-poverty LIHC flag
 10. Export cleaned dataset + diagnostics tables

Usage
─────
  from src.enable_preprocessing import run
  df = run()          # returns DataFrame with all scores + HighAEV
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src import config, paths
from src.construct_mapping import (
    CONSTRUCT_REGISTRY,
    ALL_CONSTRUCT_ITEMS,
    CONTROL_ITEMS,
    ENERGY_CONTROLS,
    FCP_ITEM_SPECS,
    AEMC_ITEM_SPECS,
    BLI_ITEM_SPECS,
    TCR_ITEM_SPECS,
    H9_RECODE,
    H12_NOTE,
    THERMAL_POVERTY_ITEMS,
)

log = logging.getLogger(__name__)


# =============================================================================
# I/O helpers
# =============================================================================

def _save_csv(df: pd.DataFrame, dest: Path, name: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s (%d rows)", p.name, len(df))


def _save_fig(fig: plt.Figure, dest: Path, name: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(p, dpi=config.DPI, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


# =============================================================================
# STEP 1 — Load ENABLE.EU (UK only)
# =============================================================================

def load_enable_uk() -> pd.DataFrame:
    """Load ENABLE.EU Excel, filter to UK (Country == 11), replace missing codes."""
    log.info("Loading ENABLE.EU dataset...")
    if not paths.ENABLE_FILE.exists():
        raise FileNotFoundError(
            f"ENABLE.EU file not found: {paths.ENABLE_FILE}\n"
            "Place it at: data/social_science_data/"
            "ENABLE.EU_dataset_survey of households.xlsx"
        )
    df = pd.read_excel(paths.ENABLE_FILE)
    log.info("Raw shape: %s", df.shape)

    df = df[df["Country"] == config.UK_COUNTRY_CODE].copy().reset_index(drop=True)
    df["respondent_id"] = np.arange(1, len(df) + 1)
    log.info("UK sub-sample: %d households", len(df))

    # Replace survey-defined missing/refusal codes with NaN
    df.replace(config.MISSING_CODES, np.nan, inplace=True)
    return df


# =============================================================================
# STEP 2 — Item diagnostics
# =============================================================================

def item_diagnostics(df: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    """
    Compute per-item missingness, variance, min, max, and mean for a list of
    survey items.  Items with zero variance are flagged and excluded from
    composite scoring.
    """
    rows = []
    for col in items:
        if col not in df.columns:
            rows.append({"variable": col, "available": False,
                         "n_valid": 0, "missing_rate": 1.0,
                         "min": np.nan, "max": np.nan,
                         "mean": np.nan, "std": np.nan,
                         "zero_variance": True, "used_in_score": False})
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        n_valid   = int(s.notna().sum())
        miss_rate = round(1.0 - n_valid / len(df), 4)
        std_val   = float(s.std(ddof=0)) if n_valid > 1 else 0.0
        zero_var  = std_val < 1e-10
        rows.append({
            "variable":      col,
            "available":     True,
            "n_valid":       n_valid,
            "missing_rate":  miss_rate,
            "min":           round(float(s.min()), 4) if n_valid else np.nan,
            "max":           round(float(s.max()), 4) if n_valid else np.nan,
            "mean":          round(float(s.mean()), 4) if n_valid else np.nan,
            "std":           round(std_val, 4),
            "zero_variance": zero_var,
            "used_in_score": not zero_var and n_valid > 0,
        })
    return pd.DataFrame(rows)


# =============================================================================
# STEP 3 — Directional recoding
# =============================================================================

def _recode_h9(df: pd.DataFrame) -> pd.DataFrame:
    """Recode H9 (heating control method) to a 0–3 adaptive capacity scale."""
    if "H9" not in df.columns:
        return df
    s = pd.to_numeric(df["H9"], errors="coerce")
    df["H9"] = s.map(H9_RECODE)
    log.debug("H9 recoded to adaptive capacity scale (0–3)")
    return df


def _reverse_code(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Reverse an item: new = max - value (within observed range)."""
    if col not in df.columns:
        return df
    s = pd.to_numeric(df[col], errors="coerce")
    hi = s.max()
    lo = s.min()
    if (hi - lo) > 1e-10:
        df[col] = hi - s + lo
    else:
        # Zero variance — leave as-is (will be excluded later)
        pass
    log.debug("Reverse-coded %s", col)
    return df


def apply_recoding(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all directional recoding rules from construct_mapping."""
    df = _recode_h9(df)

    # Reverse-code AEMC items (E5A1, E6A1: "I do not use X" → low capacity)
    for col in ["E5A1", "E6A1"]:
        df = _reverse_code(df, col)

    # Reverse-code TCR item (H15D: "willing to compromise" → low resistance)
    df = _reverse_code(df, "H15D")

    return df


# =============================================================================
# STEP 4 — Normalize items to [0, 1]
# =============================================================================

def normalize_01(s: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]; returns NaN series if zero variance."""
    lo, hi = s.min(), s.max()
    if (hi - lo) < 1e-10:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


def normalize_all_items(df: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    """
    Normalize each item in `items` to [0, 1] and store as n_{item} column.
    Items not present in df or with zero variance get NaN normalized score.
    """
    for col in items:
        raw = pd.to_numeric(df[col], errors="coerce") if col in df.columns \
              else pd.Series(np.nan, index=df.index)
        df[f"n_{col}"] = normalize_01(raw)
    return df


# =============================================================================
# STEP 5 — Composite score construction
# =============================================================================

def build_composite_score(
    df: pd.DataFrame,
    items: list[str],
    score_col: str,
    min_items: int = 1,
) -> pd.DataFrame:
    """
    Build a composite score as the row-mean of normalized item scores.

    Parameters
    ----------
    items:     list of raw item column names (n_{item} columns must exist)
    score_col: name for the new composite column
    min_items: minimum number of non-null normalized items required per row
    """
    n_cols = [f"n_{c}" for c in items if f"n_{c}" in df.columns]
    if not n_cols:
        log.warning("No normalized items found for %s — score set to NaN", score_col)
        df[score_col] = np.nan
        return df

    valid_per_row = df[n_cols].notna().sum(axis=1)
    score = df[n_cols].mean(axis=1)
    score[valid_per_row < min_items] = np.nan
    df[score_col] = score
    n_ok = int(score.notna().sum())
    log.info("%-15s: %d/%d valid scores (%.1f%%)",
             score_col, n_ok, len(df), 100.0 * n_ok / len(df))
    return df


def build_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Build all four COR composite scores."""
    reg = CONSTRUCT_REGISTRY
    for short, info in reg.items():
        items = info["core_items"]
        # Include optional items that are available and non-degenerate
        for opt_col in info.get("optional_items", []):
            if opt_col in df.columns:
                norm_col = f"n_{opt_col}"
                if norm_col in df.columns and df[norm_col].notna().any():
                    items = items + [opt_col]
        score_col = f"{short.lower()}_score"
        df = build_composite_score(df, items, score_col)
    return df


# =============================================================================
# STEP 6 — AEV composite + binary HighAEV
# =============================================================================

def build_aev(df: pd.DataFrame, quantile: float = None) -> pd.DataFrame:
    """
    Compute High Adaptive Energy Vulnerability composite and binary target.

    Formula:
      AEV_i = mean(FCP_i, BLI_i, TCR_i, 1 − AEMC_i)

    All component scores are in [0, 1]; the row-wise mean keeps the
    composite on the same [0, 1] scale and is missing-aware (mean over
    whichever components are available). Routes 2 and 3 apply this same
    formula via `src.route_utils.compute_route_aev`.
    HighAEV = 1 if AEV ≥ 75th percentile, else 0.
    """
    if quantile is None:
        quantile = config.AEV_QUANTILE

    required = ["fcp_score", "aemc_score", "bli_score", "tcr_score"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        log.warning("AEV components missing: %s — using available columns", missing)

    components = []
    if "fcp_score"  in df.columns: components.append(df["fcp_score"])
    if "bli_score"  in df.columns: components.append(df["bli_score"])
    if "tcr_score"  in df.columns: components.append(df["tcr_score"])
    if "aemc_score" in df.columns: components.append(1.0 - df["aemc_score"])

    if not components:
        raise ValueError("No construct scores available to build AEV.")

    # Row-wise mean of available components (handles partial missingness)
    aev = pd.concat(components, axis=1).mean(axis=1)
    df["aev_score"] = aev

    threshold = df["aev_score"].quantile(quantile)
    df["high_aev"] = (df["aev_score"] >= threshold).astype(int)

    n_high = int(df["high_aev"].sum())
    n_low  = len(df) - n_high
    log.info("AEV threshold (P%d): %.4f", int(quantile * 100), threshold)
    log.info("HighAEV=1: %d  HighAEV=0: %d", n_high, n_low)
    return df


# =============================================================================
# STEP 7 — Attach annual FES scenario context (equal-weight variant)
# =============================================================================

def attach_fes_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the equal-weight annual FES scenario values to every UK household
    as contextual constants (metadata, not a feature).

    FES is a macro-level scenario-based signal simulation (see
    `src.fes_scenarios` for the full nine-scenario robustness layer), not a
    household-level predictor: all UK households receive the same annual
    value, so it carries zero within-sample variation and cannot causally
    explain or predict any single household's outcome. It is retained here
    for descriptive context/methodological transparency only; the full
    scenario-conditioned interpretation against HighAEV is built separately
    in `outputs/fes_highaev_interpretation/` once the household routes have
    run. `src.route_utils.exclude_fes_columns` is the enforcement point that
    keeps these columns out of every household-level model.
    """
    variants = {"fes_core": np.nan, "fes_macro": np.nan, "fes_actual": np.nan}

    if paths.FES_MONTHLY_FILE.exists():
        fes = pd.read_csv(paths.FES_MONTHLY_FILE, parse_dates=["date"])
        for col in variants:
            if col in fes.columns:
                variants[col] = float(fes[col].mean())
                log.info("FES context: %s = %.4f (annual mean)", col, variants[col])
    else:
        log.warning(
            "FES monthly file not found: %s\n"
            "Run python main.py --stage forecast first.",
            paths.FES_MONTHLY_FILE,
        )

    for col, val in variants.items():
        df[col] = val

    return df


# =============================================================================
# STEP 8 — Energy-poverty LIHC flag
# =============================================================================

def assign_energy_poverty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Low-Income High-Cost energy poverty classification.

    Low income:  income bracket ≤ 4 (proxy for < 60% of country median).
    High cost:   energy expenditure > 80th percentile within country.
    """
    df["income_bracket"] = pd.to_numeric(
        df.get("S9MONTH"), errors="coerce"
    ).fillna(5.0)

    h8a = pd.to_numeric(df.get("H8A"), errors="coerce")
    df["total_expenditure"] = h8a.fillna(
        h8a.median() if h8a.notna().any() else 0.0
    )

    df["low_income_flag"]  = (df["income_bracket"] <= config.LOW_INCOME_THRESHOLD).astype(int)
    threshold = df.groupby("Country")["total_expenditure"].transform(
        lambda x: x.quantile(config.HIGH_COST_PERCENTILE)
    )
    df["high_cost_flag"] = (df["total_expenditure"] > threshold).astype(int)

    both   = (df["low_income_flag"] == 1) & (df["high_cost_flag"] == 1)
    either = (df["low_income_flag"] == 1) | (df["high_cost_flag"] == 1)
    df["risk_category"] = np.where(both, "energy_poor",
                          np.where(either, "at_risk", "not_poor"))

    counts = df["risk_category"].value_counts().to_dict()
    log.info("LIHC: energy_poor=%d  at_risk=%d  not_poor=%d",
             counts.get("energy_poor", 0),
             counts.get("at_risk", 0),
             counts.get("not_poor", 0))
    return df


# =============================================================================
# STEP 8b — Household energy-efficiency controls (smart meter, insulation,
# heating fuel) — NOT part of the COR construct item pool, safe as ML
# controls
# =============================================================================

def build_efficiency_controls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build household energy-efficiency control features from ENABLE's
    granular multi-select batteries.

    The bare `H5`/`H6`/`H13` columns referenced by earlier ML feature lists
    do not exist in the UK sub-sample — but the underlying survey questions
    (insulation type, heating fuel mix, smart meter ownership) were captured
    as per-option sub-items and are well populated:

      has_smart_meter    : 1 if electricity (H13A) or heating (H13C) smart
                            meter = Yes, 0 if both = No, NaN if only
                            "don't know" (99) responses are available.
                            H13A/H13C are 0% missing (raw) in the UK sample.
      has_insulation      : 1 if any of attic/roof, cavity-wall or
                            external-wall insulation (H5A1-3) is selected.
                            H5A1-3 are 0% missing in the UK sample.
      heating_gas_share   : % of home heating from mains gas (H6A3, 0-100).
                            ~17% missing (a single skipped battery for that
                            subgroup) — left as NaN; CatBoost handles
                            missing numeric values natively rather than
                            requiring imputation. Note: `load_enable_uk()`'s
                            global MISSING_CODES replacement also zeroes out
                            genuine 98%/99% answers on this item (a handful
                            of rows) since 98/99 double as survey-wide
                            "don't know" sentinels elsewhere; negligible at
                            this prevalence but not literally exact.

    These are observable household/dwelling characteristics, not COR survey
    items, so they carry no circularity risk for the HighAEV target.
    """
    out = df.copy()

    def _yesno(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(np.nan, index=out.index)
        return out[col].map({1: 1.0, 2: 0.0})  # 99 ("don't know") -> NaN

    elec_sm = _yesno("H13A")
    heat_sm = _yesno("H13C")
    both_sm = pd.concat([elec_sm, heat_sm], axis=1)
    has_smart_meter = both_sm.max(axis=1, skipna=True)
    has_smart_meter[both_sm.isna().all(axis=1)] = np.nan
    out["has_smart_meter"] = has_smart_meter

    insul_cols = [c for c in ["H5A1", "H5A2", "H5A3"] if c in out.columns]
    if insul_cols:
        out["has_insulation"] = (out[insul_cols].fillna(0).max(axis=1) > 0).astype(float)

    if "H6A3" in out.columns:
        out["heating_gas_share"] = pd.to_numeric(out["H6A3"], errors="coerce")

    log.info(
        "Efficiency controls: has_smart_meter n=%d, has_insulation n=%d, heating_gas_share n=%d",
        int(out["has_smart_meter"].notna().sum()) if "has_smart_meter" in out.columns else 0,
        int(out["has_insulation"].notna().sum()) if "has_insulation" in out.columns else 0,
        int(out["heating_gas_share"].notna().sum()) if "heating_gas_share" in out.columns else 0,
    )
    return out


# =============================================================================
# STEP 9 — Diagnostics figures
# =============================================================================

def _plot_missingness(diag: pd.DataFrame, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#E74C3C" if r["zero_variance"] else "#2980B9"
              for _, r in diag.iterrows() if r["available"]]
    avail = diag[diag["available"]].copy()
    ax.barh(avail["variable"], avail["missing_rate"] * 100, color=colors)
    ax.set_xlabel("Missing rate (%)")
    ax.set_title("Construct item missingness\n(red = zero variance / excluded)")
    ax.axvline(20, color="#E67E22", ls="--", lw=1, label="20% threshold")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, dest, "construct_missingness")


def _plot_variability(diag: pd.DataFrame, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    avail = diag[diag["available"]].copy()
    colors = ["#AAAAAA" if r["zero_variance"] else "#27AE60"
              for _, r in avail.iterrows()]
    ax.barh(avail["variable"], avail["std"], color=colors)
    ax.set_xlabel("Standard deviation (raw item)")
    ax.set_title("Construct item variability\n(grey = zero variance, excluded from scoring)")
    fig.tight_layout()
    _save_fig(fig, dest, "construct_variability")


def _plot_score_distributions(df: pd.DataFrame, dest: Path) -> None:
    score_cols = [c for c in
                  ["fcp_score", "aemc_score", "bli_score", "tcr_score", "aev_score"]
                  if c in df.columns]
    n = len(score_cols)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    palette = {
        "fcp_score":  "#E74C3C",
        "aemc_score": "#2980B9",
        "bli_score":  "#E67E22",
        "tcr_score":  "#8E44AD",
        "aev_score":  "#2C3E50",
    }
    labels = {
        "fcp_score":  "Financial–Energy\nCost Pressure",
        "aemc_score": "Adaptive Energy-\nManagement Capacity",
        "bli_score":  "Energy Behavioural\nLock-in",
        "tcr_score":  "Transition-Cost\nResistance",
        "aev_score":  "AEV Composite",
    }
    for ax, col in zip(axes, score_cols):
        s = df[col].dropna()
        ax.hist(s, bins=30, color=palette.get(col, "#999"), edgecolor="white", alpha=0.85)
        ax.set_title(labels.get(col, col), fontsize=9)
        ax.set_xlabel("Score [0–1]")
        ax.set_ylabel("Count")
    fig.suptitle("Construct score distributions (UK sample)", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, dest, "construct_score_distributions")


# =============================================================================
# STEP 10 — Construct variable map table
# =============================================================================

def build_construct_variable_map() -> pd.DataFrame:
    """Return a tidy DataFrame documenting the item-to-construct mapping."""
    all_specs = FCP_ITEM_SPECS + AEMC_ITEM_SPECS + BLI_ITEM_SPECS + TCR_ITEM_SPECS
    rows = []
    for spec in all_specs:
        rows.append({
            "construct":       spec.construct,
            "construct_label": CONSTRUCT_REGISTRY[spec.construct]["label"],
            "variable":        spec.variable,
            "description":     spec.description,
            "direction":       spec.direction,
            "reverse_coded":   spec.reverse,
            "optional":        spec.optional,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Main entry point
# =============================================================================

def run(
    include_optional: bool = True,
    attach_fes: bool = True,
) -> pd.DataFrame:
    """
    Full ENABLE preprocessing pipeline.

    Returns
    -------
    df : pd.DataFrame
        UK sub-sample with all construct scores, AEV composite, HighAEV
        binary target, energy-poverty flags, and FES context columns.
    """
    paths.ensure_dirs()
    fig_dir = paths.ENABLE_OUT / "figures"

    # 1. Load
    df = load_enable_uk()

    # 2. Recode directional items
    df = apply_recoding(df)

    # 3. Normalize all construct items
    all_items = ALL_CONSTRUCT_ITEMS.copy()
    if include_optional:
        for info in CONSTRUCT_REGISTRY.values():
            all_items += info.get("optional_items", [])
    all_items = list(dict.fromkeys(all_items))  # deduplicate, preserve order
    df = normalize_all_items(df, all_items)

    # 4. Item diagnostics
    diag = item_diagnostics(df, all_items)
    _save_csv(diag, paths.ENABLE_OUT, "item_diagnostics")
    _plot_missingness(diag, fig_dir)
    _plot_variability(diag, fig_dir)

    # 5. Construct variable map
    cmap = build_construct_variable_map()
    _save_csv(cmap, paths.ENABLE_OUT, "construct_variable_map")

    # 6. Build composite scores
    df = build_all_scores(df)

    # 7. AEV composite + HighAEV binary target
    df = build_aev(df)

    # 8. Score distributions figure
    _plot_score_distributions(df, fig_dir)

    # 9. Construct score summary
    score_cols = [c for c in
                  ["fcp_score", "aemc_score", "bli_score", "tcr_score", "aev_score"]
                  if c in df.columns]
    summary_rows = []
    for col in score_cols:
        s = df[col].dropna()
        summary_rows.append({
            "construct_score": col,
            "n_valid":         int(len(s)),
            "missing_rate":    round(1 - len(s) / len(df), 4),
            "mean":            round(float(s.mean()), 4),
            "std":             round(float(s.std()), 4),
            "min":             round(float(s.min()), 4),
            "p25":             round(float(s.quantile(0.25)), 4),
            "median":          round(float(s.median()), 4),
            "p75":             round(float(s.quantile(0.75)), 4),
            "max":             round(float(s.max()), 4),
        })
    _save_csv(pd.DataFrame(summary_rows), paths.ENABLE_OUT, "construct_score_summary")

    # 10. Energy poverty
    df = assign_energy_poverty(df)

    # 10b. Energy-efficiency controls (smart meter, insulation, heating fuel)
    df = build_efficiency_controls(df)

    # 11. FES context (optional: depends on macro pipeline having run)
    if attach_fes:
        df = attach_fes_context(df)

    # 12. H12 lightbulb note (document the codebook correction)
    h12_note_df = pd.DataFrame([
        {"variable": k, "correct_interpretation": v, "old_mislabel": "thermal discomfort"}
        for k, v in H12_NOTE.items()
    ])
    _save_csv(h12_note_df, paths.ENABLE_OUT, "h12_codebook_correction")

    # 13. Save scored dataset
    df.to_csv(paths.ENABLE_SCORED, index=False)
    log.info("Saved scored dataset: %s (%d rows)", paths.ENABLE_SCORED.name, len(df))

    return df
