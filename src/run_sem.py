"""
run_sem.py
──────────
COR Composite-Score Path Analysis
ENABLE.EU UK Household Survey x FES 2018 Annual Context

Methodology
───────────
The anticipatory energy-carbon stress index (FES) is a UK-level, annually
aggregated macro-indicator.  Every UK household receives the same annual FES
value for each of the three variants (fes_core, fes_macro, fes_actual).

Because there is no household-level variation in FES within the UK annual
merge, FES is treated strictly as contextual macro-stress exposure -- NOT as
a varying household-level predictor.  Regression models do not include FES
as a right-hand-side variable.

The estimable behavioural mechanism is the COR pathway:

    Insecurity  ->  Resource Preservation  ->  Thermal Discomfort

This follows Conservation of Resources theory (Hobfoll 1989): perceived
resource threat (insecurity) motivates defensive adaptation (preservation),
which imposes physical consequences (discomfort).

FES contextual scenarios
─────────────────────────
  fes_core   : main forecast (core-only models)
  fes_macro  : Robustness 1 (macro-augmented models)
  fes_actual : Robustness 2 (realised 2018 prices)

These are reported as background context and compared descriptively.
They are NOT tested as household-level predictors.

Usage
─────
  python -m src.run_sem

Pipeline outputs
────────────────
  outputs/social_sem/tables/       ← 8 CSV tables
  outputs/social_sem/figures/      ← 7 figures (PNG)
  outputs/social_sem/enable_fes_cor_scored.csv
  outputs/social_sem/README_social_sem.md
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── optional SEM via semopy ───────────────────────────────────────────────────
try:
    from semopy import Model as _SemModel, calc_stats as _calc_stats
    _HAS_SEMOPY = True
except ImportError:
    _HAS_SEMOPY = False

# ==============================================================================
# CONFIG
# ==============================================================================

_ROOT = Path(__file__).parent.parent

DATA_PATH        = _ROOT / "data" / "social_science_data" / \
                   "ENABLE.EU_dataset_survey of households.xlsx"
FES_MONTHLY_PATH = _ROOT / "outputs" / "fes" / "fes_monthly_2018.csv"
OUT_DIR          = _ROOT / "outputs" / "social_sem"
FIGS_DIR         = OUT_DIR / "figures"

UK_CODE      = 11
MISSING_CODES = [9, 98, 99, 999, 9999, 99999]

# ── COR construct items (UK-sub-sample validated) ────────────────────────────
# All items confirmed present and non-null for Country == 11.
# Items listed as missing in UK (C3, C4A-C4M, C1A, C1B, C5A-C5I, C7A-C7F)
# have been excluded after inspection of the raw dataset.

INSECURITY_ITEMS = ["S8", "E2A", "E2B", "E7A", "E7B", "E7C", "E7D", "E7E"]

PRESERVATION_ITEMS = [
    "H9",
    "E5A1", "E5A2", "E5A3", "E5A4", "E5A5", "E5A9",
    "E6A1", "E6A2", "E6A3", "E6A4", "E6A5", "E6A6", "E6A7", "E6A8",
]

DISCOMFORT_ITEMS = [
    "H12A", "H12B",
    "H15A", "H15B", "H15C", "H15D", "H15E",
]

CONTROLS = ["S8", "H1", "H2", "H3"]

ITEM_ROLES = {
    "S8":   ("Insecurity",    "income difficulty",                  "higher = more insecure"),
    "E2A":  ("Insecurity",    "perceived energy cost",              "higher = more insecure"),
    "E2B":  ("Insecurity",    "perceived energy cost",              "higher = more insecure"),
    "E7A":  ("Insecurity",    "energy worry -- bills",               "higher = more insecure"),
    "E7B":  ("Insecurity",    "energy worry -- supply",              "higher = more insecure"),
    "E7C":  ("Insecurity",    "energy worry -- environment",         "higher = more insecure"),
    "E7D":  ("Insecurity",    "energy worry -- dependence",          "higher = more insecure"),
    "E7E":  ("Insecurity",    "energy worry -- access",              "higher = more insecure"),
    "H9":   ("Preservation",  "thermostat control",                 "higher = more adaptive"),
    "E5A1": ("Preservation",  "energy saving: switched off lights", "1 = yes, 0 = no"),
    "E5A2": ("Preservation",  "energy saving: reduced heating",     "1 = yes, 0 = no"),
    "E5A3": ("Preservation",  "energy saving: used less hot water", "1 = yes, 0 = no"),
    "E5A4": ("Preservation",  "energy saving: wore more clothes",   "1 = yes, 0 = no"),
    "E5A5": ("Preservation",  "energy saving: heat only some rooms","1 = yes, 0 = no"),
    "E5A9": ("Preservation",  "energy saving: other",               "1 = yes, 0 = no"),
    "E6A1": ("Preservation",  "behaviour: checked meter",           "1 = yes, 0 = no"),
    "E6A2": ("Preservation",  "behaviour: compared tariffs",        "1 = yes, 0 = no"),
    "E6A3": ("Preservation",  "behaviour: changed supplier",        "1 = yes, 0 = no"),
    "E6A4": ("Preservation",  "behaviour: sought advice",           "1 = yes, 0 = no"),
    "E6A5": ("Preservation",  "behaviour: reduced overall usage",   "1 = yes, 0 = no"),
    "E6A6": ("Preservation",  "behaviour: installed smart meter",   "1 = yes, 0 = no"),
    "E6A7": ("Preservation",  "behaviour: improved insulation",     "1 = yes, 0 = no"),
    "E6A8": ("Preservation",  "behaviour: other",                   "1 = yes, 0 = no"),
    "H12A": ("Discomfort",    "satisfaction with heating (winter)", "higher = less satisfied"),
    "H12B": ("Discomfort",    "satisfaction with heating (summer)", "higher = less satisfied"),
    "H15A": ("Discomfort",    "renovation barrier / comfort issue", "higher = more constrained"),
    "H15B": ("Discomfort",    "renovation barrier / comfort issue", "higher = more constrained"),
    "H15C": ("Discomfort",    "renovation barrier / comfort issue", "higher = more constrained"),
    "H15D": ("Discomfort",    "renovation barrier / comfort issue", "higher = more constrained"),
    "H15E": ("Discomfort",    "renovation barrier / comfort issue", "higher = more constrained"),
}

_PALETTE = {
    "insecurity":   "#E74C3C",
    "preservation": "#2980B9",
    "discomfort":   "#27AE60",
    "fes_core":     "#8E44AD",
    "fes_macro":    "#E74C3C",
    "fes_actual":   "#2C3E50",
    "grid":         "#EAECEE",
}


# ==============================================================================
# I/O helpers
# ==============================================================================

def _save_csv(df: pd.DataFrame, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tables").mkdir(exist_ok=True)
    p = OUT_DIR / "tables" / f"{name}.csv"
    df.to_csv(p, index=False)
    print(f"  [saved] {p.name}")
    return p


def _save_fig(fig: plt.Figure, name: str) -> Path:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    p = FIGS_DIR / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p.name}")
    return p


# ==============================================================================
# STEP 1 -- Load ENABLE.EU (UK only)
# ==============================================================================

def load_enable_uk() -> pd.DataFrame:
    print("\n[1] Loading ENABLE.EU dataset (UK only)...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"ENABLE.EU file not found:\n  {DATA_PATH}\n"
            "Place it at: data/social_science_data/"
            "ENABLE.EU_dataset_survey of households.xlsx"
        )
    df = pd.read_excel(DATA_PATH)
    print(f"  Raw shape: {df.shape}")

    df = df[df["Country"] == UK_CODE].copy().reset_index(drop=True)
    df["respondent_id"] = np.arange(1, len(df) + 1)
    print(f"  UK sub-sample: {len(df):,} households")

    # Replace survey missing codes with NaN
    df.replace(MISSING_CODES, np.nan, inplace=True)

    return df


# ==============================================================================
# STEP 2 -- Attach annual FES (contextual constant for all UK households)
# ==============================================================================

def attach_annual_fes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach annual FES 2018 values to every UK household.

    Annual FES is the mean of the 12 monthly FES values from the forecasting
    pipeline.  Every UK household receives the same value for each variant
    because FES is a UK-level macro indicator, not a household-level variable.

    Interview-date merging is NOT attempted:
    - T3 values are '#NULL!' for UK respondents in this dataset.
    - Even if dates were available, all UK interviews fall within 2018, so
      matching to a specific forecast month would not add identifying variation.

    FES is therefore treated as contextual annual macro-stress exposure and
    reported descriptively only -- it is NOT entered into path regressions.
    """
    print("\n[2] Attaching annual FES 2018 context...")

    fes_variants = {"fes_core": np.nan, "fes_macro": np.nan, "fes_actual": np.nan}

    if FES_MONTHLY_PATH.exists():
        fes_monthly = pd.read_csv(FES_MONTHLY_PATH, parse_dates=["date"])
        for col in fes_variants:
            if col in fes_monthly.columns:
                fes_variants[col] = float(fes_monthly[col].mean())
    else:
        print(f"  [warning] FES monthly file not found: {FES_MONTHLY_PATH}")
        print("  Run `python run_pipeline.py` first to generate FES outputs.")

    for col, val in fes_variants.items():
        df[col] = val
        if not np.isnan(val):
            print(f"  {col:12s} = {val:+.4f}  (annual mean of 12 monthly values)")
        else:
            print(f"  {col:12s} = NaN  (FES pipeline not yet run)")

    # Save contextual summary table
    ctx_rows = []
    interp = {
        "fes_core":   "Main model -- core-only forecast stress",
        "fes_macro":  "Robustness 1 -- macro-augmented forecast stress",
        "fes_actual": "Robustness 2 -- realised 2018 price benchmark",
    }
    for col, val in fes_variants.items():
        ctx_rows.append({
            "fes_variant":    col,
            "annual_mean":    round(val, 5) if not np.isnan(val) else "n/a",
            "interpretation": interp[col],
            "methodological_note": (
                "Constant for all UK households (annual UK macro indicator). "
                "Not entered as predictor in path regressions. "
                "Reported as contextual background exposure only."
            ),
        })
    _save_csv(pd.DataFrame(ctx_rows), "annual_fes_context")

    return df


# ==============================================================================
# STEP 3 -- Energy poverty (LIHC, inlined)
# ==============================================================================

def assign_energy_poverty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Low Income High Cost energy poverty flag.
    Low income:  income_bracket ≤ 4  (proxy for < 60 % of country median).
    High cost:   energy expenditure > 80th percentile within country.
    """
    print("\n[3] Assigning energy poverty (LIHC)...")

    df["income_bracket"] = pd.to_numeric(
        df.get("S9MONTH"), errors="coerce"
    ).fillna(5)

    # Energy expenditure proxy from H8A (monthly electricity bill in local units)
    h8a = pd.to_numeric(df.get("H8A"), errors="coerce")
    df["total_expenditure"] = h8a.fillna(h8a.median() if h8a.notna().any() else 0.0)

    df["low_income_flag"] = df["income_bracket"].astype(float) <= 4

    threshold = df.groupby("Country")["total_expenditure"].transform(
        lambda x: x.quantile(0.80)
    )
    df["high_cost_flag"] = df["total_expenditure"] > threshold

    both   = df["low_income_flag"] & df["high_cost_flag"]
    either = df["low_income_flag"] | df["high_cost_flag"]
    df["risk_category"] = np.where(
        both, "energy_poor", np.where(either, "at_risk", "not_poor")
    )

    counts = df["risk_category"].value_counts().to_dict()
    print(f"  energy_poor={counts.get('energy_poor',0):,}  "
          f"at_risk={counts.get('at_risk',0):,}  "
          f"not_poor={counts.get('not_poor',0):,}")
    return df


# ==============================================================================
# STEP 4 -- COR composite-score construction
# ==============================================================================

def _normalize_01(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-10:
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / (hi - lo)


def construct_cor_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build three COR latent composite scores using normalised item means.

    Method:
      1. Convert each item to numeric; replace survey missing codes (already done).
      2. Rescale each item to [0, 1] (normalize_01).
         Items with zero variance (constant after cleaning) return NaN and
         are automatically excluded from composite means.
      3. Composite score = row-mean of normalised items (min 1 non-null item).
      4. Save item diagnostics and construct map.
    """
    print("\n[4] Constructing COR composite scores...")

    all_items = {
        "Insecurity":          INSECURITY_ITEMS,
        "Resource Preservation": PRESERVATION_ITEMS,
        "Thermal Discomfort":  DISCOMFORT_ITEMS,
    }

    diag_rows = []
    map_rows  = []

    for construct, items in all_items.items():
        avail = [c for c in items if c in df.columns]
        for col in avail:
            s_raw = pd.to_numeric(df[col], errors="coerce")
            n_val   = int(s_raw.notna().sum())
            miss_rt = round(1 - n_val / len(df), 4)
            s_norm  = _normalize_01(s_raw)
            std_val = float(s_raw.std(ddof=0)) if n_val > 1 else 0.0
            zv      = bool(std_val < 1e-10)
            used    = not zv
            df[f"n_{col}"] = s_norm

            diag_rows.append({
                "construct":     construct,
                "variable":      col,
                "n_valid":       n_val,
                "missing_rate":  miss_rt,
                "min":           round(float(s_raw.min()), 4) if n_val else np.nan,
                "max":           round(float(s_raw.max()), 4) if n_val else np.nan,
                "mean":          round(float(s_raw.mean()), 4) if n_val else np.nan,
                "std":           round(std_val, 4),
                "zero_variance": zv,
                "used_in_score": used,
            })

            role_info = ITEM_ROLES.get(col, (construct, col, ""))
            map_rows.append({
                "construct":         construct,
                "variable":          col,
                "theoretical_role":  role_info[1],
                "scoring_direction": role_info[2],
                "zero_variance":     zv,
                "included":          used,
            })

    _save_csv(pd.DataFrame(diag_rows), "item_diagnostics")
    _save_csv(pd.DataFrame(map_rows),  "construct_variable_map")

    # Build composite scores
    score_map = {
        "insecurity_score":   INSECURITY_ITEMS,
        "preservation_score": PRESERVATION_ITEMS,
        "discomfort_score":   DISCOMFORT_ITEMS,
    }
    score_rows = []
    for score_col, items in score_map.items():
        norm_cols = [f"n_{c}" for c in items if f"n_{c}" in df.columns]
        if norm_cols:
            valid_per_row = df[norm_cols].notna().sum(axis=1)
            score         = df[norm_cols].mean(axis=1)
            score[valid_per_row < 1] = np.nan
        else:
            score = pd.Series(np.nan, index=df.index)
        df[score_col] = score

        n_ok  = int(df[score_col].notna().sum())
        std_v = float(df[score_col].std(ddof=0)) if n_ok > 1 else 0.0

        # Fallback: if discomfort score is degenerate, use high_cost_flag
        if score_col == "discomfort_score" and (n_ok == 0 or std_v < 1e-10):
            if "high_cost_flag" in df.columns and df["high_cost_flag"].std() > 1e-10:
                df[score_col] = df["high_cost_flag"].astype(float)
                n_ok  = int(df[score_col].notna().sum())
                std_v = float(df[score_col].std(ddof=0))
                print(f"  [Discomfort] H12/H15 items unavailable for UK -> "
                      f"using high_cost_flag as proxy")

        construct_name = {
            "insecurity_score":   "Insecurity",
            "preservation_score": "Resource Preservation",
            "discomfort_score":   "Thermal Discomfort",
        }[score_col]

        n_items_used = sum(
            1 for c in items
            if f"n_{c}" in df.columns
            and df[f"n_{c}"].notna().any()
            and df[f"n_{c}"].std() > 1e-10
        )

        print(f"  [{construct_name}] items_used={n_items_used}  "
              f"scored={n_ok:,}/{len(df):,}  "
              f"mean={df[score_col].mean():.4f}  std={std_v:.4f}")

        score_rows.append({
            "construct": construct_name,
            "n_items_used": n_items_used,
            "n_scored": n_ok,
            "mean": round(df[score_col].mean(), 4),
            "std":  round(std_v, 4),
            "min":  round(float(df[score_col].min()), 4) if n_ok else np.nan,
            "max":  round(float(df[score_col].max()), 4) if n_ok else np.nan,
        })

    _save_csv(pd.DataFrame(score_rows), "cor_score_summary")
    return df


# ==============================================================================
# STEP 5 -- COR path analysis (OLS)
# ==============================================================================

def _ols(y: pd.Series, X_df: pd.DataFrame, label: str) -> dict | None:
    """Fit OLS; return result dict with coefficients and fit stats."""
    try:
        import statsmodels.api as sm
    except ImportError:
        print("  [error] statsmodels not installed. pip install statsmodels")
        return None

    sub = pd.concat([y, X_df], axis=1).dropna()
    if len(sub) < 20:
        print(f"  [{label}] too few complete cases ({len(sub)}) -- skipped")
        return None
    if sub.iloc[:, 0].std(ddof=0) < 1e-10:
        print(f"  [{label}] outcome has zero variance -- skipped")
        return None

    y_sub = sub.iloc[:, 0]
    X_sub = sm.add_constant(sub.iloc[:, 1:].astype(float))
    res   = sm.OLS(y_sub.astype(float), X_sub).fit()
    return {"result": res, "n": int(res.nobs), "label": label}


def cor_path_analysis(df: pd.DataFrame) -> dict:
    """
    COR composite-score path model.

    M1: preservation_score ~ insecurity_score + controls
    M2: discomfort_score   ~ preservation_score + insecurity_score + controls
    M0: discomfort_score   ~ insecurity_score + controls  (total effect)

    FES is NOT included as a predictor (constant for all UK households).
    """
    print("\n[5] COR path analysis...")

    controls = [c for c in CONTROLS if c in df.columns]

    m1 = _ols(
        df["preservation_score"],
        df[["insecurity_score"] + controls],
        "M1: Insecurity -> Preservation",
    )
    m2 = _ols(
        df["discomfort_score"],
        df[["preservation_score", "insecurity_score"] + controls],
        "M2: Preservation + Insecurity -> Discomfort",
    )
    m0 = _ols(
        df["discomfort_score"],
        df[["insecurity_score"] + controls],
        "M0: Insecurity -> Discomfort (total effect)",
    )

    path_rows = []
    for m_info in [m1, m2, m0]:
        if m_info is None:
            continue
        res = m_info["result"]
        for var, coef, se, tv, pv in zip(
            res.params.index, res.params, res.bse, res.tvalues, res.pvalues
        ):
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            path_rows.append({
                "model":        m_info["label"],
                "outcome":      res.model.endog_names,
                "predictor":    var,
                "coef":         round(float(coef), 5),
                "std_error":    round(float(se),   5),
                "t_value":      round(float(tv),   4),
                "p_value":      round(float(pv),   5),
                "significance": sig,
                "n":            m_info["n"],
                "r_squared":    round(float(res.rsquared),     4),
                "adj_r_squared":round(float(res.rsquared_adj), 4),
            })
            print(f"  {m_info['label'][:40]:40s}  "
                  f"{var:28s}  b={coef:+.4f}  p={pv:.4f} {sig}")

    _save_csv(pd.DataFrame(path_rows), "path_model_estimates")
    return {"m0": m0, "m1": m1, "m2": m2}


# ==============================================================================
# STEP 6 -- Mediation analysis (bootstrap)
# ==============================================================================

def mediation_analysis(df: pd.DataFrame, path_results: dict) -> pd.DataFrame:
    """
    Indirect pathway: Insecurity -> Preservation -> Discomfort

    a  = b(insecurity) in M1
    b  = b(preservation) in M2
    c' = b(insecurity)  in M2  (direct effect)
    c  = b(insecurity)  in M0  (total effect)
    indirect = a * b
    Bootstrap CI (n=1000, seed=42) for the indirect effect.
    """
    print("\n[6] Mediation analysis (bootstrap n=1000)...")

    m0 = path_results.get("m0")
    m1 = path_results.get("m1")
    m2 = path_results.get("m2")

    if m1 is None or m2 is None:
        print("  Cannot compute mediation (M1 or M2 failed).")
        return pd.DataFrame()

    def _get_coef(m_info: dict, var: str) -> float:
        try:
            return float(m_info["result"].params[var])
        except KeyError:
            return np.nan

    a       = _get_coef(m1, "insecurity_score")
    b       = _get_coef(m2, "preservation_score")
    c_prime = _get_coef(m2, "insecurity_score")
    c       = _get_coef(m0, "insecurity_score") if m0 else np.nan

    indirect = a * b
    prop_med = indirect / c if (c and abs(c) > 1e-10) else np.nan

    # Bootstrap CI for indirect effect
    controls = [col for col in CONTROLS if col in df.columns]
    needed   = ["insecurity_score", "preservation_score", "discomfort_score"] + controls
    sub      = df[needed].dropna()

    rng      = np.random.default_rng(42)
    bs_ab    = []
    for _ in range(1000):
        idx  = rng.integers(0, len(sub), size=len(sub))
        boot = sub.iloc[idx].reset_index(drop=True)
        try:
            import statsmodels.api as sm
            X1  = sm.add_constant(boot[["insecurity_score"] + controls].astype(float))
            a_b = sm.OLS(boot["preservation_score"].astype(float), X1).fit().params.get(
                "insecurity_score", np.nan
            )
            X2  = sm.add_constant(
                boot[["preservation_score", "insecurity_score"] + controls].astype(float)
            )
            b_b = sm.OLS(boot["discomfort_score"].astype(float), X2).fit().params.get(
                "preservation_score", np.nan
            )
            bs_ab.append(a_b * b_b)
        except Exception:
            pass

    ci_lo = float(np.percentile(bs_ab, 2.5))  if bs_ab else np.nan
    ci_hi = float(np.percentile(bs_ab, 97.5)) if bs_ab else np.nan
    sig   = "yes" if (not np.isnan(ci_lo) and ci_lo * ci_hi > 0) else "no"

    rows = [{
        "pathway":            "Insecurity -> Preservation -> Discomfort",
        "a_path":             round(a,        5),
        "b_path":             round(b,        5),
        "direct_effect":      round(c_prime,  5),
        "indirect_effect":    round(indirect, 5),
        "total_effect":       round(c,        5) if not np.isnan(c) else np.nan,
        "proportion_mediated":round(prop_med, 4) if not np.isnan(prop_med) else np.nan,
        "ci_lower":           round(ci_lo,    5) if not np.isnan(ci_lo) else np.nan,
        "ci_upper":           round(ci_hi,    5) if not np.isnan(ci_hi) else np.nan,
        "significant":        sig,
    }]
    med_df = pd.DataFrame(rows)
    _save_csv(med_df, "mediation_effects")

    print(f"  a (insecurity -> preservation) = {a:+.4f}")
    print(f"  b (preservation -> discomfort) = {b:+.4f}")
    print(f"  indirect effect (axb)         = {indirect:+.4f}  "
          f"95%CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  significant={sig}")
    print(f"  direct effect (c')            = {c_prime:+.4f}")
    print(f"  total effect (c)              = {c:+.4f}")
    return med_df


# ==============================================================================
# STEP 7 -- COR mechanism validation
# ==============================================================================

def mechanism_validation(path_results: dict, med_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate whether the COR pathway is supported by the data."""
    print("\n[7] COR mechanism validation...")

    m1 = path_results.get("m1")
    m2 = path_results.get("m2")

    def _pval(m_info, var):
        try:
            return float(m_info["result"].pvalues[var])
        except (KeyError, TypeError):
            return np.nan

    p_a = _pval(m1, "insecurity_score")   # threat -> preservation
    p_b = _pval(m2, "preservation_score") # preservation -> discomfort

    indirect_sig = (
        med_df.iloc[0]["significant"] == "yes"
        if not med_df.empty else False
    )

    def _sup(p, threshold=0.05):
        if np.isnan(p):
            return "no data"
        return "supported" if p < threshold else "not supported"

    rows = [
        {
            "test": "Threat -> Preservation pathway",
            "criterion": "insecurity_score -> preservation_score, p < 0.05",
            "result": _sup(p_a),
            "interpretation": (
                "Perceived energy insecurity drives defensive resource preservation"
                if p_a < 0.05
                else "No significant insecurity-preservation association in UK sample"
            ),
        },
        {
            "test": "Preservation -> Discomfort pathway",
            "criterion": "preservation_score -> discomfort_score, p < 0.05",
            "result": _sup(p_b),
            "interpretation": (
                "Preservation behaviour significantly predicts thermal discomfort"
                if p_b < 0.05
                else "No significant preservation-discomfort association in UK sample"
            ),
        },
        {
            "test": "Mediated COR mechanism",
            "criterion": "Bootstrap 95% CI of indirect effect excludes zero",
            "result": "supported" if indirect_sig else "not supported",
            "interpretation": (
                "The insecurity -> preservation -> discomfort pathway is mediated"
                if indirect_sig
                else "Indirect pathway CI includes zero; mediation not confirmed"
            ),
        },
    ]

    validated = sum(r["result"] == "supported" for r in rows)
    overall = (
        "fully supported"    if validated == 3 else
        "partially supported" if validated >= 1 else
        "not supported"
    )
    rows.append({
        "test": "Overall COR mechanism",
        "criterion": "All three pathway tests pass",
        "result": overall,
        "interpretation": f"{validated}/3 pathway tests supported",
    })

    val_df = pd.DataFrame(rows)
    _save_csv(val_df, "cor_mechanism_validation")
    print(f"  Overall COR mechanism: {overall.upper()}")
    return val_df


# ==============================================================================
# STEP 8 -- FES contextual scenario comparison
# ==============================================================================

def fes_context_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive FES contextual scenario comparison (monthly + annual).
    FES is NOT estimated as a predictor -- reported as context only.
    """
    print("\n[8] FES contextual scenario summary...")

    rows = []
    for variant, interpretation, role in [
        ("fes_core",
         "Main anticipatory stress: core-only model (gas, electricity, carbon growth)",
         "Primary FES scenario; core energy-price forecasts only"),
        ("fes_macro",
         "Robustness 1: macro-augmented model (inflation, weather, GDP as exogenous inputs)",
         "Alternative scenario; tests if macro context alters stress signal"),
        ("fes_actual",
         "Robustness 2: realised 2018 energy prices (benchmark comparison)",
         "Ground-truth benchmark; verifies forecast direction and magnitude"),
    ]:
        val = df[variant].iloc[0] if variant in df.columns else np.nan
        rows.append({
            "variant":         variant,
            "annual_fes":      round(float(val), 5) if not np.isnan(val) else "n/a",
            "interpretation":  interpretation,
            "role_in_analysis":role,
        })

    ctx_df = pd.DataFrame(rows)
    _save_csv(ctx_df, "fes_context_summary")

    # Also load and save monthly FES for comparison
    if FES_MONTHLY_PATH.exists():
        fes_m = pd.read_csv(FES_MONTHLY_PATH, parse_dates=["date"])
        _save_csv(fes_m, "fes_monthly_reference")
        print(f"  Monthly FES reference table saved (12 rows)")

    return ctx_df


# ==============================================================================
# STEP 9 -- Save final scored dataset
# ==============================================================================

def save_final_dataset(df: pd.DataFrame) -> None:
    print("\n[9] Saving final scored dataset...")
    keep_cols = ["respondent_id", "Country"]
    keep_cols += ["fes_core", "fes_macro", "fes_actual"]
    keep_cols += ["insecurity_score", "preservation_score", "discomfort_score"]
    keep_cols += ["risk_category", "low_income_flag", "high_cost_flag"]
    keep_cols += [c for c in CONTROLS if c in df.columns]
    n_cols     = [f"n_{c}" for c in INSECURITY_ITEMS + PRESERVATION_ITEMS + DISCOMFORT_ITEMS
                  if f"n_{c}" in df.columns]
    keep_cols += n_cols

    out = df[[c for c in keep_cols if c in df.columns]].copy()
    p   = OUT_DIR / "enable_fes_cor_scored.csv"
    out.to_csv(p, index=False)
    print(f"  [saved] enable_fes_cor_scored.csv  ({len(out):,} rows x {out.shape[1]} cols)")


# ==============================================================================
# STEP 10 -- Figures
# ==============================================================================

def generate_figures(
    df: pd.DataFrame,
    path_results: dict,
    med_df: pd.DataFrame,
) -> None:
    print("\n[10] Generating figures...")

    # ── Fig 1: COR path diagram ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5.5); ax.axis("off")
    fig.patch.set_facecolor("#FDFEFE")

    # FES context box (top)
    fes_vals = {
        v: df[v].iloc[0] if v in df.columns and not np.isnan(df[v].iloc[0]) else np.nan
        for v in ["fes_core", "fes_macro", "fes_actual"]
    }
    fes_text = "\n".join(
        f"FES_{v.split('_')[1]}: {fes_vals[v]:+.3f}"
        for v in ["fes_core", "fes_macro", "fes_actual"]
        if not np.isnan(fes_vals[v])
    ) or "FES: not yet computed"

    rect_fes = mpatches.FancyBboxPatch(
        (3.5, 3.8), 7.0, 1.1, boxstyle="round,pad=0.1",
        facecolor="#D5E8D4", edgecolor="#82B366", linewidth=1.5,
    )
    ax.add_patch(rect_fes)
    ax.text(7.0, 4.35,
            f"UK Annual FES 2018  (contextual macro-stress background)\n{fes_text}",
            ha="center", va="center", fontsize=8.5, color="#2C3E50",
            fontweight="bold")

    ax.annotate("", xy=(7.0, 3.0), xytext=(7.0, 3.8),
                arrowprops=dict(arrowstyle="-|>", color="#82B366", lw=1.5,
                                linestyle="dashed"))
    ax.text(7.2, 3.4, "contextual\nbackground", fontsize=7.5,
            color="#82B366", style="italic")

    boxes = [
        ("Insecurity\n(perceived threat)",  2.0, 2.2, _PALETTE["insecurity"]),
        ("Resource Preservation\n(adaptation)", 7.0, 2.2, _PALETTE["preservation"]),
        ("Thermal Discomfort\n(consequences)", 12.0, 2.2, _PALETTE["discomfort"]),
    ]
    W, H = 2.4, 1.2
    for label, x, y, col in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - W/2, y - H/2), W, H, boxstyle="round,pad=0.1",
            facecolor=col, edgecolor="white", linewidth=1.5,
        ))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold")

    # Path arrows with coefficients
    m1 = path_results.get("m1")
    m2 = path_results.get("m2")
    a = m1["result"].params.get("insecurity_score", np.nan) if m1 else np.nan
    b = m2["result"].params.get("preservation_score", np.nan) if m2 else np.nan
    d = m2["result"].params.get("insecurity_score", np.nan) if m2 else np.nan

    for x1, x2, coef, lab in [
        (2.0 + W/2, 7.0 - W/2, a, "a"),
        (7.0 + W/2, 12.0 - W/2, b, "b"),
        (2.0 + W/2, 12.0 - W/2, d, "c'"),
    ]:
        y_arrow = 2.2 if lab != "c'" else 1.4
        y_curve = 2.2 if lab != "c'" else 1.4
        ax.annotate("", xy=(x2, y_curve), xytext=(x1, y_arrow),
                    arrowprops=dict(arrowstyle="-|>", color="#555555",
                                    connectionstyle="arc3,rad=0.2" if lab == "c'" else "arc3,rad=0",
                                    lw=2.0))
        coef_str = f"{coef:+.3f}" if not np.isnan(coef) else "?"
        mid_x = (x1 + x2) / 2
        mid_y = 2.65 if lab != "c'" else 1.1
        ax.text(mid_x, mid_y, f"{lab} = {coef_str}", ha="center",
                fontsize=9, fontweight="bold", color="#333333")

    ax.text(7.0, 0.5,
            "COR Theory (Hobfoll 1989)  |  "
            "Composite-score path analysis (OLS)  |  UK ENABLE.EU n=1,015",
            ha="center", fontsize=8, style="italic", color="#7F8C8D")
    ax.set_title(
        "COR Path Model -- Anticipatory Energy Stress Context + Behavioural Pathway",
        fontsize=13, fontweight="bold", pad=8,
    )
    _save_fig(fig, "cor_path_diagram")

    # ── Fig 2: COR score distributions ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")
    for ax, (col, label, colour) in zip(axes, [
        ("insecurity_score",   "Insecurity",          _PALETTE["insecurity"]),
        ("preservation_score", "Resource Preservation",_PALETTE["preservation"]),
        ("discomfort_score",   "Thermal Discomfort",   _PALETTE["discomfort"]),
    ]):
        ax.set_facecolor("white")
        vals = df[col].dropna()
        if vals.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(vals, bins=30, color=colour, alpha=0.75, edgecolor="white")
            ax.axvline(vals.mean(), color="#2C3E50", lw=1.8, linestyle="--",
                       label=f"μ={vals.mean():.3f}")
            ax.legend(fontsize=9)
        ax.set_title(f"{label} Score (0-1)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Score (0 = low, 1 = high)", fontsize=10)
        ax.set_ylabel("Households", fontsize=10)
        ax.grid(True, color=_PALETTE["grid"], linewidth=0.7)
    fig.suptitle("COR Composite Score Distributions -- UK Households (ENABLE.EU 2018)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_fig(fig, "cor_score_distributions")

    # ── Fig 3: Path coefficients bar chart ────────────────────────────────────
    if m1 and m2:
        coefs = {
            "Insecurity\n-> Preservation\n(a path)": a,
            "Preservation\n-> Discomfort\n(b path)": b,
            "Insecurity\n-> Discomfort\n(direct c')": d,
        }
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor("white"); fig.patch.set_facecolor("white")
        labels = list(coefs.keys())
        values = list(coefs.values())
        colours = [_PALETTE["insecurity"], _PALETTE["preservation"],
                   _PALETTE["discomfort"]]
        bars = ax.bar(labels, values, color=colours, edgecolor="white",
                      linewidth=0.8, width=0.55)
        for bar, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        v + (0.005 if v >= 0 else -0.015),
                        f"{v:+.4f}", ha="center", fontsize=10, fontweight="bold")
        ax.axhline(0, color="#95A5A6", lw=0.8)
        ax.set_title("COR Path Coefficients (OLS standardised composites)",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("OLS Coefficient", fontsize=11)
        ax.grid(axis="y", color=_PALETTE["grid"], lw=0.8)
        plt.tight_layout()
        _save_fig(fig, "cor_path_coefficients")

    # ── Fig 4: Mediation effects ───────────────────────────────────────────────
    if not med_df.empty:
        r      = med_df.iloc[0]
        ind    = float(r["indirect_effect"])   if pd.notna(r["indirect_effect"])   else 0.0
        direct = float(r["direct_effect"])     if pd.notna(r["direct_effect"])     else 0.0
        total  = float(r["total_effect"])      if pd.notna(r["total_effect"])      else 0.0
        ci_lo  = float(r["ci_lower"])          if pd.notna(r["ci_lower"])          else 0.0
        ci_hi  = float(r["ci_upper"])          if pd.notna(r["ci_upper"])          else 0.0

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor("white"); fig.patch.set_facecolor("white")
        labels = ["Indirect\n(axb)", "Direct\n(c')", "Total\n(c)"]
        values = [ind, direct, total]
        colours = ["#8E44AD", "#2980B9", "#2C3E50"]
        bars = ax.bar(labels, values, color=colours, edgecolor="white",
                      linewidth=0.8, width=0.5)
        # CI whiskers on indirect
        ax.errorbar(0, ind, yerr=[[ind - ci_lo], [ci_hi - ind]],
                    fmt="none", color="#2C3E50", capsize=8, lw=2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (0.003 if v >= 0 else -0.01),
                    f"{v:+.4f}", ha="center", fontsize=10, fontweight="bold")
        ax.axhline(0, color="#95A5A6", lw=0.8)
        ax.set_title("Mediation Effects: Insecurity -> Preservation -> Discomfort\n"
                     "(95% Bootstrap CI on indirect effect, n=1000)",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Effect size (OLS coefficient)", fontsize=11)
        ax.grid(axis="y", color=_PALETTE["grid"], lw=0.8)
        plt.tight_layout()
        _save_fig(fig, "mediation_effects")

    # ── Fig 5: FES context bar chart ──────────────────────────────────────────
    fes_plot = {
        v: df[v].iloc[0]
        for v in ["fes_core", "fes_macro", "fes_actual"]
        if v in df.columns and not np.isnan(df[v].iloc[0])
    }
    if fes_plot:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor("white"); fig.patch.set_facecolor("white")
        labels = [k.replace("fes_", "FES ").capitalize() for k in fes_plot]
        values = list(fes_plot.values())
        colours = [_PALETTE[k] for k in fes_plot]
        bars = ax.bar(labels, values, color=colours, edgecolor="white",
                      linewidth=0.8, width=0.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (0.01 if v >= 0 else -0.03),
                    f"{v:+.4f}", ha="center", fontsize=11, fontweight="bold")
        ax.axhline(0, color="#95A5A6", lw=0.8)
        ax.set_title("FES Annual Contextual Scenarios 2018 (UK)\n"
                     "(sum of z-scored components; not household-level predictors)",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("FES Annual Mean (z-score sum)", fontsize=11)
        ax.grid(axis="y", color=_PALETTE["grid"], lw=0.8)
        plt.tight_layout()
        _save_fig(fig, "fes_context_bar")

    # ── Fig 6: Construct item missingness ─────────────────────────────────────
    diag_path = OUT_DIR / "tables" / "item_diagnostics.csv"
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
        diag = diag[~diag["variable"].str.startswith("n_")]
        diag["miss_pct"] = diag["missing_rate"] * 100

        fig, ax = plt.subplots(figsize=(max(12, len(diag) * 0.45), 5))
        ax.set_facecolor("white"); fig.patch.set_facecolor("white")
        col_map = {
            "Insecurity": _PALETTE["insecurity"],
            "Resource Preservation": _PALETTE["preservation"],
            "Thermal Discomfort":    _PALETTE["discomfort"],
        }
        bar_colours = [col_map.get(c, "#95A5A6") for c in diag["construct"]]
        ax.bar(diag["variable"], diag["miss_pct"], color=bar_colours,
               edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Item", fontsize=10); ax.tick_params(axis="x", rotation=70, labelsize=7)
        ax.set_ylabel("Missing rate (%)", fontsize=10)
        ax.set_title("Construct Item Missing Rate (UK sub-sample)", fontsize=12,
                     fontweight="bold")
        legend_patches = [
            mpatches.Patch(color=v, label=k) for k, v in col_map.items()
        ]
        ax.legend(handles=legend_patches, fontsize=9)
        ax.grid(axis="y", color=_PALETTE["grid"], lw=0.7)
        plt.tight_layout()
        _save_fig(fig, "construct_item_missingness")

    # ── Fig 7: Construct item variability ─────────────────────────────────────
    if diag_path.exists():
        fig, ax = plt.subplots(figsize=(max(12, len(diag) * 0.45), 5))
        ax.set_facecolor("white"); fig.patch.set_facecolor("white")
        bar_colours2 = [
            "#AAAAAA" if zv else col_map.get(c, "#95A5A6")
            for c, zv in zip(diag["construct"], diag["zero_variance"])
        ]
        ax.bar(diag["variable"], diag["std"], color=bar_colours2,
               edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Item", fontsize=10); ax.tick_params(axis="x", rotation=70, labelsize=7)
        ax.set_ylabel("Standard Deviation", fontsize=10)
        ax.set_title("Construct Item Variability -- grey bars = zero variance (excluded)",
                     fontsize=12, fontweight="bold")
        ax.legend(handles=legend_patches + [
            mpatches.Patch(color="#AAAAAA", label="Zero variance (excluded)")
        ], fontsize=9)
        ax.grid(axis="y", color=_PALETTE["grid"], lw=0.7)
        plt.tight_layout()
        _save_fig(fig, "construct_item_variability")


# ==============================================================================
# STEP 11 -- README
# ==============================================================================

def write_readme() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    content = """\
# Social SEM Pipeline -- UK ENABLE.EU x FES 2018

## Methodology

### Why annual FES is used

The Forecasted Energy-Carbon Stress Index (FES) is computed at the UK-country
level as the annual mean of 12 monthly forecasts for 2018.  Every UK household
in the ENABLE.EU survey receives the same FES value because FES is a macro
indicator of the anticipated energy-market stress environment shared by all
UK residents.

The interview-date variable (T3) returns '#NULL!' for all UK respondents in this
dataset, making monthly matching impossible.  Even if dates were available, all
UK interviews fall within 2018, so matching to a specific forecast month would
not add identifying variation -- all households would still receive the same
annual contextual value.

### Why moderation with FES is not estimated

Because FES is constant for all UK households (zero within-country variance),
it cannot be estimated as a household-level predictor.  OLS regression would
absorb it into the intercept.  Any apparent interaction (FES x insecurity) would
be entirely non-identified within the UK annual sample.

To identify FES as a predictor, one would need either:
  (a) Cross-country variation in FES (different country-level FES values), or
  (b) Longitudinal variation (multiple survey years with different FES values).

### FES as contextual macro-stress exposure

FES is treated as the anticipated stress environment in which UK households make
energy decisions.  It defines the shared economic context of 2018, not a
property that varies across households.  This is consistent with stress exposure
research where contextual stressors affect all members of a community equally.

Three FES scenarios are reported descriptively:
  - **fes_core**  : primary forecast using core energy-price models
  - **fes_macro** : alternative forecast augmented with macroeconomic inputs
  - **fes_actual**: realised 2018 energy prices (benchmark)

### COR construct operationalisation

COR latent constructs are operationalised as composite scores:

| Construct | Items | Source |
|-----------|-------|--------|
| Insecurity | S8, E2A, E2B, E7A-E7E | Income difficulty + energy worry scale |
| Resource Preservation | H9, E5A1-E5A5, E5A9, E6A1-E6A8 | Thermostat control + energy behaviours |
| Thermal Discomfort | H12A, H12B, H15A-H15E | Heating satisfaction + comfort barriers |

Items are normalised to [0, 1] and averaged row-wise.  Items with zero variance
(all UK respondents gave the same answer) are excluded.

Note: Traditional ENABLE.EU thermal discomfort items (C1A, C1B, C5A-C5I,
C7A-C7F) are entirely missing in the UK sub-sample of this dataset.  UK-specific
substitutes (H12, H15) are used.

### Composite-score path analysis vs full latent SEM

This pipeline uses **composite-score path analysis** (OLS regression on derived
composite scores), not full latent SEM.  Full latent SEM would simultaneously
estimate measurement model parameters (factor loadings) and structural path
parameters, accounting for measurement error.  Composite-score path analysis
treats each composite as an observed variable, which underestimates standard
errors but is computationally simpler and more transparent.

The path model estimates:

  M1: preservation_score ~ insecurity_score + controls
  M2: discomfort_score ~ preservation_score + insecurity_score + controls
  M0: discomfort_score ~ insecurity_score + controls  (total effect for mediation)

### Mediation

The indirect COR pathway (Insecurity -> Preservation -> Discomfort) is quantified
as the product of path coefficients a x b, with 95% bootstrap confidence
intervals (1,000 resamples, seed 42).

## Outputs

| File | Description |
|------|-------------|
| `tables/annual_fes_context.csv` | FES annual values and methodological notes |
| `tables/item_diagnostics.csv` | Per-item validity, missingness, variance |
| `tables/construct_variable_map.csv` | Item-to-construct mapping with roles |
| `tables/cor_score_summary.csv` | Construct-level descriptive statistics |
| `tables/path_model_estimates.csv` | OLS path coefficients, SE, t, p |
| `tables/mediation_effects.csv` | Indirect effect with bootstrap CI |
| `tables/cor_mechanism_validation.csv` | COR pathway support summary |
| `tables/fes_context_summary.csv` | FES scenario descriptive comparison |
| `enable_fes_cor_scored.csv` | Final household dataset with all scores |

## Key wording

"The current UK-only ENABLE analysis does not identify household-level variation
in FES because annual FES is common to all UK respondents.  Therefore, FES is
interpreted as a contextual annual macro-stress environment, while the estimable
behavioural mechanism is the COR pathway linking insecurity, resource
preservation, and discomfort."
"""
    readme_path = OUT_DIR / "README_social_sem.md"
    readme_path.write_text(content, encoding="utf-8")
    print(f"  [saved] README_social_sem.md")


# ==============================================================================
# STEP 12 -- Final console summary
# ==============================================================================

def print_final_summary(
    df: pd.DataFrame,
    path_results: dict,
    med_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> None:
    m1 = path_results.get("m1")
    m2 = path_results.get("m2")

    def _coef(m_info, var):
        try: return float(m_info["result"].params[var])
        except: return np.nan
    def _pval(m_info, var):
        try: return float(m_info["result"].pvalues[var])
        except: return np.nan

    a  = _coef(m1, "insecurity_score")
    b  = _coef(m2, "preservation_score")
    d  = _coef(m2, "insecurity_score")
    pa = _pval(m1, "insecurity_score")
    pb = _pval(m2, "preservation_score")

    ind = float(med_df.iloc[0]["indirect_effect"]) if not med_df.empty else np.nan
    ci_lo = float(med_df.iloc[0]["ci_lower"])      if not med_df.empty else np.nan
    ci_hi = float(med_df.iloc[0]["ci_upper"])      if not med_df.empty else np.nan

    overall = val_df.iloc[-1]["result"] if not val_df.empty else "unknown"

    print(f"""
{'='*65}
  SOCIAL SEM PIPELINE -- FINAL SUMMARY
{'='*65}

  UK sample size          : {len(df):,} households
  FES context (annual)    :
    fes_core              = {df['fes_core'].iloc[0]:+.4f}  (main forecast)
    fes_macro             = {df['fes_macro'].iloc[0]:+.4f}  (Robustness 1)
    fes_actual            = {df['fes_actual'].iloc[0]:+.4f}  (Robustness 2)
  Note: FES constant for all UK households -- contextual only, not a predictor

  COR construct items used:
    Insecurity       : {sum(1 for c in INSECURITY_ITEMS if f'n_{c}' in df.columns and df[f'n_{c}'].std()>1e-10)} items
    Preservation     : {sum(1 for c in PRESERVATION_ITEMS if f'n_{c}' in df.columns and df[f'n_{c}'].std()>1e-10)} items
    Discomfort       : {sum(1 for c in DISCOMFORT_ITEMS if f'n_{c}' in df.columns and df[f'n_{c}'].std()>1e-10)} items

  Households scored:
    insecurity_score : {df['insecurity_score'].notna().sum():,}
    preservation_score:{df['preservation_score'].notna().sum():,}
    discomfort_score : {df['discomfort_score'].notna().sum():,}

  Main path coefficients (OLS):
    a (insecurity -> preservation)  = {a:+.4f}  (p={pa:.4f})
    b (preservation -> discomfort)  = {b:+.4f}  (p={pb:.4f})
    c' (direct insecurity -> disc.) = {d:+.4f}

  Mediation (bootstrap 95% CI, n=1000):
    indirect effect axb = {ind:+.4f}  [{ci_lo:+.4f}, {ci_hi:+.4f}]

  COR mechanism: {overall.upper()}

  Outputs saved to: {OUT_DIR}
{'='*65}""")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print(f"\n{'='*65}")
    print("  COR COMPOSITE-SCORE PATH ANALYSIS")
    print("  ENABLE.EU UK x FES 2018 Annual Context")
    print(f"{'='*65}")

    df = load_enable_uk()
    df = attach_annual_fes(df)
    df = assign_energy_poverty(df)
    df = construct_cor_scores(df)

    path_results = cor_path_analysis(df)
    med_df       = mediation_analysis(df, path_results)
    val_df       = mechanism_validation(path_results, med_df)
    _            = fes_context_summary(df)

    save_final_dataset(df)
    generate_figures(df, path_results, med_df)
    write_readme()

    print_final_summary(df, path_results, med_df, val_df)


if __name__ == "__main__":
    main()
