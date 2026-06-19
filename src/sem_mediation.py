"""
sem_mediation.py
────────────────
Revised COR path analysis: Financial–Energy Cost Pressure →
Adaptive Energy-Management Capacity → High Adaptive Energy Vulnerability.

Methodology note
────────────────
All path estimates are OLS-based directional associations consistent with
Conservation of Resources theory (Hobfoll 1989).  These estimates do NOT
prove causal relationships.  They test whether the data are consistent with
the theoretically specified COR-consistent directional associations.

FES context
───────────
The annual FES 2017 value is the same for every UK household (zero
within-household variation).  FES is therefore NOT entered into any
regression as a predictor.  It is reported as contextual macro-stress
background only and compared descriptively across the three variants.

Estimable COR pathways
──────────────────────
  a-path:  Financial–Energy Cost Pressure  →  Adaptive Energy-Management Capacity
  b-path:  Adaptive Energy-Management Capacity  →  AEV composite
  c'-path: Financial–Energy Cost Pressure  →  AEV composite  (direct)
  d-path:  Energy Behavioural Lock-in  →  AEV composite
  e-path:  Transition-Cost Resistance  →  AEV composite

Indirect effect (a × b) and bootstrap 95% CI are computed for the
primary FCP → AEMC → AEV mediation.

Usage
─────
  from src.sem_mediation import run
  run(df)   # df from enable_preprocessing.run()
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from src import config, paths
from src.construct_mapping import CONSTRUCT_REGISTRY

log = logging.getLogger(__name__)

# Palette consistent across all social science figures
_PALETTE = {
    "FCP":      "#E74C3C",
    "AEMC":     "#2980B9",
    "BLI":      "#E67E22",
    "TCR":      "#8E44AD",
    "aev":      "#2C3E50",
    "fes_core":   "#8E44AD",
    "fes_macro":  "#E74C3C",
    "fes_actual": "#2C3E50",
    "grid":     "#EAECEE",
    "pos":      "#27AE60",
    "neg":      "#E74C3C",
}


# =============================================================================
# I/O helpers
# =============================================================================

def _save_csv(df: pd.DataFrame, name: str) -> None:
    paths.SEM_TABLES.mkdir(parents=True, exist_ok=True)
    p = paths.SEM_TABLES / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p.name)


def _save_fig(fig: plt.Figure, name: str) -> None:
    paths.SEM_FIGURES.mkdir(parents=True, exist_ok=True)
    p = paths.SEM_FIGURES / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(p, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


# =============================================================================
# OLS path estimation
# =============================================================================

def _ols_path(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    label: str = "",
) -> dict:
    """
    Fit an OLS regression and return path coefficients, SE, t, p.

    Only rows with complete data across y and all x variables are used.
    """
    import statsmodels.api as sm

    sub = df[[y_col] + x_cols].dropna()
    if len(sub) < 10:
        log.warning("Too few observations for %s regression (%d rows)", label, len(sub))
        return {}

    X = sm.add_constant(sub[x_cols].astype(float))
    y = sub[y_col].astype(float)
    res = sm.OLS(y, X).fit()

    rows = []
    for var in x_cols:
        if var not in res.params.index:
            continue
        rows.append({
            "path":          label,
            "predictor":     var,
            "outcome":       y_col,
            "coef":          round(float(res.params[var]), 5),
            "std_err":       round(float(res.bse[var]), 5),
            "t_stat":        round(float(res.tvalues[var]), 4),
            "p_value":       round(float(res.pvalues[var]), 4),
            "r_squared":     round(float(res.rsquared), 4),
            "n":             int(len(sub)),
            "significant":   bool(res.pvalues[var] < 0.05),
        })

    log.info("[OLS %s] R²=%.3f  n=%d", label, res.rsquared, len(sub))
    return {"rows": rows, "result": res, "data": sub}


# =============================================================================
# Bootstrap mediation
# =============================================================================

def bootstrap_mediation(
    df: pd.DataFrame,
    x_col: str,
    m_col: str,
    y_col: str,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = None,
) -> dict:
    """
    Bootstrap the indirect effect (a × b) for the X → M → Y mediation.

    Uses the percentile method for the confidence interval.
    Returns indirect, direct, total effects with bootstrap CIs.
    """
    if seed is None:
        seed = config.RANDOM_SEED
    rng = np.random.default_rng(seed)
    import statsmodels.api as sm

    sub = df[[x_col, m_col, y_col]].dropna()
    n   = len(sub)
    if n < 30:
        log.warning("Bootstrap mediation: only %d complete observations", n)
        return {}

    X, M, Y = sub[x_col].values, sub[m_col].values, sub[y_col].values

    def _ab(X, M, Y):
        # a-path: X → M
        Xa = sm.add_constant(X)
        a  = sm.OLS(M, Xa).fit().params[1]
        # b-path and c': M, X → Y
        XM = sm.add_constant(np.column_stack([X, M]))
        res = sm.OLS(Y, XM).fit()
        b   = res.params[2]   # M coefficient (b-path)
        cp  = res.params[1]   # X coefficient (direct, c')
        return a, b, a * b, cp

    a_obs, b_obs, ab_obs, cp_obs = _ab(X, M, Y)

    boot_ab = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        _, _, ab_i, _ = _ab(X[idx], M[idx], Y[idx])
        boot_ab[i] = ab_i

    alpha = 1.0 - ci_level
    lo    = np.nanpercentile(boot_ab, 100 * alpha / 2)
    hi    = np.nanpercentile(boot_ab, 100 * (1 - alpha / 2))
    sig   = bool((lo > 0) or (hi < 0))

    # Total effect: X → Y (without M)
    Xc = sm.add_constant(X)
    c  = sm.OLS(Y, Xc).fit().params[1]

    return {
        "x":         x_col,
        "mediator":  m_col,
        "y":         y_col,
        "a_path":    round(float(a_obs), 5),
        "b_path":    round(float(b_obs), 5),
        "indirect":  round(float(ab_obs), 5),
        "direct":    round(float(cp_obs), 5),
        "total":     round(float(c), 5),
        "boot_lo":   round(float(lo), 5),
        "boot_hi":   round(float(hi), 5),
        "n_boot":    n_boot,
        "ci_level":  ci_level,
        "n_obs":     n,
        "sig_mediation": sig,
    }


# =============================================================================
# Figures
# =============================================================================

def _plot_path_diagram(path_rows: list[dict]) -> None:
    """
    Draw the COR path diagram with OLS path coefficients annotated.

    Layout (simplified):
      FCP ──a──► AEMC ──b──► AEV
       │                      ▲
       └──────── c' ──────────┘
      BLI ───── d ────────────►
      TCR ───── e ────────────►
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = {
        "FCP":  (1.2, 4.5, "Financial–Energy\nCost Pressure\n(FCP)"),
        "AEMC": (4.5, 4.5, "Adaptive Energy-\nManagement Capacity\n(AEMC)"),
        "AEV":  (8.0, 3.0, "High Adaptive\nEnergy Vulnerability\n(AEV)"),
        "BLI":  (1.2, 2.0, "Behavioural\nLock-in\n(BLI)"),
        "TCR":  (1.2, 0.8, "Transition-Cost\nResistance\n(TCR)"),
    }
    box_kw = dict(ha="center", va="center", fontsize=8)
    for key, (x, y, label) in boxes.items():
        ax.text(x, y, label, **box_kw,
                bbox=dict(boxstyle="round,pad=0.4",
                          ec=_PALETTE.get(key, "grey"),
                          fc="#F9F9F9", lw=1.8))

    # Annotate path coefficients
    coef_map = {r["predictor"]: r for r in path_rows if isinstance(r, dict)}

    def _arrow(x1, y1, x2, y2, label, color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.15
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=8,
                color=color, fontweight="bold")

    def _coef_label(pred, fallback=""):
        row = coef_map.get(pred)
        if row:
            p = row.get("p_value", 1)
            sig = "*" if p < 0.05 else ""
            return f"β={row.get('coef', '?'):.3f}{sig}"
        return fallback

    _arrow(2.4, 4.5, 3.3, 4.5, _coef_label("fcp_score", "a-path"), _PALETTE["FCP"])
    _arrow(5.7, 4.5, 7.0, 3.4, _coef_label("aemc_score", "b-path"), _PALETTE["AEMC"])
    _arrow(2.4, 4.2, 7.0, 3.0, _coef_label("fcp_score", "c'-path"), "#888")
    _arrow(2.4, 2.0, 7.0, 2.8, _coef_label("bli_score", "d-path"), _PALETTE["BLI"])
    _arrow(2.4, 0.9, 7.0, 2.6, _coef_label("tcr_score", "e-path"), _PALETTE["TCR"])

    ax.set_title("COR-consistent path diagram: OLS directional associations",
                 fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    _save_fig(fig, "cor_path_diagram")


def _plot_path_coefficients(path_rows: list[dict]) -> None:
    """Horizontal bar chart of path coefficients with significance markers."""
    if not path_rows:
        return
    df_p = pd.DataFrame(path_rows)
    if df_p.empty or "coef" not in df_p.columns:
        return

    df_p = df_p.sort_values("coef")
    colors = [_PALETTE["pos"] if c >= 0 else _PALETTE["neg"]
              for c in df_p["coef"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df_p["predictor"], df_p["coef"], color=colors, alpha=0.85)
    for bar, sig in zip(bars, df_p.get("significant", [False] * len(df_p))):
        if sig:
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    "*", va="center", fontsize=12, color="black")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("OLS coefficient (β)")
    ax.set_title("COR path coefficients (* p < 0.05)")
    fig.tight_layout()
    _save_fig(fig, "cor_path_coefficients")


def _plot_mediation(med: dict) -> None:
    if not med:
        return
    labels  = ["Indirect (a×b)", "Direct (c')", "Total (c)"]
    vals    = [med.get("indirect"), med.get("direct"), med.get("total")]
    lo_err  = [med.get("indirect", 0) - med.get("boot_lo", 0), 0, 0]
    hi_err  = [med.get("boot_hi", 0) - med.get("indirect", 0), 0, 0]
    colors  = [_PALETTE["FCP"], _PALETTE["AEMC"], _PALETTE["aev"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = [2, 1, 0]
    ax.barh(y_pos, vals, xerr=[lo_err, hi_err], color=colors,
            alpha=0.85, error_kw=dict(ecolor="black", capsize=4))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Effect size (OLS β)")
    ci_pct = int(med.get("ci_level", 0.95) * 100)
    ax.set_title(
        f"FCP → AEMC → AEV mediation\n"
        f"Indirect {ci_pct}% CI: [{med.get('boot_lo', '?'):.4f}, {med.get('boot_hi', '?'):.4f}]"
    )
    sig_label = "Significant mediation" if med.get("sig_mediation") else "Non-significant mediation"
    ax.text(0.98, 0.02, sig_label, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            color=_PALETTE["pos"] if med.get("sig_mediation") else _PALETTE["neg"])
    fig.tight_layout()
    _save_fig(fig, "mediation_effects")


def _plot_fes_context(fes_summary: pd.DataFrame) -> None:
    if fes_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    x  = np.arange(len(fes_summary))
    bars = ax.bar(fes_summary["fes_variant"], fes_summary["annual_mean"].astype(float),
                  color=[_PALETTE.get(v, "#999") for v in fes_summary["fes_variant"]],
                  alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("FES annual mean (z-score units)")
    ax.set_title(
        "FES contextual background: three scenarios (2017)\n"
        "(constant for all UK households — NOT a household predictor)"
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, "fes_context_bar")


# =============================================================================
# Main entry point
# =============================================================================

def run(df: pd.DataFrame) -> None:
    """
    Estimate COR-consistent path analysis for the revised construct framework.

    Paths estimated:
      a:  FCP  → AEMC
      b:  AEMC → AEV_score
      c': FCP  → AEV_score  (direct)
      d:  BLI  → AEV_score
      e:  TCR  → AEV_score
    Plus bootstrap mediation for FCP → AEMC → AEV_score.
    """
    paths.SEM_TABLES.mkdir(parents=True, exist_ok=True)
    paths.SEM_FIGURES.mkdir(parents=True, exist_ok=True)

    # ── Map score column names ────────────────────────────────────────────────
    score_cols = {
        "fcp_score":  "FCP",
        "aemc_score": "AEMC",
        "bli_score":  "BLI",
        "tcr_score":  "TCR",
        "aev_score":  "AEV",
    }
    available = [c for c in score_cols if c in df.columns]
    log.info("Available scores: %s", available)

    # ── Path 1: FCP → AEMC (a-path) ─────────────────────────────────────────
    all_path_rows = []

    if "fcp_score" in df.columns and "aemc_score" in df.columns:
        r = _ols_path(df, "aemc_score", ["fcp_score"], label="a: FCP→AEMC")
        all_path_rows.extend(r.get("rows", []))

    # ── Path 2: AEMC, FCP, BLI, TCR → AEV (b, c', d, e paths) ──────────────
    aev_predictors = [c for c in
                      ["fcp_score", "aemc_score", "bli_score", "tcr_score"]
                      if c in df.columns]
    if "aev_score" in df.columns and aev_predictors:
        r2 = _ols_path(df, "aev_score", aev_predictors, label="b/c'/d/e: → AEV")
        all_path_rows.extend(r2.get("rows", []))

    path_df = pd.DataFrame(all_path_rows)
    _save_csv(path_df, "sem_path_estimates")

    # ── Bootstrap mediation: FCP → AEMC → AEV ────────────────────────────────
    med_result = {}
    if all(c in df.columns for c in ["fcp_score", "aemc_score", "aev_score"]):
        log.info("Running bootstrap mediation (n_boot=%d)...", 2000)
        med_result = bootstrap_mediation(
            df, "fcp_score", "aemc_score", "aev_score",
            n_boot=2000, ci_level=0.95,
        )
        if med_result:
            _save_csv(pd.DataFrame([med_result]), "mediation_effects")
            log.info(
                "Mediation: indirect=%.4f  [%.4f, %.4f]  sig=%s",
                med_result.get("indirect", np.nan),
                med_result.get("boot_lo", np.nan),
                med_result.get("boot_hi", np.nan),
                med_result.get("sig_mediation"),
            )

    # ── COR mechanism validation summary ─────────────────────────────────────
    validation_rows = []
    path_map = {r["predictor"]: r for r in all_path_rows if isinstance(r, dict)}

    def _check(name, pred, outcome, expected_sign, label):
        row = path_map.get(pred, {})
        coef = row.get("coef", np.nan)
        p    = row.get("p_value", np.nan)
        supported = (
            not np.isnan(coef) and not np.isnan(p) and
            p < 0.05 and
            ((expected_sign > 0 and coef > 0) or (expected_sign < 0 and coef < 0))
        )
        validation_rows.append({
            "pathway": name,
            "predictor": pred,
            "outcome": outcome,
            "expected_direction": "positive" if expected_sign > 0 else "negative",
            "coef": round(float(coef), 5) if not np.isnan(coef) else np.nan,
            "p_value": round(float(p), 4) if not np.isnan(p) else np.nan,
            "significant": bool(p < 0.05) if not np.isnan(p) else None,
            "direction_consistent": supported,
            "note": label,
        })

    _check("a-path", "fcp_score", "aemc_score", -1,
           "Higher financial pressure → lower adaptive capacity (COR: threat depletes resources)")
    _check("b-path", "aemc_score", "aev_score", -1,
           "Higher adaptive capacity → lower AEV (protective effect)")
    _check("c'-path", "fcp_score", "aev_score", +1,
           "Higher financial pressure → higher AEV (direct stress pathway)")
    _check("d-path", "bli_score", "aev_score", +1,
           "Higher lock-in → higher AEV (habitual constraint)")
    _check("e-path", "tcr_score", "aev_score", +1,
           "Higher transition resistance → higher AEV (cost-sensitivity barrier)")

    _save_csv(pd.DataFrame(validation_rows), "cor_mechanism_validation")

    # ── FES contextual context table ──────────────────────────────────────────
    fes_cols = [c for c in ["fes_core", "fes_macro", "fes_actual"] if c in df.columns]
    fes_rows = []
    for col in fes_cols:
        val = df[col].dropna().mean() if col in df.columns else np.nan
        fes_rows.append({
            "fes_variant": col,
            "annual_mean": round(float(val), 5) if not np.isnan(val) else np.nan,
            "interpretation": {
                "fes_core":   "Main forecast — core-only model stress",
                "fes_macro":  "Robustness 1 — macro-augmented forecast stress",
                "fes_actual": "Robustness 2 — realised 2017 price benchmark",
            }.get(col, ""),
            "methodological_note": (
                "Constant for all UK households (UK-level macro indicator). "
                "NOT entered as a predictor in any path regression. "
                "Reported as contextual macro-stress background only."
            ),
        })
    fes_summary = pd.DataFrame(fes_rows)
    _save_csv(fes_summary, "fes_context_summary")

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_path_diagram(all_path_rows)
    _plot_path_coefficients(all_path_rows)
    if med_result:
        _plot_mediation(med_result)
    if not fes_summary.empty:
        _plot_fes_context(fes_summary)

    log.info("SEM mediation stream complete.")
