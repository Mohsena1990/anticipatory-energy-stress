"""
ukhls_forward_prediction.py
────────────────────────────
Stage 5 — Forward Vulnerability Prediction.

Every earlier stage answers "who is vulnerable now, and what explains it" --
a CONTEMPORANEOUS question: Stage 2b/3 explain a household's OWN interview-
year `fuel_to_income_ratio` using `fes_delta` (the anticipated next-year
shock, forecast using only data through that year) as a moderator.

This stage answers a genuinely different, PROSPECTIVE question: "who is
about to become vulnerable, and roughly when" -- using a household's
CURRENT (wave t) COR-SEM/COR-CVAE profile plus the signal already forecast
for their own next year (`fes_magnitude`, already attached by
`src.ukhls_preprocessing.attach_fes_delta`), predict whether THAT SAME
household will be vulnerable at wave t+1 -- before wave t+1's data exists.

Mechanism
─────────
UKHLS's per-wave rows carry no stable cross-wave household key on their own
(`hidp` is reissued whenever household composition changes), so the first
requirement is a genuine person-level anchor: `hrpid` (household reference
person's `pidp`), added to the panel by `src.ukhls_mapping.HH_LINK_VARS`.
Matching `hrpid_t == hrpid_{t+1}` directly at the household level links the
same reference person's household across one wave gap -- verified against
the raw wave a/b hhresp files: 72.5% of households link (21,886/30,169), a
normal UKHLS wave-to-wave attrition/HRP-turnover rate, not a bug.

For every consecutive wave pair (a->b, b->c, ..., n->o -- 14 pairs spanning
2009-2024), each linked household contributes one training example:
  features (wave t)   : COR-SEM factor scores, fes_magnitude (t's own
                         forecast for t+1), plus a few plain controls
  label    (wave t+1)  : high_fuel_vulnerable / fuel_to_income_ratio

A logistic regression (transparent odds ratios, matching Stage 3's own
established choice to drop a black-box model "on request") is walk-forward
validated -- trained on the earliest 12 transitions (a->b ... l->m), held
out on the 2 most recent KNOWN transitions (m->n, n->o) -- then refit on
ALL 14 known transitions and applied to the most recent wave (o) using each
household's own profile + their own already-forecast `fes_magnitude`, to
produce a genuinely forward, currently-unobserved prediction for their own
next interview year (2024 or 2025, depending on exactly when within wave
o's fielding window they were interviewed).

Caveats (documented, not hidden -- same practice as every other stage)
────────────────────────────────────────────────────────────────────────
  - Only the ~70-75% of households that link wave-to-wave via hrpid are
    used, both for training and for the final target list -- attrition
    itself may correlate with vulnerability, and is not corrected for here.
  - The COR-SEM scores used as features are fit on the WHOLE pooled
    2009-2024 panel (already a documented Stage 2b limitation), so
    they carry a mild amount of whole-panel information into every wave's
    "as of time t" feature -- a second-order effect on the features, not a
    leak of the actual t+1 label itself, but worth restating here because
    this stage's validity claim ("genuinely forward") is narrower than
    Stage 2b/3's.
  - hrpid-continuity tracks the reference PERSON, not a fixed dwelling --
    the same person heading a different household still counts as linked.

Usage
─────
  from src.ukhls_forward_prediction import run
  result = run(df, sem_scores)   # df from ukhls_preprocessing.run()
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src import config, paths
from src.logging_utils import get_logger
from src.ukhls_mapping import WAVE_LETTERS, GOR_LABELS
from src.ukhls_geo_maps import plot_choropleth

log = get_logger("ukhls_forward_prediction")


# Deliberately excludes baseline_resource_score: it's a loading-weighted
# composite OF these four (see src.ukhls_cor_sem.compute_baseline_score),
# correlated with each at r=0.65-0.77 in this panel -- including both the
# aggregate and its own components in the SAME logistic regression produces
# severe multicollinearity (verified empirically: including it swung
# object/condition/personal/energy_score to odds ratios of 15-28 and
# baseline_resource_score itself to 0.0001, compensating artifacts, not a
# real effect). The 4 first-order scores alone give genuinely interpretable,
# non-redundant per-dimension odds ratios -- more actionable for policy
# ("which resource type") than one aggregate number would be anyway.
_SEM_COLS  = ["object_score", "condition_score", "personal_score", "energy_score"]
# Deliberately excludes the COR-CVAE latents (cvae_object_z etc.): they're
# trained via an explicit alignment loss (src.ukhls_cor_cvae) to match these
# same four SEM scores, and are correlated with their SEM counterpart at
# r=0.65-0.83 in this panel -- putting both the SEM score and its aligned
# CVAE latent in the SAME logistic regression is the identical
# multicollinearity failure mode as the baseline_resource_score case above
# (verified empirically: with both included, several coefficients swung
# sign/magnitude and previously-significant predictors became
# non-significant). The SEM scores alone are kept as the interpretable
# per-dimension features.
# Plain controls not already summarized by the SEM factors themselves
# (dvage/heatch were deliberately excluded from the SEM measurement model --
# see ukhls_mapping.py's COR_FACTOR_ITEMS comments -- so they still carry
# independent information here). lone_parent/large_family/workless_household
# (src.ukhls_preprocessing) are new household-composition/employment
# controls, same "plain control, not folded into the SEM" status --
# demographic/labour-market flags, not reflective indicators of the
# underlying resource construct.
_EXTRA_CONTROL_COLS = [
    "financial_strain_score", "dvage", "heatch",
    "lone_parent", "large_family", "workless_household",
]
_SIGNAL_COL = "fes_magnitude"

# Most recent N wave-transitions held out for walk-forward validation
# (trained on everything earlier) -- consistent with Stage 1's own
# walk-forward philosophy, not a random split.
N_VALIDATION_PAIRS = 2

_PALETTE = {"low": "#27AE60", "mid": "#E67E22", "high": "#E74C3C", "grid": "#EAECEE"}


# =============================================================================
# I/O helpers
# =============================================================================

def _save_csv(df: pd.DataFrame, name: str) -> None:
    paths.UKHLS_FORWARD_TABLES.mkdir(parents=True, exist_ok=True)
    p = paths.UKHLS_FORWARD_TABLES / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p.name)


def _save_fig(fig: plt.Figure, name: str) -> None:
    paths.UKHLS_FORWARD_FIGURES.mkdir(parents=True, exist_ok=True)
    p = paths.UKHLS_FORWARD_FIGURES / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(p, dpi=config.DPI, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


def _feature_cols(work: pd.DataFrame) -> list[str]:
    return [c for c in _SEM_COLS + _EXTRA_CONTROL_COLS + [_SIGNAL_COL]
            if c in work.columns]


# =============================================================================
# STEP 1 — Build hrpid-linked wave-to-wave transition pairs
# =============================================================================

def build_transition_pairs(work: pd.DataFrame) -> pd.DataFrame:
    """
    For each consecutive UKHLS wave pair, link households via hrpid and
    build one row per matched pair: features from wave t, label from wave
    t+1. See module docstring for the linkage rationale/verified match rate.
    """
    if "hrpid" not in work.columns:
        log.error("hrpid not found in panel -- cannot build transition pairs. "
                   "Check src.ukhls_mapping.HH_LINK_VARS is wired into "
                   "ukhls_preprocessing.load_wave_hhresp.")
        return pd.DataFrame()

    feature_cols = _feature_cols(work)
    t1_keep = ["hrpid", "high_fuel_vulnerable", "fuel_to_income_ratio",
               "interview_year", "interview_month"]
    t1_keep = [c for c in t1_keep if c in work.columns]
    t1_rename = {
        "high_fuel_vulnerable": "high_fuel_vulnerable_t1",
        "fuel_to_income_ratio": "fuel_to_income_ratio_t1",
        "interview_year": "interview_year_t1",
        "interview_month": "interview_month_t1",
    }

    pairs = list(zip(WAVE_LETTERS[:-1], WAVE_LETTERS[1:]))
    frames: list[pd.DataFrame] = []
    for wave_t, wave_t1 in pairs:
        df_t = (
            work[work["wave"] == wave_t]
            .dropna(subset=["hrpid"])
            .drop_duplicates(subset="hrpid")
        )
        df_t1 = (
            work[work["wave"] == wave_t1][t1_keep]
            .dropna(subset=["hrpid"])
            .drop_duplicates(subset="hrpid")
            .rename(columns=t1_rename)
        )
        if df_t.empty or df_t1.empty:
            continue

        merged = df_t.merge(df_t1, on="hrpid", how="inner")
        if merged.empty:
            continue

        row = pd.DataFrame({
            "as_of_wave": wave_t,
            "target_wave": wave_t1,
            "hrpid": merged["hrpid"].values,
            "gor_dv": merged.get("gor_dv"),
            "interview_year_t": merged["interview_year"].values,
            "interview_month_t": merged["interview_month"].values,
            "interview_year_t1": merged["interview_year_t1"].values,
            "high_fuel_vulnerable_t1": merged["high_fuel_vulnerable_t1"].values,
            "fuel_to_income_ratio_t1": merged["fuel_to_income_ratio_t1"].values,
        })
        for c in feature_cols:
            row[c] = merged[c].values
        frames.append(row)

        log.info("Transition %s->%s: %d/%d wave-%s households linked via hrpid (%.1f%%)",
                  wave_t, wave_t1, len(merged), len(df_t), wave_t,
                  100 * len(merged) / len(df_t) if len(df_t) else 0.0)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# STEP 2 — Logistic model (transparent odds ratios, matches Stage 3's choice)
# =============================================================================

def _fit_logit(transition_df: pd.DataFrame, feature_cols: list[str],
               label_col: str = "high_fuel_vulnerable_t1", min_n: int = 30):
    import statsmodels.api as sm

    sub = transition_df[feature_cols + [label_col]].dropna()
    if len(sub) < min_n:
        log.warning("Stage 5: too few rows (%d) to fit the forward model", len(sub))
        return None
    X = sm.add_constant(sub[feature_cols].astype(float))
    y = sub[label_col].astype(float)
    try:
        return sm.Logit(y, X).fit(disp=0, maxiter=200)
    except Exception as e:
        log.error("Stage 5: logistic fit failed: %s", e)
        return None


def _predict_proba(fit_result, df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    import statsmodels.api as sm

    X = sm.add_constant(df[feature_cols].astype(float), has_constant="add")
    X = X.reindex(columns=fit_result.params.index)
    return pd.Series(fit_result.predict(X), index=df.index)


def _coef_table(fit_result, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for var in feature_cols:
        if var not in fit_result.params.index:
            continue
        coef = fit_result.params[var]
        rows.append({
            "predictor": var,
            "coef": round(float(coef), 5),
            "odds_ratio": round(float(np.exp(coef)), 4),
            "p_value": round(float(fit_result.pvalues[var]), 4),
            "significant": bool(fit_result.pvalues[var] < 0.05),
        })
    return pd.DataFrame(rows).sort_values("odds_ratio", ascending=False)


# =============================================================================
# STEP 3 — Walk-forward validation (train on earlier waves, hold out latest)
# =============================================================================

def run_walk_forward_validation(transition_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Train on the earliest transitions, validate on the N_VALIDATION_PAIRS
    most recent KNOWN transitions -- a temporal holdout, not a random split,
    consistent with Stage 1's own walk-forward philosophy. This is the
    number that says whether the final (all-data) model's 2025 predictions
    should be trusted at all.
    """
    from sklearn.metrics import roc_auc_score

    pairs = list(zip(WAVE_LETTERS[:-1], WAVE_LETTERS[1:]))
    val_pairs = set(pairs[-N_VALIDATION_PAIRS:])
    pair_col = pd.Series(list(zip(transition_df["as_of_wave"], transition_df["target_wave"])),
                          index=transition_df.index)
    is_val = pair_col.isin(val_pairs)

    train_df = transition_df[~is_val]
    val_df   = transition_df[is_val]

    fit = _fit_logit(train_df, feature_cols)
    if fit is None:
        log.warning("Stage 5: walk-forward validation skipped -- training fit failed.")
        return {}

    val_sub = val_df[feature_cols + ["high_fuel_vulnerable_t1", "fuel_to_income_ratio_t1"]].dropna(
        subset=feature_cols + ["high_fuel_vulnerable_t1"]
    )
    if val_sub.empty:
        log.warning("Stage 5: walk-forward validation skipped -- no complete held-out rows.")
        return {}

    proba = _predict_proba(fit, val_sub, feature_cols)
    y_true = val_sub["high_fuel_vulnerable_t1"].astype(float)
    auc = roc_auc_score(y_true, proba) if y_true.nunique() > 1 else np.nan
    pearson_r = proba.corr(val_sub["fuel_to_income_ratio_t1"])

    val_pair_labels = ", ".join(f"{a}->{b}" for a, b in sorted(val_pairs))
    metrics_df = pd.DataFrame([{
        "validation_pairs": val_pair_labels,
        "n_train": len(train_df.dropna(subset=feature_cols + ["high_fuel_vulnerable_t1"])),
        "n_validation": len(val_sub),
        "auc_vs_high_fuel_vulnerable_t1": round(float(auc), 4) if pd.notna(auc) else np.nan,
        "pearson_r_vs_fuel_to_income_ratio_t1": round(float(pearson_r), 4) if pd.notna(pearson_r) else np.nan,
    }])
    _save_csv(metrics_df, "stage5_validation_metrics")
    log.info(
        "Stage 5 walk-forward validation (held out %s): AUC=%s, Pearson r=%s, n_train=%d, n_val=%d",
        val_pair_labels,
        f"{auc:.4f}" if pd.notna(auc) else "NA",
        f"{pearson_r:.4f}" if pd.notna(pearson_r) else "NA",
        metrics_df["n_train"].iloc[0], len(val_sub),
    )

    _plot_validation_roc(y_true, proba, auc)

    return {"fit": fit, "metrics": metrics_df, "proba": proba, "y_true": y_true}


def _plot_validation_roc(y_true: pd.Series, proba: pd.Series, auc: float) -> None:
    from sklearn.metrics import roc_curve

    if y_true.nunique() < 2 or pd.isna(auc):
        log.warning("Stage 5: ROC figure skipped (degenerate validation labels)")
        return
    fpr, tpr, _ = roc_curve(y_true, proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.plot(fpr, tpr, color=_PALETTE["high"], linewidth=2.2, label=f"Stage 5 model (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#BDC3C7", linestyle="--", linewidth=1.0, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Stage 5 Walk-Forward Validation\n"
                 "(held-out wave transitions -- genuinely forward-predicted, not fit)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, color=_PALETTE["grid"], linewidth=0.8)
    fig.tight_layout()
    _save_fig(fig, "stage5_validation_roc")


# =============================================================================
# STEP 4 — Refit on all known transitions, score the latest wave
# =============================================================================

def predict_latest_wave(work: pd.DataFrame, transition_df: pd.DataFrame,
                         feature_cols: list[str]) -> pd.DataFrame:
    """
    Refit on every known transition (a->b ... n->o), then score every
    household in the most recent wave using their OWN profile + their OWN
    already-forecast fes_magnitude for next year -- genuinely unobserved,
    the actual early-warning output ("2025" for households interviewed in
    2024; "2024" for those interviewed earlier within wave o's fielding
    window).
    """
    fit = _fit_logit(transition_df, feature_cols)
    if fit is None:
        log.warning("Stage 5: final production model could not be fit.")
        return pd.DataFrame()

    _save_csv(_coef_table(fit, feature_cols), "stage5_driver_coefficients")

    latest_wave = WAVE_LETTERS[-1]
    latest = work[work["wave"] == latest_wave].copy()
    scoreable = latest.dropna(subset=feature_cols)
    if scoreable.empty:
        log.warning("Stage 5: no scoreable households in wave %s (all missing >=1 feature)", latest_wave)
        return pd.DataFrame()

    proba = _predict_proba(fit, scoreable, feature_cols)
    out = pd.DataFrame({
        "hrpid": scoreable["hrpid"].values,
        "wave": latest_wave,
        "gor_dv": scoreable.get("gor_dv"),
        "region": scoreable["gor_dv"].map(GOR_LABELS) if "gor_dv" in scoreable.columns else np.nan,
        "interview_year": scoreable["interview_year"].values,
        "interview_month": scoreable["interview_month"].values,
        "predicted_target_year": scoreable["interview_year"].values + 1,
        "fes_magnitude_used": scoreable.get(_SIGNAL_COL),
        "predicted_vulnerable_probability": proba.round(4).values,
    })
    _save_csv(out, "stage5_forward_predictions")
    log.info(
        "Stage 5: scored %d/%d wave-%s households for their own next interview year "
        "(mean predicted probability=%.4f, %d could not be scored -- missing features)",
        len(out), len(latest), latest_wave,
        out["predicted_vulnerable_probability"].mean(), len(latest) - len(out),
    )
    return out


# =============================================================================
# Figures — policy application of the forward predictions
# =============================================================================

def plot_forward_prediction_map(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Real UK region map of mean predicted forward-vulnerability
    probability -- where should next-year intervention be prioritised."""
    if pred_df.empty or "region" not in pred_df.columns:
        return pd.DataFrame()
    by_region = pred_df.groupby("region", dropna=True).agg(
        mean_predicted_probability=("predicted_vulnerable_probability", "mean"),
        n=("predicted_vulnerable_probability", "size"),
    ).reset_index()
    # Plotted/labelled as a percentage rather than the raw 0-1 probability --
    # values cluster tightly (~0.06-0.08), so a 2-decimal fraction rounds
    # nearly every region to the same "0.07" and hides real cross-region
    # variation that's visible once expressed as e.g. 6.2% vs 7.6%.
    by_region["mean_predicted_probability_pct"] = by_region["mean_predicted_probability"] * 100
    plot_choropleth(
        by_region, value_col="mean_predicted_probability_pct",
        title="Predicted Forward Vulnerability by UK Region\n"
              "(mean predicted probability of becoming high fuel-to-income "
              "vulnerable, households' own next interview year)",
        out_name="stage5_forward_prediction_map",
        tables_dir=paths.UKHLS_FORWARD_TABLES, figures_dir=paths.UKHLS_FORWARD_FIGURES,
        cmap="YlOrRd", cbar_label="Predicted probability (%)", fmt="{:.1f}%",
    )
    return by_region


def plot_forward_prediction_by_month(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Which target month carries the highest predicted risk -- directly
    answers 'which month should an intervention target'. interview_month
    here doubles as the target month: FES Magnitude (and so this
    prediction) matches interview month to the SAME calendar month one
    year ahead (documented project-wide approximation, see README)."""
    valid = pred_df.dropna(subset=["interview_month", "predicted_vulnerable_probability"]).copy()
    if valid.empty:
        return pd.DataFrame()
    valid["interview_month"] = valid["interview_month"].astype(int)
    by_month = valid.groupby("interview_month").agg(
        mean_predicted_probability=("predicted_vulnerable_probability", "mean"),
        n=("predicted_vulnerable_probability", "size"),
    ).reset_index().sort_values("interview_month")
    _save_csv(by_month, "stage5_forward_prediction_by_month")

    # Plotted as a percentage -- same reasoning as plot_forward_prediction_map:
    # values cluster tightly enough that a 2-decimal fraction rounds most
    # months to the same figure and hides real month-to-month variation.
    by_month["mean_predicted_probability_pct"] = by_month["mean_predicted_probability"] * 100

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    med = by_month["mean_predicted_probability_pct"].median()
    colors = [_PALETTE["high"] if v >= med else _PALETTE["low"]
              for v in by_month["mean_predicted_probability_pct"]]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    labels = [months[m - 1] for m in by_month["interview_month"]]
    ax.bar(labels, by_month["mean_predicted_probability_pct"], color=colors, alpha=0.85)
    for i, r in by_month.iterrows():
        ax.text(labels[list(by_month.index).index(i)], r["mean_predicted_probability_pct"] + 0.05,
                 f"{r['mean_predicted_probability_pct']:.1f}%", ha="center", fontsize=8)
    ax.set_ylabel("Mean predicted probability (%)")
    ax.set_xlabel("Target month (same calendar month, one year ahead of interview)")
    ax.set_title("Which Month Carries the Highest Predicted Risk\n"
                 "(mean forward-predicted vulnerability probability, by target month)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", color=_PALETTE["grid"])
    fig.tight_layout()
    _save_fig(fig, "stage5_forward_prediction_by_month")
    return by_month


# =============================================================================
# Main entry point
# =============================================================================

def run(df: pd.DataFrame, sem_scores: pd.DataFrame) -> dict:
    """
    Stage 5 full pipeline: build hrpid-linked wave-to-wave transition pairs
    across 2009-2024, walk-forward validate a logistic forward-prediction
    model, refit on everything, then score the most recent wave.

    Resilient to failure: returns {} (with a logged error) if hrpid is
    missing or transition pairs can't be built, rather than raising.
    """
    paths.UKHLS_FORWARD_TABLES.mkdir(parents=True, exist_ok=True)
    paths.UKHLS_FORWARD_FIGURES.mkdir(parents=True, exist_ok=True)

    work = df.join(sem_scores, how="left")
    feature_cols = _feature_cols(work)
    log.info("Stage 5 features: %s", feature_cols)

    log.info("Stage 5: building hrpid-linked wave-to-wave transition pairs...")
    transition_df = build_transition_pairs(work)
    if transition_df.empty:
        log.error("Stage 5: no transition pairs could be built -- aborting.")
        return {}
    _save_csv(transition_df, "stage5_transition_pairs")
    n_pairs = transition_df[["as_of_wave", "target_wave"]].drop_duplicates().shape[0]
    log.info("Stage 5: %d linked household transition rows across %d wave-pairs",
              len(transition_df), n_pairs)

    log.info("Stage 5: walk-forward validation (train on earlier waves, hold out the %d most "
              "recent known transitions)...", N_VALIDATION_PAIRS)
    val_result = run_walk_forward_validation(transition_df, feature_cols)

    log.info("Stage 5: refitting on all known transitions and scoring the latest wave...")
    pred_df = predict_latest_wave(work, transition_df, feature_cols)

    if not pred_df.empty:
        log.info("Stage 5: forward-prediction policy figures...")
        plot_forward_prediction_map(pred_df)
        plot_forward_prediction_by_month(pred_df)

    log.info("Stage 5 (forward vulnerability prediction) complete.")
    return {
        "transition_pairs": transition_df,
        "validation": val_result,
        "predictions": pred_df,
    }
