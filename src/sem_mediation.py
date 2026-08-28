"""
sem_mediation.py
────────────────
`ols_path()` -- a small, generic OLS path-estimation helper (coefficient,
SE, t, p per predictor) reused by `src.ukhls_cor_sem.test_fes_moderation`
for the COR-SEM FES-moderation test.

Originally part of the removed ENABLE-based Route 1 composite-path
architecture (`run()`, `bootstrap_mediation()`, `_plot_path_diagram()`, and
the rest of that module's FCP/AEMC/BLI/TCR/AEV-specific machinery) -- all
of that was deleted since it referenced columns (`fcp_score`, `aemc_score`,
...) that only ever existed in the removed `enable_preprocessing.py`
pipeline and could never run against the current UKHLS-based panel. Only
this one generic function survived the migration.

Methodology note
─────────────────
OLS path estimates are directional associations, not causal claims -- they
test whether the data are consistent with a theoretically specified
direction and significance, nothing stronger.
"""

from __future__ import annotations

from src.logging_utils import get_logger

log = get_logger("sem_mediation")


def ols_path(
    df,
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
