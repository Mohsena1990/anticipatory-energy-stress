"""
Stage 7 (analysis_plan_rerun.md): scope changes and retained sensitivity.

  * Equivalisation check, relabelled "sensitivity to equivalising income
    only": the primary 10% flag vs the same flag with net income divided by
    the modified-OECD scale (fuel spend is NOT equivalised). v2 outcome.
  * Descriptive refreshes of v1 figures the thesis quotes (no models):
      prepayment-meter use by vulnerability status;
      change in regional prevalence, waves a-e vs k-o;
      prevalence by tercile of the primary FES (growth-only magnitude and
      Delta), interviews 2010+.
  * Appendix scope note: CVAE, fuzzy c-means and one-class SVM are v1
    exploratory results moved to an appendix, not re-estimated in v2;
    vector-shift map dropped.

Weighted rates = mean of per-wave weighted rates (hh_xw), as elsewhere.
Outputs: outputs_v2/stage7/ and outputs_v2/reports/appendix_scope_note.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.paths import OUTPUTS_DIR, UKHLS_PANEL  # noqa: E402

OUT = OUTPUTS_DIR / "stage7"
NAVY, GRID = "#2E5077", "#E6E9ED"
GOR = {1: "North East", 2: "North West", 3: "Yorkshire and the Humber", 4: "East Midlands",
       5: "West Midlands", 6: "East of England", 7: "London", 8: "South East", 9: "South West",
       10: "Wales", 11: "Scotland", 12: "Northern Ireland"}


def wave_mean(g: pd.DataFrame, col: str) -> float:
    g = g[g[col].notna() & (g.hh_xw > 0)]
    r = [np.average(x[col].astype(float), weights=x.hh_xw) for _, x in g.groupby("wave")]
    return 100 * float(np.mean(r)) if r else np.nan


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    cols = ["wave", "gor_dv", "hh_xw", "hhsize", "high_fuel_vulnerable", "high_fuel_vulnerable_equivalised",
            "prepayment_meter", "fes_magnitude_growth3", "fes_delta_growth3"]
    df = pd.read_csv(UKHLS_PANEL, usecols=cols)
    df["region"] = df.gor_dv.map(GOR)

    # Equivalisation sensitivity.
    both = df.dropna(subset=["high_fuel_vulnerable", "high_fuel_vulnerable_equivalised", "hhsize"]).copy()
    both["size"] = pd.cut(both.hhsize, [0, 1, 2, 3, 4, 100], labels=["1", "2", "3", "4", "5+"])
    rows = []
    for s, g in both.groupby("size", observed=True):
        a, b = g.high_fuel_vulnerable.astype(int), g.high_fuel_vulnerable_equivalised.astype(int)
        rows.append(dict(household_size=str(s), n=len(g),
                         pct_flagged_primary_weighted=wave_mean(g, "high_fuel_vulnerable"),
                         pct_flagged_equivalised_income_weighted=wave_mean(g, "high_fuel_vulnerable_equivalised"),
                         pct_flip=100 * (a != b).mean(), pct_flip_in_to_out=100 * ((a == 1) & (b == 0)).mean(),
                         pct_flip_out_to_in=100 * ((a == 0) & (b == 1)).mean()))
    eq = pd.DataFrame(rows)
    eq.to_csv(OUT / "sensitivity_equivalised_income_only.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.bar(eq.household_size, eq.pct_flip, color=NAVY, width=0.6)
    for x, v in zip(eq.household_size, eq.pct_flip):
        ax.text(x, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_xlabel("Household size")
    ax.set_ylabel("% of households whose 10% flag changes")
    ax.set_title("Sensitivity to equivalising income only (fuel spend not equivalised)", fontsize=9, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"sensitivity_equivalised_income_only.{ext}", dpi=220)
    plt.close(fig)

    # Prepayment by vulnerability status.
    pp = []
    for v, g in df.dropna(subset=["high_fuel_vulnerable", "prepayment_meter"]).groupby("high_fuel_vulnerable"):
        pp.append(dict(status="vulnerable (>=10%)" if v == 1 else "not vulnerable (<10%)", n=len(g),
                       pct_prepayment_weighted=wave_mean(g, "prepayment_meter"),
                       pct_prepayment_unweighted=100 * g.prepayment_meter.mean()))
    pd.DataFrame(pp).to_csv(OUT / "prepayment_by_vulnerability.csv", index=False)

    # Regional change, early (a-e) vs late (k-o) waves.
    ch = []
    for r, g in df.dropna(subset=["region"]).groupby("region"):
        e, l_ = g[g.wave.isin(list("abcde"))], g[g.wave.isin(list("klmno"))]
        ch.append(dict(region=r, n_early=int(e.high_fuel_vulnerable.notna().sum()),
                       n_late=int(l_.high_fuel_vulnerable.notna().sum()),
                       pct_early_waves_a_e=wave_mean(e, "high_fuel_vulnerable"),
                       pct_late_waves_k_o=wave_mean(l_, "high_fuel_vulnerable")))
    ch = pd.DataFrame(ch)
    ch["change_pp"] = ch.pct_late_waves_k_o - ch.pct_early_waves_a_e
    ch.sort_values("change_pp").to_csv(OUT / "regional_change_early_late.csv", index=False)

    # Prevalence by FES tercile (national FES varies by year-month only).
    ft = []
    for fes in ["fes_magnitude_growth3", "fes_delta_growth3"]:
        g = df.dropna(subset=[fes, "high_fuel_vulnerable"]).copy()
        g["tercile"] = pd.qcut(g[fes].rank(method="first"), 3, labels=["low", "middle", "high"])
        for t, x in g.groupby("tercile", observed=True):
            ft.append(dict(fes=fes, tercile=str(t), n=len(x), fes_range=f"{x[fes].min():.2f} to {x[fes].max():.2f}",
                           pct_vulnerable_weighted=wave_mean(x, "high_fuel_vulnerable")))
    pd.DataFrame(ft).to_csv(OUT / "prevalence_by_fes_tercile.csv", index=False)

    note = """# Appendix scope note (Stage 7)

*analysis_plan_rerun.md, Stage 7. Written 2026-09-26.*

- **Moved to an appendix, not re-estimated in v2:** the COR-CVAE (latent resource representation and
  its alignment with the COR-SEM factors), fuzzy c-means vulnerability membership, and the one-class SVM
  anomaly scores. They are exploratory v1 results computed on the v1 outcome (which dropped off-gas-grid
  households and zero-filled non-response), the v1 second-order SEM (abandoned in v2), and v1 FES timing.
  They must be labelled as v1 exploratory analyses in the appendix, and no main-text claim rests on them.
  Source: tag `submitted-draft-v1`, `outputs/ukhls_cor_cvae/`, `outputs/ukhls_vulnerability/`.
  Re-estimating them on the v2 outcome is possible but was not part of the plan.
- **Dropped:** the vector-shift map (CVAE counterfactual FES shift). Not reported anywhere.
- **Dropped with the SEM route:** the second-order baseline-resource factor and its map (replaced by
  unit-weighted formative composites, Stage 4).
- **Retained, relabelled:** the equivalisation check, now "sensitivity to equivalising income only"
  (fuel spend is not equivalised). Recomputed on the v2 outcome: `outputs_v2/stage7/`.
- **Not re-estimated in v2 (v1 figures to drop or replace):** regional driver models by region
  (v1 Fig. 4-36, built on the v1 strain composite); v1 forward-risk maps by month and region and the
  "1.1% above 50% risk" figure (v1 Stage 5 scoring of wave o; v2 Stage 6 evaluates prediction on held-out
  transitions and does not score a future wave).
"""
    rep = OUTPUTS_DIR / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "appendix_scope_note.md").write_text(note)

    pd.set_option("display.width", 200)
    print(eq.round(2).to_string(index=False))
    print(pd.DataFrame(pp).round(2).to_string(index=False))
    print(ch.sort_values("change_pp").round(2).to_string(index=False))
    print(pd.DataFrame(ft).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
