"""
POST-HOC / EXPLORATORY (logged deviation 2026-09-26): P3 = P0 + P1.

Same A6 common sample, split and training-only standardisation as
scripts/stage6_prediction.py (functions imported from there, not copied).
Reports ROC-AUC, PR-AUC and top-10% sensitivity/PPV for P0, P1 and P3, and
paired PSU-bootstrap delta AUC for P3 - P0 and P3 - P1. Not part of the
pre-specified analysis; no pre-specified conclusion depends on it.

Outputs (outputs_v2/stage6/): posthoc_p3_metrics.csv, posthoc_p3_delta_auc.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import stage6_prediction as S  # noqa: E402


def main() -> None:
    pairs, _ = S.transitions()
    tr_all = pairs[~pairs.transition.isin(S.VALIDATION)].copy()
    va_all = pairs[pairs.transition.isin(S.VALIDATION)].copy()
    S.build_composites(tr_all, va_all)
    allp = pd.concat([tr_all, va_all])
    need = ["y", "psu"] + S.P0 + S.P1 + [S.FES_T1]
    cs = ~allp[need].isna().any(axis=1)
    tr, va = tr_all[cs.loc[tr_all.index]].copy(), va_all[cs.loc[va_all.index]].copy()
    S.standardise(tr, va, sorted(set(S.P0 + S.P1 + [S.FES_T1])))

    models = {"P0": S.P0, "P1": S.P1, "P3 (post-hoc) = P0 + P1": S.P0 + S.P1}
    preds = {m: S.fit_predict(tr, va, cols)[1] for m, cols in models.items()}
    y = va.y.values.astype(int)
    stats, _ = S.bootstrap(va, preds)
    rows = []
    for m, p in preds.items():
        s10, v10 = S.topk(y, p, 0.10)
        for k, v in [("auc", roc_auc_score(y, p)), ("prauc", average_precision_score(y, p)),
                     ("sens10", s10), ("ppv10", v10)]:
            rows.append(dict(model=m, metric=k, estimate=v, ci_low=np.percentile(stats[m][k], 2.5),
                             ci_high=np.percentile(stats[m][k], 97.5), n_train=len(tr), n_validation=len(y),
                             label="POST-HOC / EXPLORATORY" if m.startswith("P3") else "pre-specified (reference)"))
    pd.DataFrame(rows).to_csv(S.OUT / "posthoc_p3_metrics.csv", index=False)
    p3 = "P3 (post-hoc) = P0 + P1"
    d = []
    for ref in ["P0", "P1"]:
        diff = np.array(stats[p3]["auc"]) - np.array(stats[ref]["auc"])
        d.append(dict(comparison=f"P3 - {ref}", delta_auc=roc_auc_score(y, preds[p3]) - roc_auc_score(y, preds[ref]),
                      ci_low=np.percentile(diff, 2.5), ci_high=np.percentile(diff, 97.5),
                      label="POST-HOC / EXPLORATORY"))
    pd.DataFrame(d).to_csv(S.OUT / "posthoc_p3_delta_auc.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(d).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
