"""
Stage 1 audit (analysis_plan_rerun.md, incl. amendment A1): how the v1
outcome treats UKHLS fuel-expenditure codes. Read-only: changes no pipeline
code or v1 output.

Compares, row by row, v1's `compute_fuel_to_income` spend rule with the
routing-aware A1 rule built from the UKHLS questionnaire routing:

  fuelhave1..4  "which fuels does the household use" (elec/gas/oil/other)
  fuelduel      asked only if elec AND gas are used (1 one bill, 2 separately)
  xpduely       asked if fuelduel == 1
  xpelecy/xpgasy asked if fuelduel == 2, or if fuelduel is DK/refused,
                 or if only that one of elec/gas is used
  xpoily        asked if fuelhave3 == 1;  xpsfly asked if fuelhave4 == 1

Negative codes: -8 inapplicable (routed out -> true structural zero or not
applicable), -1 don't know / -2 refused / -9 missing (item nonresponse ->
unknown amount, never zero).

A1 spend definitions:
  primary  complete-case; electricity must be reported (gas-only and
           oil/other-without-electricity households excluded)
  S1       lower bound: item nonresponse amounts set to 0, routing correct
  S2       primary + electricity-not-reported households (spend as reported)

Outputs (outputs_v2/):
  audit_fuel_codes.csv                         code dictionary + pooled counts
  audit/fuel_status_by_region_wave.csv         v1-vs-A1 row status
  audit/zero_filled_components_by_region_wave.csv
  audit/fuelduel_by_region_year.csv
  audit/ni_oil_lost_by_wave.csv
  audit/indicative_prevalence.csv              v1 vs A1 primary/S1/S2 rates
  audit/sample_flow_reconciliation.csv         339,201 -> analytical n
  audit/gap_explanations.csv
  audit/missing_vs_observed_spend.csv          by region/tenure/income/wave
  audit/elec_not_reported_rent_check.csv       fuel-in-rent + tenure
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.paths import OUTPUTS_DIR, V1_OUTPUTS_DIR, UKHLS_RAW_DIR  # noqa: E402

OUT = OUTPUTS_DIR
AUD = OUT / "audit"
AUD.mkdir(parents=True, exist_ok=True)

WAVES = "abcdefghijklmno"
FUEL = ["fuelduel", "xpduely", "xpgasy", "xpelecy", "xpoily", "xpsfly"]
HAVE = ["fuelhave1", "fuelhave2", "fuelhave3", "fuelhave4", "fuelhave96"]
PAY = ["elecpay", "gaspay", "duelpay"]
EXTRA = ["tenure_dv", "hsownd"]
NONRESP = (-1, -2, -9)
INCOME_FLOOR = 1200.0
GOR = {1: "North East", 2: "North West", 3: "Yorkshire and the Humber",
       4: "East Midlands", 5: "West Midlands", 6: "East of England",
       7: "London", 8: "South East", 9: "South West", 10: "Wales",
       11: "Scotland", 12: "Northern Ireland"}
TENURE = {1: "Owned outright", 2: "Owned with mortgage", 3: "Local authority rent",
          4: "Housing assoc rented", 5: "Rented from employer",
          6: "Rented private unfurnished", 7: "Rented private furnished", 8: "Other"}
LABEL = {-1: "don't know", -2: "refusal", -8: "inapplicable", -9: "missing"}


def load_raw() -> tuple[pd.DataFrame, dict]:
    frames, labels = [], {}
    for w in WAVES:
        p = UKHLS_RAW_DIR / f"{w}_hhresp.dta"
        r = pd.io.stata.StataReader(p)
        avail = set(r.variable_labels())
        want = ["hidp", "gor_dv", "fihhmnnet1_dv"] + FUEL + HAVE + PAY + EXTRA
        cols = [f"{w}_{v}" for v in want if f"{w}_{v}" in avail]
        xw = [c for c in avail if c.endswith("_xw")]
        d = pd.read_stata(p, columns=cols + xw, convert_categoricals=False)
        d = d.rename(columns={xw[0]: "hh_xw"}) if xw else d.assign(hh_xw=np.nan)
        d.columns = [c[2:] if c.startswith(f"{w}_") else c for c in d.columns]
        d["wave"] = w
        d["xw_name"] = xw[0][2:] if xw else ""
        frames.append(d)
        r2 = pd.io.stata.StataReader(p)
        r2.read(nrows=1)
        vl = r2.value_labels()
        for name, lbl in zip(r2._varlist, r2._lbllist):
            bare = name[2:]
            if bare in FUEL + HAVE and lbl in vl:
                for k, v in vl[lbl].items():
                    labels.setdefault((bare, int(k)), set()).add(v.strip())
    df = pd.concat(frames, ignore_index=True)
    for c in PAY:
        if c not in df:
            df[c] = np.nan
    return df, labels


def v1_spend(df: pd.DataFrame) -> pd.Series:
    """Exact replica of v1 src.ukhls_preprocessing.compute_fuel_to_income spend."""
    x = df[FUEL].where(df[FUEL] >= 0)          # v1: every negative code -> NaN
    separate = x.xpgasy.fillna(0) + x.xpelecy.fillna(0)
    eg = np.where(x.fuelduel == 1, x.xpduely,
                  np.where(x.fuelduel == 2, separate, np.nan))
    eg = pd.Series(eg, index=df.index)
    tot = eg + x.xpoily.fillna(0) + x.xpsfly.fillna(0)
    tot[eg.isna()] = np.nan
    return tot


def a1_spend(df: pd.DataFrame, nonresp_as_zero: bool = False) -> pd.Series:
    """Routing-aware spend for every household with a defined fuel set.

    nonresp_as_zero=False: item nonresponse -> NaN (primary / S2).
    nonresp_as_zero=True : item nonresponse -> 0   (S1 lower bound).
    Population restrictions (electricity reported etc.) are applied by
    `a1_reason`, not here.
    """
    elec, gas = df.fuelhave1 == 1, df.fuelhave2 == 1
    oil, oth = df.fuelhave3 == 1, df.fuelhave4 == 1

    def val(c):
        s = df[c].where(df[c] >= 0)
        if nonresp_as_zero:
            s = s.where(~df[c].isin(NONRESP), 0.0)
        return s

    both = elec & gas
    eg = pd.Series(0.0, index=df.index)          # neither elec nor gas: 0
    eg[both & (df.fuelduel == 1)] = val("xpduely")
    sep_rows = both & ((df.fuelduel == 2) | df.fuelduel.isin(NONRESP))
    eg[sep_rows] = val("xpgasy") + val("xpelecy")
    eg[elec & ~gas] = val("xpelecy")
    eg[gas & ~elec] = val("xpgasy")

    oil_v = pd.Series(0.0, index=df.index)
    oil_v[oil] = val("xpoily")
    oth_v = pd.Series(0.0, index=df.index)
    oth_v[oth] = val("xpsfly")
    return eg + oil_v + oth_v


def a1_reason(df: pd.DataFrame, spend_primary_raw: pd.Series) -> pd.Series:
    """First applicable A1 exclusion, in reconciliation order."""
    elec, gas = df.fuelhave1 == 1, df.fuelhave2 == 1
    oil, oth = df.fuelhave3 == 1, df.fuelhave4 == 1
    r = pd.Series("in_scope", index=df.index)
    r[spend_primary_raw.isna()] = "item_nonresponse"
    r[~elec & ~gas & (oil | oth)] = "elec_not_reported_oil_or_other_only"
    r[gas & ~elec] = "elec_not_reported_gas_only"
    r[~elec & ~gas & ~oil & ~oth] = "no_fuel_reported"
    r[(df[HAVE[:4]] < 0).any(axis=1)] = "fuelhave_module_nonresponse"
    return r


def which_item_missing(df: pd.DataFrame) -> pd.Series:
    """For item-nonresponse rows, name the first required amount that is missing."""
    elec, gas = df.fuelhave1 == 1, df.fuelhave2 == 1
    nr = lambda c: df[c].isin(NONRESP)  # noqa: E731
    both = elec & gas
    out = pd.Series("", index=df.index)
    conds = [
        ("xpduely", both & (df.fuelduel == 1) & nr("xpduely")),
        ("xpgasy", ((both & ((df.fuelduel == 2) | df.fuelduel.isin(NONRESP))) | (gas & ~elec)) & nr("xpgasy")),
        ("xpelecy", ((both & ((df.fuelduel == 2) | df.fuelduel.isin(NONRESP))) | (elec & ~gas)) & nr("xpelecy")),
        ("xpoily", (df.fuelhave3 == 1) & nr("xpoily")),
        ("xpsfly", (df.fuelhave4 == 1) & nr("xpsfly")),
    ]
    for name, c in reversed(conds):
        out[c] = name
    return out


def ratio_and_guard(spend: pd.Series, income_m: pd.Series) -> tuple[pd.Series, pd.Series]:
    # Sentinels (-10 = "not available for IEMB") are missing; other negative
    # values are real losses and fall under the £1,200 guard, as in v1.
    missing = income_m.isna() | income_m.isin([-10, -9, -8, -7, -2, -1])
    inc = income_m.where(~missing) * 12
    guard = pd.Series("ok", index=spend.index)
    guard[missing] = "income_missing"
    guard[~missing & (inc < INCOME_FLOOR)] = "income_below_1200"
    ratio = (spend / inc).where(guard == "ok").clip(upper=1.0)
    return ratio, guard


def status(v1: pd.Series, ref: pd.Series, reason: pd.Series) -> pd.Series:
    s = pd.Series("", index=v1.index)
    both = v1.notna() & ref.notna()
    s[both & np.isclose(v1, ref)] = "A_same"
    s[both & ~np.isclose(v1, ref)] = "A2_valid_but_different"
    s[v1.notna() & ref.isna()] = "B_v1_kept_A1_excludes_" + reason[v1.notna() & ref.isna()]
    s[v1.isna() & ref.notna()] = "C_v1_dropped_but_observed"
    s[v1.isna() & ref.isna()] = "D_dropped_" + reason[v1.isna() & ref.isna()]
    return s


def code_dictionary(df, labels) -> pd.DataFrame:
    elec, gas = df.fuelhave1 == 1, df.fuelhave2 == 1
    ctx = np.select([elec & gas, elec & ~gas, gas & ~elec],
                    ["elec+gas", "elec_only", "gas_only"], "neither_elec_nor_gas")
    rows = []
    for v in FUEL:
        s = df[v]
        cats = np.where(s < 0, s, np.where(s == 0, 0, 1)).astype(int)
        t = pd.DataFrame({"code": cats, "context": ctx, "fuelduel": df.fuelduel})
        for (code, c, fd), n in t.groupby(["code", "context", "fuelduel"]).size().items():
            lab = ("positive amount" if code == 1 and v != "fuelduel"
                   else "zero amount" if code == 0 and v != "fuelduel"
                   else "/".join(sorted(labels.get((v, code), {LABEL.get(code, "")}))))
            rows.append(dict(variable=v, code=code, label=lab, household_fuels=c,
                             fuelduel=int(fd), n=int(n),
                             routing_meaning=_meaning(v, code),
                             v1_treatment=_v1(v, code, fd),
                             correct_treatment=_correct(v, code)))
    for v in HAVE:
        for code, n in df[v].value_counts().items():
            rows.append(dict(variable=v, code=int(code),
                             label="/".join(sorted(labels.get((v, int(code)), {""}))),
                             household_fuels="", fuelduel=np.nan, n=int(n),
                             routing_meaning=_meaning(v, int(code)),
                             v1_treatment="not used by v1 outcome",
                             correct_treatment="defines routing (which amounts are required)"))
    return pd.DataFrame(rows).sort_values(["variable", "code", "household_fuels", "fuelduel"])


def _meaning(v, code):
    if code == -8:
        return "not applicable (routed out: fuel not used / billing branch not taken)"
    if code in NONRESP:
        return "item nonresponse (amount unknown)"
    if v.startswith("fuelhave"):
        return "fuel used" if code == 1 else "fuel not used" if code == 0 else ""
    if v == "fuelduel":
        return {1: "one combined gas+elec bill", 2: "separate gas and elec bills"}.get(code, "")
    return "reported amount" if code == 1 else "reported zero spend"


def _v1(v, code, fd):
    if code >= 0:
        return "used as reported" if v != "fuelduel" else "selects billing branch"
    if v == "fuelduel":
        return "NaN -> whole household dropped from outcome"
    if v == "xpduely":
        return "NaN -> household dropped" if fd == 1 else "ignored (branch not taken)"
    if v in ("xpgasy", "xpelecy"):
        if fd == 2:
            return "NaN -> filled with 0 (spend understated)" if code != -8 else "0 (correct)"
        return "ignored: household dropped because fuelduel is NaN"
    return "NaN -> filled with 0" + (" (spend understated)" if code != -8 else " (correct)")


def _correct(v, code):
    if code == -8:
        return "structural zero / not applicable"
    if code in NONRESP:
        return "missing (unknown amount): household spend missing"
    return "use value"


def reconciliation(df: pd.DataFrame) -> pd.DataFrame:
    rows = [("0", "All UKHLS household-wave rows (waves a-o)", len(df), np.nan)]
    remaining = len(df)
    order = [("fuelhave_module_nonresponse", "Fuel-use module nonresponse (fuelhave* < 0)"),
             ("no_fuel_reported", "No fuel reported (fuelhave96 / none mentioned)"),
             ("elec_not_reported_gas_only", "Electricity not reported: gas only [S2 adds back]"),
             ("elec_not_reported_oil_or_other_only", "Electricity not reported: oil/other only [S2 adds back]")]
    for i, (k, lab) in enumerate(order, 1):
        n = int((df.a1_reason == k).sum())
        remaining -= n
        rows.append((str(i), lab, -n, remaining))
    item = df[df.a1_reason == "item_nonresponse"]
    for j, (k, n) in enumerate(item.a1_missing_item.value_counts().items()):
        remaining -= int(n)
        rows.append((f"5{'abcde'[j]}", f"Item nonresponse, first missing amount = {k} [S1 adds back as 0]",
                     -int(n), remaining))
    ins = df[df.a1_reason == "in_scope"]
    for k, lab in [("income_missing", "Household net income missing (sentinel code)"),
                   ("income_below_1200", "Annual net income < £1,200 guard (never logged in v1)")]:
        n = int((ins.a1_guard == k).sum())
        remaining -= n
        rows.append(("6" if k == "income_missing" else "7", lab, -n, remaining))
    rows.append(("=", "Primary analytical n (fuel_to_income_ratio non-missing)",
                 int(df.a1_ratio.notna().sum()), np.nan))
    rows.append(("info", "  of which ratio capped at 1.0 (kept, not excluded)",
                 int((ins.a1_ratio_uncapped > 1).sum()), np.nan))
    rows.append(("info", "v1 analytical n for comparison", int(df.fuel_to_income_ratio.notna().sum()), np.nan))
    rows.append(("info", "S1 (lower-bound) analytical n", int(df.s1_ratio.notna().sum()), np.nan))
    rows.append(("info", "S2 (+elec-not-reported) analytical n", int(df.s2_ratio.notna().sum()), np.nan))
    out = pd.DataFrame(rows, columns=["step", "description", "n_change_or_total", "remaining"])
    assert out.iloc[-5].n_change_or_total == remaining, "reconciliation does not close"
    return out


def v1_reconciliation(df: pd.DataFrame) -> list[tuple]:
    fd = df.fuelduel
    rows = [("v1.0", "All rows", len(df))]
    m1 = fd == -8
    m2 = fd.isin(NONRESP)
    m3 = (fd == 1) & df.xpduely.isin(NONRESP)
    kept = df.v1_spend.notna()
    rows += [("v1.1", "fuelduel = -8 inapplicable (not dual-fuel) -> dropped", -int(m1.sum())),
             ("v1.2", "fuelduel DK/refused/missing -> dropped", -int(m2.sum())),
             ("v1.3", "fuelduel = 1 and xpduely nonresponse -> dropped", -int(m3.sum())),
             ("v1.4", "= rows with v1 spend", int(kept.sum())),
             ("v1.5", "  income missing", -int((kept & (df.v1_guard == "income_missing")).sum())),
             ("v1.6", "  income < £1,200 guard (not logged in v1)", -int((kept & (df.v1_guard == "income_below_1200")).sum())),
             ("v1.7", "= v1 analytical n", int(df.fuel_to_income_ratio.notna().sum()))]
    assert rows[4][2] == len(df) + rows[1][2] + rows[2][2] + rows[3][2]
    assert rows[7][2] == rows[4][2] + rows[5][2] + rows[6][2]
    return rows


def missing_vs_observed(df: pd.DataFrame) -> pd.DataFrame:
    """Among households with electricity reported (primary population before the
    item-nonresponse step), compare spend-missing vs spend-observed."""
    pop = df[df.a1_reason.isin(["in_scope", "item_nonresponse"])].copy()
    pop["spend_status"] = np.where(pop.a1_reason == "item_nonresponse", "missing", "observed")
    inc = pop.fihhmnnet1_dv.where(pop.fihhmnnet1_dv >= 0)
    pop["income_band"] = inc.groupby(pop.wave).transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)]))
    pop["income_band"] = pop.income_band.astype(object).fillna("income missing")
    pop["tenure"] = pop.tenure_dv.map(TENURE).fillna("Missing tenure")
    out = []
    for var in ["region", "tenure", "income_band", "wave"]:
        t = pd.crosstab(pop[var], pop.spend_status)
        t["pct_missing_within_group"] = 100 * t.missing / (t.missing + t.observed)
        t["pct_of_all_missing"] = 100 * t.missing / t.missing.sum()
        t["pct_of_all_observed"] = 100 * t.observed / t.observed.sum()
        t = t.reset_index().rename(columns={var: "group"})
        t.insert(0, "dimension", var)
        out.append(t)
    res = pd.concat(out, ignore_index=True)
    res.loc[len(res)] = ["all", "all", pop.spend_status.eq("missing").sum(),
                         pop.spend_status.eq("observed").sum(),
                         100 * pop.spend_status.eq("missing").mean(), 100.0, 100.0]
    return res


def rent_check(df: pd.DataFrame) -> pd.DataFrame:
    groups = {"gas_only": df.a1_reason == "elec_not_reported_gas_only",
              "oil_or_other_only": df.a1_reason == "elec_not_reported_oil_or_other_only",
              "primary_in_scope (reference)": df.a1_reason == "in_scope"}
    rows = []
    renter = df.tenure_dv.isin([3, 4, 5, 6, 7])
    for g, m in groups.items():
        sub = df[m]
        waves_c_o = sub.wave >= "c"
        r = dict(group=g, n=int(m.sum()), n_waves_c_to_o=int(waves_c_o.sum()),
                 pct_renting=100 * renter[m].mean(),
                 pct_social_rent=100 * df.tenure_dv[m].isin([3, 4]).mean(),
                 pct_private_rent=100 * df.tenure_dv[m].isin([6, 7]).mean(),
                 pct_owner=100 * df.tenure_dv[m].isin([1, 2]).mean())
        for c in PAY:
            s = sub.loc[waves_c_o, c]
            asked = s.notna() & (s != -8)
            r[f"{c}_asked_n"] = int(asked.sum())
            r[f"{c}_included_in_rent_n"] = int((s == 5).sum())
            r[f"{c}_included_in_rent_pct_of_asked"] = (100 * (s == 5).sum() / asked.sum()
                                                       if asked.sum() else np.nan)
        r["pct_with_positive_reported_spend"] = 100 * (df.s2_spend[m] > 0).mean()
        r["median_reported_spend_gbp"] = df.s2_spend[m].median()
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    df, labels = load_raw()
    panel = pd.read_csv(V1_OUTPUTS_DIR / "ukhls_cleaned" / "ukhls_panel.csv",
                        usecols=["hidp", "wave", "interview_year", "total_fuel_spend",
                                 "fuel_to_income_ratio", "high_fuel_vulnerable"])
    df = df.merge(panel, on=["hidp", "wave"], how="left", validate="1:1")
    df["region"] = df.gor_dv.map(GOR).fillna("Missing region")

    # --- v1 replica + replication check ---------------------------------
    df["v1_spend"] = v1_spend(df)
    mism = ~((df.v1_spend.isna() & df.total_fuel_spend.isna())
             | np.isclose(df.v1_spend, df.total_fuel_spend))
    print(f"v1 replication: {int(mism.sum())} mismatches of {len(df)} rows")
    assert mism.sum() == 0, "audit v1 replica does not reproduce v1 panel"
    df["v1_ratio"], df["v1_guard"] = ratio_and_guard(df.v1_spend, df.fihhmnnet1_dv)
    assert ((df.v1_ratio.isna() & df.fuel_to_income_ratio.isna())
            | np.isclose(df.v1_ratio, df.fuel_to_income_ratio)).all()

    # --- A1 outcome: primary, S1, S2 -------------------------------------
    raw_primary = a1_spend(df)
    df["a1_reason"] = a1_reason(df, raw_primary)
    df["a1_missing_item"] = np.where(df.a1_reason == "item_nonresponse", which_item_missing(df), "")
    in_scope = df.a1_reason == "in_scope"
    elec_nr = df.a1_reason.str.startswith("elec_not_reported")
    df["a1_spend"] = raw_primary.where(in_scope)
    df["s1_spend"] = a1_spend(df, nonresp_as_zero=True).where(
        in_scope | (df.a1_reason == "item_nonresponse"))
    df["s2_spend"] = raw_primary.where(in_scope | elec_nr)
    inc12 = df.fihhmnnet1_dv * 12
    df["a1_ratio_uncapped"] = df.a1_spend / inc12
    df["a1_ratio"], df["a1_guard"] = ratio_and_guard(df.a1_spend, df.fihhmnnet1_dv)
    df["s1_ratio"], _ = ratio_and_guard(df.s1_spend, df.fihhmnnet1_dv)
    df["s2_ratio"], _ = ratio_and_guard(df.s2_spend, df.fihhmnnet1_dv)
    df["status"] = status(df.v1_spend, df.a1_spend, df.a1_reason)

    # --- outputs ---------------------------------------------------------
    code_dictionary(df, labels).to_csv(OUT / "audit_fuel_codes.csv", index=False)
    (df.groupby(["region", "wave", "status"]).size().unstack(fill_value=0)
       .reset_index().to_csv(AUD / "fuel_status_by_region_wave.csv", index=False))

    kept = df.v1_spend.notna()
    comp = pd.DataFrame({
        "xpelecy_nonresp_zeroed": kept & (df.fuelduel == 2) & df.xpelecy.isin(NONRESP),
        "xpgasy_nonresp_zeroed": kept & (df.fuelduel == 2) & df.xpgasy.isin(NONRESP),
        "xpoily_nonresp_zeroed": kept & df.xpoily.isin(NONRESP),
        "xpsfly_nonresp_zeroed": kept & df.xpsfly.isin(NONRESP),
    })
    comp["any_component_zeroed"] = comp.any(axis=1)
    comp[["region", "wave"]] = df[["region", "wave"]]
    comp.groupby(["region", "wave"]).sum().reset_index().to_csv(
        AUD / "zero_filled_components_by_region_wave.csv", index=False)

    fd = df.fuelduel.map({1: "1_one_bill", 2: "2_separate", -8: "-8_inapplicable",
                          -1: "-1_dont_know", -2: "-2_refused", -9: "-9_missing"})
    (pd.crosstab([df.region, df.interview_year], fd).reset_index()
       .to_csv(AUD / "fuelduel_by_region_year.csv", index=False))

    ni = df[df.region == "Northern Ireland"]
    ni_oil = ni.fuelhave3 == 1
    lost = ni_oil & ni.v1_spend.isna() & (ni.fuelduel < 0)
    ni_tab = pd.DataFrame({
        "ni_households": ni.groupby("wave").size(),
        "ni_oil_households": ni_oil.groupby(ni.wave).sum(),
        "ni_oil_dropped_by_fuelduel_rule": lost.groupby(ni.wave).sum(),
        "of_which_in_A1_primary": (lost & ni.a1_spend.notna()).groupby(ni.wave).sum(),
        "ni_oil_kept_by_v1": (ni_oil & ni.v1_spend.notna()).groupby(ni.wave).sum(),
    }).reset_index()
    ni_tab.loc[len(ni_tab)] = ["all"] + ni_tab.iloc[:, 1:].sum().tolist()
    ni_tab.to_csv(AUD / "ni_oil_lost_by_wave.csv", index=False)

    def rates(k):
        res = {}
        for name, col in [("v1", "v1_ratio"), ("a1_primary", "a1_ratio"),
                          ("s1_lower_bound", "s1_ratio"), ("s2_plus_elec_nr", "s2_ratio")]:
            flag = (df[col] >= 0.10).where(df[col].notna())
            g = flag.groupby(df[k] if k else pd.Series("UK", index=df.index))
            res[f"{name}_n"] = g.count()
            res[f"{name}_pct"] = 100 * g.mean()
        return pd.DataFrame(res)
    ind = pd.concat([rates(None).assign(level="all"),
                     rates("region").assign(level="region"),
                     rates("wave").assign(level="wave")])
    ind.index.name = "group"
    ind.reset_index().to_csv(AUD / "indicative_prevalence.csv", index=False)

    rec = reconciliation(df)
    v1r = pd.DataFrame(v1_reconciliation(df), columns=["step", "description", "n_change_or_total"])
    pd.concat([rec, v1r], ignore_index=True).to_csv(AUD / "sample_flow_reconciliation.csv", index=False)

    # Gap explanations for the Stage 1 report (draft reference rule counts).
    a2 = df[df.status == "A2_valid_but_different"]
    v1_spend_rows = int(kept.sum())
    gaps = pd.DataFrame([
        ("status_sum_off_by_one",
         "Stage 1 status counts summed to 339,200 without the single A2 row "
         "(v1 kept, A1 kept, spend differs). It is fuelduel=2 with gas not "
         "reported: v1 adds xpgasy, A1 counts electricity only.",
         len(a2), "; ".join(f"hidp={h} wave={w} v1={v:.0f} A1={a:.0f}"
                            for h, w, v, a in a2[["hidp", "wave", "v1_spend", "a1_spend"]].values)),
        ("v1_spend_rows_minus_v1_analytical_n",
         "Rows with a v1 spend (same + zero-filled + A2) minus v1 analytical n: "
         "removed by the income step inside compute_fuel_to_income.",
         v1_spend_rows - int(df.fuel_to_income_ratio.notna().sum()),
         f"income missing={int((kept & (df.v1_guard=='income_missing')).sum())}; "
         f"income<£1,200={int((kept & (df.v1_guard=='income_below_1200')).sum())}"),
        ("a1_spend_rows_minus_a1_analytical_n",
         "Same gap for the A1 primary outcome.",
         int(df.a1_spend.notna().sum() - df.a1_ratio.notna().sum()),
         f"income missing={int((in_scope & (df.a1_guard=='income_missing')).sum())}; "
         f"income<£1,200={int((in_scope & (df.a1_guard=='income_below_1200')).sum())}"),
    ], columns=["gap", "explanation", "n", "detail"])
    gaps.to_csv(AUD / "gap_explanations.csv", index=False)

    missing_vs_observed(df).to_csv(AUD / "missing_vs_observed_spend.csv", index=False)
    rent_check(df).to_csv(AUD / "elec_not_reported_rent_check.csv", index=False)

    pd.set_option("display.width", 220)
    print(df.status.value_counts().to_string())
    print(pd.concat([rec, v1r]).to_string(index=False))
    print(gaps.to_string(index=False))
    print(ni_tab.tail(1).to_string(index=False))
    print(ind.round(2).to_string())


if __name__ == "__main__":
    main()
