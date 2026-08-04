"""
unsupervised_latent.py
───────────────────────
Route 1 (COR Composite Route) — empirical recovery check: does the
data-driven latent structure of the survey items recover the same four
dimensions as Route 1's formative composite scores?

Purpose
───────
The COR-derived household vulnerability dimensions are theory-specified.
This stream asks: are those same dimensions recoverable from the empirical
structure of the item-level data, without imposing the COR framework?

This is Route 1's own internal robustness check, not an alternative latent-
variable "route" in its own right — Route 2 (`src.cor_sem`, true CFA/SEM)
and Route 3 (`src.cor_vae`, theory-informed VAE) are the two genuinely
different estimation routes compared against Route 1 in
`src.route_comparison`. `build_item_matrix()` and `fit_encode_train_test()`
below are also reused by `src.ml_classification` for leakage-free PCA/EFA/
linear-AE feature fitting.

Methods
───────
  1. PCA              — linear variance decomposition
  2. EFA              — factor_analyzer / PCA fallback
  3. Linear autoencoder — Keras encoder with linear activations and a
                          narrow bottleneck (config.N_LATENT_DIMS neurons)

Robustness extensions
─────────────────────
  4. Train / val reconstruction gap
  5. Seed stability: 30 seeds, Hungarian-aligned |r|, mean ± SD
  6. Bottleneck sweep: n_latent ∈ config.AE_BOTTLENECK_SIZES

Comparison
──────────
  corr(COR_composite_scores, ML_latent_scores)

  A high Pearson/Spearman correlation between a COR construct score and a
  data-driven latent dimension indicates that the theory-specified construct
  is empirically recoverable from the items.

  Note: the unsupervised ML stream does NOT replace COR theory.
  It tests whether the theory-derived household vulnerability dimensions are
  recoverable from the empirical structure of the data.

Usage
─────
  from src.unsupervised_latent import run
  run(df)   # df from enable_preprocessing.run()
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

from src import config, paths
from src.construct_mapping import ALL_CONSTRUCT_ITEMS, CONSTRUCT_REGISTRY

log = logging.getLogger(__name__)

_CONSTRUCT_SCORES = {
    "FCP":  "fcp_score",
    "AEMC": "aemc_score",
    "BLI":  "bli_score",
    "TCR":  "tcr_score",
}


# =============================================================================
# I/O helpers
# =============================================================================

def _save_csv(df: pd.DataFrame, name: str) -> None:
    paths.LATENT_TABLES.mkdir(parents=True, exist_ok=True)
    p = paths.LATENT_TABLES / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p.name)


def _save_fig(fig: plt.Figure, name: str) -> None:
    paths.LATENT_FIGURES.mkdir(parents=True, exist_ok=True)
    p = paths.LATENT_FIGURES / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(p, dpi=config.DPI, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


# =============================================================================
# Item matrix preparation
# =============================================================================

def build_item_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Collect all available normalized item columns (n_{item}) and return a
    matrix with listwise-complete rows.
    """
    all_n_cols = [f"n_{c}" for c in ALL_CONSTRUCT_ITEMS if f"n_{c}" in df.columns]
    if not all_n_cols:
        raise ValueError("No normalized item columns found.  Run enable_preprocessing first.")

    mat = df[all_n_cols].dropna(how="any").copy()
    items = [c.replace("n_", "") for c in all_n_cols]
    log.info("Item matrix: %d rows × %d items (after listwise deletion)", *mat.shape)
    return mat, items


# =============================================================================
# PCA
# =============================================================================

def run_pca(item_matrix: pd.DataFrame, n_components: int = None) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if n_components is None:
        n_components = min(config.PCA_COMPONENTS, item_matrix.shape[1])

    X = StandardScaler().fit_transform(item_matrix.values)
    pca = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
    scores = pca.fit_transform(X)

    scores_df = pd.DataFrame(
        scores,
        index=item_matrix.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=item_matrix.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    loadings_df.index = [c.replace("n_", "") for c in loadings_df.index]
    evr_df = pd.DataFrame({
        "component":            [f"PC{i+1}" for i in range(n_components)],
        "explained_variance":   pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_evr":       np.cumsum(pca.explained_variance_ratio_),
    })

    log.info("PCA: first %d components explain %.1f%% of variance",
             n_components, 100 * pca.explained_variance_ratio_[:n_components].sum())
    return {"scores": scores_df, "loadings": loadings_df, "evr": evr_df, "pca": pca}


# =============================================================================
# EFA
# =============================================================================

def run_efa_latent(item_matrix: pd.DataFrame, n_factors: int = None) -> dict:
    from sklearn.preprocessing import StandardScaler

    if n_factors is None:
        n_factors = min(config.EFA_FACTORS, item_matrix.shape[1])

    try:
        from factor_analyzer import FactorAnalyzer
        X = StandardScaler().fit_transform(item_matrix.values)
        fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax", method="ml")
        fa.fit(pd.DataFrame(X, columns=item_matrix.columns))
        loadings = fa.loadings_
        scores   = fa.transform(pd.DataFrame(X, columns=item_matrix.columns))
        ev, _    = fa.get_eigenvalues()
        evr      = ev[:n_factors] / ev.sum()
    except (ImportError, Exception) as e:
        log.info("EFA: falling back to PCA approximation (%s)", e)
        res = run_pca(item_matrix, n_components=n_factors)
        return {
            "scores":   res["scores"].rename(
                columns={f"PC{i+1}": f"EFA{i+1}" for i in range(n_factors)}),
            "loadings": res["loadings"],
            "evr":      res["evr"],
            "method":   "pca_fallback",
        }

    scores_df = pd.DataFrame(
        scores, index=item_matrix.index,
        columns=[f"EFA{i+1}" for i in range(n_factors)],
    )
    loadings_df = pd.DataFrame(
        loadings,
        index=[c.replace("n_", "") for c in item_matrix.columns],
        columns=[f"EFA{i+1}" for i in range(n_factors)],
    )
    evr_df = pd.DataFrame({
        "factor": [f"EFA{i+1}" for i in range(n_factors)],
        "eigenvalue":          ev[:n_factors],
        "explained_var_ratio": evr,
    })
    log.info("EFA: %d factors, explaining %.1f%% variance", n_factors, 100 * evr.sum())
    return {"scores": scores_df, "loadings": loadings_df, "evr": evr_df, "method": "efa"}


# =============================================================================
# Linear autoencoder — core
# =============================================================================

def _train_autoencoder(
    n_features: int, n_latent: int, lr: float,
    X_all: np.ndarray, X_tr: np.ndarray, X_val: np.ndarray,
    epochs: int,
) -> tuple:
    """
    Build *and* train a linear autoencoder entirely within a single tf.device
    context so that variables and ops always live on the same device.

    Forced to CPU when config.AE_FORCE_CPU is True, which avoids XLA/Triton
    compilation errors on GPUs with unsupported Compute Capability (Blackwell
    CC 12.0a with TF toolkit 12.5).

    Returns (latent, overall_mse, train_mse, val_mse).
    """
    import tensorflow as tf

    device = "/CPU:0" if config.AE_FORCE_CPU else "/GPU:0"
    cb = tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)

    with tf.device(device):
        tf.keras.backend.clear_session()
        inp     = tf.keras.Input(shape=(n_features,))
        encoded = tf.keras.layers.Dense(n_latent, activation="linear", name="bottleneck")(inp)
        decoded = tf.keras.layers.Dense(n_features, activation="linear", name="reconstruction")(encoded)
        autoencoder = tf.keras.Model(inp, decoded)
        encoder     = tf.keras.Model(inp, encoded)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")

        autoencoder.fit(
            X_tr, X_tr,
            epochs=epochs,
            batch_size=64,
            validation_data=(X_val, X_val),
            verbose=0,
            callbacks=[cb],
        )
        latent    = encoder.predict(X_all, verbose=0)
        recon_all = autoencoder.predict(X_all, verbose=0)
        recon_tr  = autoencoder.predict(X_tr,  verbose=0)
        recon_val = autoencoder.predict(X_val, verbose=0)

    return (
        latent,
        float(np.mean((X_all - recon_all) ** 2)),
        float(np.mean((X_tr  - recon_tr)  ** 2)),
        float(np.mean((X_val - recon_val) ** 2)),
    )


def run_linear_autoencoder(
    item_matrix: pd.DataFrame,
    n_latent: int = None,
    epochs: int = None,
    lr: float = None,
    seed: int = None,
) -> dict:
    """
    Fit a single linear autoencoder.

    Returns dict with:
      scores     — DataFrame of bottleneck activations (index = item_matrix.index)
      recon_mse  — overall reconstruction MSE
      train_mse  — training-split MSE
      val_mse    — validation-split MSE (train/val gap = val_mse - train_mse)
      n_latent   — bottleneck size
    """
    if n_latent is None: n_latent = config.N_LATENT_DIMS
    if epochs  is None: epochs   = config.AE_EPOCHS
    if lr      is None: lr       = config.AE_LEARNING_RATE
    if seed    is None: seed     = config.TF_SEED

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        np.random.seed(seed)
    except ImportError:
        log.warning("TensorFlow not available — skipping autoencoder")
        return {}

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_raw = item_matrix.values.astype(np.float32)
    X     = StandardScaler().fit_transform(X_raw)
    n_features = X.shape[1]

    X_tr, X_val = train_test_split(X, test_size=config.AE_VAL_SPLIT, random_state=seed)

    try:
        latent, recon_mse, train_mse, val_mse = _train_autoencoder(
            n_features, n_latent, lr, X, X_tr, X_val, epochs
        )
    except Exception as e:
        log.error("Autoencoder training failed (seed=%d, n_latent=%d): %s", seed, n_latent, e)
        return {}

    scores_df = pd.DataFrame(
        latent,
        index=item_matrix.index,
        columns=[f"AE{i+1}" for i in range(n_latent)],
    )
    log.info(
        "AE n_latent=%d seed=%d  overall_MSE=%.5f  train=%.5f  val=%.5f  gap=%.5f",
        n_latent, seed, recon_mse, train_mse, val_mse, val_mse - train_mse,
    )
    return {
        "scores":    scores_df,
        "recon_mse": recon_mse,
        "train_mse": train_mse,
        "val_mse":   val_mse,
        "n_latent":  n_latent,
        "method":    "linear_autoencoder",
    }


# =============================================================================
# Leakage-free within-split encoder
# =============================================================================

def fit_encode_train_test(
    item_mat_tr: pd.DataFrame,
    item_mat_te: pd.DataFrame,
    source: str,
    seed: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit an encoder (PCA / EFA / linear AE) on training items only, then
    transform both splits.  This eliminates transductive leakage that arises
    when encoders are pre-fitted on the full dataset (including test households).

    Parameters
    ----------
    item_mat_tr : normalized item matrix for the TRAINING split
    item_mat_te : normalized item matrix for the TEST split (same columns)
    source      : 'pca', 'efa', or 'ae'
    seed        : random seed

    Returns
    -------
    (latent_tr_df, latent_te_df) — DataFrames with sequential 0-based index
    """
    from sklearn.preprocessing import StandardScaler

    if seed is None:
        seed = config.RANDOM_SEED

    scaler  = StandardScaler().fit(item_mat_tr.values)
    X_tr    = scaler.transform(item_mat_tr.values).astype(np.float32)
    X_te    = scaler.transform(item_mat_te.values).astype(np.float32)
    cols_in = item_mat_tr.columns.tolist()

    if source == "pca":
        from sklearn.decomposition import PCA
        n_comp = min(config.PCA_COMPONENTS, X_tr.shape[1])
        pca    = PCA(n_components=n_comp, random_state=seed).fit(X_tr)
        lat_tr = pca.transform(X_tr)
        lat_te = pca.transform(X_te)
        prefix = "PC"
        n_dims = n_comp

    elif source == "efa":
        n_factors = min(config.EFA_FACTORS, X_tr.shape[1])
        try:
            from factor_analyzer import FactorAnalyzer
            fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax", method="ml")
            fa.fit(pd.DataFrame(X_tr, columns=cols_in))
            lat_tr = fa.transform(pd.DataFrame(X_tr, columns=cols_in))
            lat_te = fa.transform(pd.DataFrame(X_te, columns=cols_in))
        except Exception:
            from sklearn.decomposition import PCA
            pca    = PCA(n_components=n_factors, random_state=seed).fit(X_tr)
            lat_tr = pca.transform(X_tr)
            lat_te = pca.transform(X_te)
        prefix = "EFA"
        n_dims = n_factors

    elif source == "ae":
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            np.random.seed(seed)
            n_latent  = config.N_LATENT_DIMS
            n_feat    = X_tr.shape[1]
            device    = "/CPU:0" if config.AE_FORCE_CPU else "/GPU:0"
            with tf.device(device):
                tf.keras.backend.clear_session()
                inp     = tf.keras.Input(shape=(n_feat,))
                encoded = tf.keras.layers.Dense(n_latent, activation="linear",
                                                name="bottleneck")(inp)
                decoded = tf.keras.layers.Dense(n_feat, activation="linear",
                                                name="reconstruction")(encoded)
                ae      = tf.keras.Model(inp, decoded)
                encoder = tf.keras.Model(inp, encoded)
                ae.compile(optimizer=tf.keras.optimizers.Adam(config.AE_LEARNING_RATE),
                           loss="mse")
                cb = tf.keras.callbacks.EarlyStopping(patience=30,
                                                      restore_best_weights=True)
                ae.fit(X_tr, X_tr, epochs=config.AE_EPOCHS, batch_size=64,
                       validation_split=0.2, verbose=0, callbacks=[cb])
                lat_tr = encoder.predict(X_tr, verbose=0)
                lat_te = encoder.predict(X_te, verbose=0)
        except Exception as e:
            log.warning("AE within-split encoding failed (%s) — falling back to PCA", e)
            from sklearn.decomposition import PCA
            n_latent = config.N_LATENT_DIMS
            pca      = PCA(n_components=n_latent, random_state=seed).fit(X_tr)
            lat_tr   = pca.transform(X_tr)
            lat_te   = pca.transform(X_te)
        prefix = "AE"
        n_dims = config.N_LATENT_DIMS

    else:
        raise ValueError(f"Unknown latent source: {source!r} — expected 'pca', 'efa', or 'ae'")

    cols = [f"{prefix}{i+1}" for i in range(n_dims)]
    return (
        pd.DataFrame(lat_tr, columns=cols),
        pd.DataFrame(lat_te, columns=cols),
    )


# =============================================================================
# Alignment: ML latents vs. COR construct scores
# =============================================================================

def _pearson_safe(x: np.ndarray, y: np.ndarray):
    try:
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


def compute_alignment(
    df: pd.DataFrame,
    latent_scores_df: pd.DataFrame,
    method_name: str,
) -> pd.DataFrame:
    """
    Tidy DataFrame of Pearson/Spearman r between each ML latent dim
    and each COR construct score.
    """
    common_idx = df.index.intersection(latent_scores_df.index)
    df_sub     = df.loc[common_idx]
    lat_sub    = latent_scores_df.loc[common_idx]

    rows = []
    for lat_col in latent_scores_df.columns:
        for short, score_col in _CONSTRUCT_SCORES.items():
            if score_col not in df_sub.columns:
                continue
            mask = df_sub[score_col].notna() & lat_sub[lat_col].notna()
            if mask.sum() < 10:
                continue
            x = lat_sub.loc[mask, lat_col].values
            y = df_sub.loc[mask, score_col].values
            r_p, p_p = _pearson_safe(x, y)
            r_s, p_s = spearmanr(x, y)
            rows.append({
                "method":           method_name,
                "latent_dim":       lat_col,
                "construct_short":  short,
                "construct_label":  CONSTRUCT_REGISTRY.get(short, {}).get("label", short),
                "construct_score":  score_col,
                "pearson_r":        round(float(r_p), 4),
                "pearson_p":        round(float(p_p), 4),
                "spearman_r":       round(float(r_s), 4),
                "spearman_p":       round(float(p_s), 4),
                "n":                int(mask.sum()),
                "strong_alignment": bool(abs(r_p) >= 0.40),
            })
    return pd.DataFrame(rows)


def _abs_corr_matrix(df_sub: pd.DataFrame, latent_df: pd.DataFrame) -> np.ndarray:
    """
    (n_latent × n_constructs) matrix of |Pearson r|.  Missing → 0.
    """
    constructs = list(_CONSTRUCT_SCORES.keys())
    mat = np.zeros((latent_df.shape[1], len(constructs)))
    for j, (short, col) in enumerate(_CONSTRUCT_SCORES.items()):
        if col not in df_sub.columns:
            continue
        for i, lat_col in enumerate(latent_df.columns):
            mask = df_sub[col].notna() & latent_df[lat_col].notna()
            if mask.sum() < 10:
                continue
            r, _ = _pearson_safe(latent_df.loc[mask, lat_col].values,
                                  df_sub.loc[mask, col].values)
            mat[i, j] = abs(r) if not np.isnan(r) else 0.0
    return mat


def _hungarian_align(corr_mat: np.ndarray) -> np.ndarray:
    """
    Best 1-to-1 assignment of latent dims → constructs maximising |r|.
    Returns the assigned |r| for each construct column (NaN if unassigned).
    """
    row_ind, col_ind = linear_sum_assignment(-corr_mat)
    assigned = np.full(corr_mat.shape[1], np.nan)
    for r, c in zip(row_ind, col_ind):
        assigned[c] = corr_mat[r, c]
    return assigned


# =============================================================================
# Seed stability (30 runs)
# =============================================================================

def run_seed_stability(
    item_matrix: pd.DataFrame,
    df: pd.DataFrame,
    n_latent: int = None,
    n_seeds: int = None,
    epochs: int = None,
    lr: float = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the autoencoder n_seeds times, align latent dims to COR constructs
    via Hungarian matching on |Pearson r|.

    Returns
    -------
    detail_df   — one row per seed: MSE metrics + aligned |r| per construct
    summary_df  — mean, SD, min, max |r| per construct across seeds
    """
    if n_latent is None: n_latent = config.N_LATENT_DIMS
    if n_seeds  is None: n_seeds  = config.AE_N_SEEDS
    if epochs   is None: epochs   = config.AE_EPOCHS
    if lr       is None: lr       = config.AE_LEARNING_RATE

    constructs  = list(_CONSTRUCT_SCORES.keys())
    common_idx  = df.index.intersection(item_matrix.index)
    df_sub      = df.loc[common_idx]
    records: list[dict] = []

    for seed in range(n_seeds):
        res = run_linear_autoencoder(item_matrix, n_latent=n_latent,
                                     epochs=epochs, lr=lr, seed=seed)
        if not res:
            continue
        lat = res["scores"].loc[res["scores"].index.intersection(df_sub.index)]
        df_aligned = df_sub.loc[lat.index]

        corr_mat = _abs_corr_matrix(df_aligned, lat)
        assigned = _hungarian_align(corr_mat)

        rec = {
            "seed":       seed,
            "recon_mse":  res["recon_mse"],
            "train_mse":  res["train_mse"],
            "val_mse":    res["val_mse"],
            "mse_gap":    res["val_mse"] - res["train_mse"],
        }
        for j, c in enumerate(constructs):
            rec[c] = assigned[j]
        records.append(rec)
        log.info(
            "Seed %2d | MSE=%.5f gap=%.5f | %s",
            seed, res["recon_mse"], rec["mse_gap"],
            "  ".join(f"{c}={rec[c]:.3f}" for c in constructs),
        )

    if not records:
        return pd.DataFrame(), pd.DataFrame()

    detail_df = pd.DataFrame(records)

    summary_rows = []
    for c in constructs:
        vals = detail_df[c].dropna().values
        if len(vals) == 0:
            continue
        summary_rows.append({
            "construct":  c,
            "mean_abs_r": round(float(np.mean(vals)), 4),
            "sd_abs_r":   round(float(np.std(vals, ddof=1)), 4),
            "min_abs_r":  round(float(np.min(vals)), 4),
            "max_abs_r":  round(float(np.max(vals)), 4),
            "n_seeds":    len(vals),
        })

    # Add MSE stability summary rows
    for col in ("recon_mse", "train_mse", "val_mse", "mse_gap"):
        vals = detail_df[col].dropna().values
        summary_rows.append({
            "construct":  col,
            "mean_abs_r": round(float(np.mean(vals)), 5),
            "sd_abs_r":   round(float(np.std(vals, ddof=1)), 5),
            "min_abs_r":  round(float(np.min(vals)), 5),
            "max_abs_r":  round(float(np.max(vals)), 5),
            "n_seeds":    len(vals),
        })

    summary_df = pd.DataFrame(summary_rows)
    log.info("Seed stability summary:\n%s", summary_df.to_string(index=False))
    return detail_df, summary_df


# =============================================================================
# Bottleneck sweep
# =============================================================================

def run_bottleneck_sweep(
    item_matrix: pd.DataFrame,
    df: pd.DataFrame,
    bottleneck_sizes: tuple = None,
    n_seeds: int = None,
    epochs: int = None,
    lr: float = None,
) -> pd.DataFrame:
    """
    Train the autoencoder with different bottleneck sizes, averaging over n_seeds.
    Reports reconstruction MSE and mean Hungarian-aligned |r| per bottleneck size.
    """
    if bottleneck_sizes is None: bottleneck_sizes = config.AE_BOTTLENECK_SIZES
    if n_seeds          is None: n_seeds          = config.AE_SWEEP_SEEDS
    if epochs           is None: epochs           = config.AE_EPOCHS
    if lr               is None: lr               = config.AE_LEARNING_RATE

    common_idx = df.index.intersection(item_matrix.index)
    df_sub     = df.loc[common_idx]
    rows = []

    for n_lat in bottleneck_sizes:
        seed_mses: list[float] = []
        seed_algns: list[float] = []

        for seed in range(n_seeds):
            res = run_linear_autoencoder(item_matrix, n_latent=n_lat,
                                         epochs=epochs, lr=lr, seed=seed)
            if not res:
                continue
            seed_mses.append(res["recon_mse"])
            lat = res["scores"].loc[res["scores"].index.intersection(df_sub.index)]
            df_a = df_sub.loc[lat.index]
            corr_mat = _abs_corr_matrix(df_a, lat)
            assigned = _hungarian_align(corr_mat)
            seed_algns.append(float(np.nanmean(assigned)))

        row = {
            "n_latent":       n_lat,
            "mean_recon_mse": round(np.mean(seed_mses),  5) if seed_mses  else np.nan,
            "sd_recon_mse":   round(np.std(seed_mses, ddof=1),  5) if len(seed_mses)  > 1 else np.nan,
            "mean_alignment": round(np.mean(seed_algns), 4) if seed_algns else np.nan,
            "sd_alignment":   round(np.std(seed_algns, ddof=1), 4) if len(seed_algns) > 1 else np.nan,
        }
        rows.append(row)
        log.info(
            "Bottleneck n=%d | MSE=%.5f±%.5f  align=%.4f±%.4f",
            n_lat,
            row.get("mean_recon_mse", np.nan), row.get("sd_recon_mse", np.nan),
            row.get("mean_alignment",  np.nan), row.get("sd_alignment",  np.nan),
        )

    return pd.DataFrame(rows)


# =============================================================================
# Figures
# =============================================================================

def _plot_alignment_heatmap(alignment_df: pd.DataFrame, method: str) -> None:
    if alignment_df.empty:
        return
    sub = alignment_df[alignment_df["method"] == method]
    if sub.empty:
        return
    pivot  = sub.pivot(index="latent_dim", columns="construct_short", values="pearson_r")
    n_lat  = len(pivot.index)
    n_con  = len(pivot.columns)

    fig, ax = plt.subplots(figsize=(max(5, n_con + 1), max(4, n_lat)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_con));  ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(n_lat));  ax.set_yticklabels(pivot.index)
    for i in range(n_lat):
        for j in range(n_con):
            v = pivot.values[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"Latent alignment: {method} vs. COR construct scores")
    ax.set_xlabel("COR construct")
    ax.set_ylabel("ML latent dimension")
    fig.tight_layout()
    _save_fig(fig, f"latent_alignment_{method.lower().replace(' ', '_')}")


def _plot_pca_scree(evr_df: pd.DataFrame) -> None:
    if evr_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(evr_df["component"], evr_df["explained_variance_ratio"] * 100,
           color="#2980B9", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(evr_df["component"], evr_df["cumulative_evr"] * 100,
             "o-", color="#E74C3C", lw=2)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax2.set_ylabel("Cumulative (%)", color="#E74C3C")
    ax2.axhline(80, color="#E74C3C", ls="--", lw=1)
    ax.set_title("PCA scree plot")
    fig.tight_layout()
    _save_fig(fig, "pca_scree")


def _plot_recon_metrics(ae_res: dict) -> None:
    """Bar chart: train MSE vs val MSE for the default-seed autoencoder."""
    if not ae_res:
        return
    labels = ["Train MSE", "Val MSE", "Overall MSE"]
    values = [ae_res["train_mse"], ae_res["val_mse"], ae_res["recon_mse"]]
    colors = ["#2980B9", "#E74C3C", "#27AE60"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("MSE")
    ax.set_title(f"Linear autoencoder reconstruction loss\n(n_latent={ae_res['n_latent']})")
    gap = ae_res["val_mse"] - ae_res["train_mse"]
    ax.set_xlabel(f"Train/val gap = {gap:+.4f}")
    fig.tight_layout()
    _save_fig(fig, "ae_reconstruction_loss")


def _plot_seed_stability(detail_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Box plot of Hungarian-aligned |r| across 30 seeds per COR construct."""
    if detail_df.empty:
        return
    constructs = [c for c in _CONSTRUCT_SCORES.keys() if c in detail_df.columns]
    if not constructs:
        return

    data = [detail_df[c].dropna().values for c in constructs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: box plot of |r| per construct
    ax = axes[0]
    bp = ax.boxplot(data, labels=constructs, patch_artist=True, notch=False)
    colors = ["#3498DB", "#E67E22", "#2ECC71", "#9B59B6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("|Pearson r| (Hungarian-aligned)")
    ax.set_xlabel("COR construct")
    ax.set_title(f"Seed stability across {len(detail_df)} seeds")
    ax.axhline(0.40, color="red", ls="--", lw=1, label="|r| = 0.40")
    ax.legend(fontsize=8)

    # Right: MSE across seeds
    ax2 = axes[1]
    mse_vals = detail_df["recon_mse"].dropna().values
    ax2.hist(mse_vals, bins=15, color="#2980B9", alpha=0.8, edgecolor="white")
    ax2.axvline(np.mean(mse_vals), color="red", ls="--",
                label=f"mean={np.mean(mse_vals):.4f}")
    ax2.set_xlabel("Overall reconstruction MSE")
    ax2.set_ylabel("Count")
    ax2.set_title("Reconstruction MSE distribution across seeds")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save_fig(fig, "ae_seed_stability")


def _plot_bottleneck_sweep(sweep_df: pd.DataFrame) -> None:
    if sweep_df.empty or sweep_df["mean_recon_mse"].isna().all():
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.errorbar(
        sweep_df["n_latent"], sweep_df["mean_recon_mse"],
        yerr=sweep_df.get("sd_recon_mse", 0),
        fmt="o-", color="#2980B9", capsize=4,
    )
    ax.set_xlabel("Bottleneck size (n_latent)")
    ax.set_ylabel("Mean reconstruction MSE")
    ax.set_title("Reconstruction MSE vs bottleneck size")
    ax.set_xticks(sweep_df["n_latent"])

    ax2 = axes[1]
    ax2.errorbar(
        sweep_df["n_latent"], sweep_df["mean_alignment"],
        yerr=sweep_df.get("sd_alignment", 0),
        fmt="s-", color="#E74C3C", capsize=4,
    )
    ax2.axhline(0.40, color="gray", ls="--", lw=1)
    ax2.set_xlabel("Bottleneck size (n_latent)")
    ax2.set_ylabel("Mean Hungarian-aligned |r|")
    ax2.set_title("COR alignment quality vs bottleneck size")
    ax2.set_xticks(sweep_df["n_latent"])

    fig.tight_layout()
    _save_fig(fig, "ae_bottleneck_sweep")


def _plot_ae_cor_heatmap(alignment_df: pd.DataFrame) -> None:
    """
    Final heatmap: AE dims (rows) × COR constructs (cols).
    Signed Pearson r, highlighting the best-aligned cell per construct.
    """
    sub = alignment_df[alignment_df["method"] == "Linear_AE"]
    if sub.empty:
        return
    pivot = sub.pivot(index="latent_dim", columns="construct_short", values="pearson_r")
    n_lat = len(pivot.index)
    n_con = len(pivot.columns)

    fig, ax = plt.subplots(figsize=(max(5, n_con * 1.3), max(4, n_lat)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    # Highlight best-aligned cell per construct
    best_rows = np.nanargmax(np.abs(pivot.values), axis=0)
    for j, i in enumerate(best_rows):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="black", lw=2.5))

    ax.set_xticks(range(n_con));  ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(n_lat));  ax.set_yticklabels(pivot.index, fontsize=10)
    for i in range(n_lat):
        for j in range(n_con):
            v = pivot.values[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(
        "Linear autoencoder latent dims vs. COR construct scores\n"
        "(best-aligned cell outlined per construct)",
        fontsize=11,
    )
    ax.set_xlabel("COR construct")
    ax.set_ylabel("Autoencoder latent dimension")
    fig.tight_layout()
    _save_fig(fig, "ae_cor_alignment_heatmap")


# =============================================================================
# Main entry point
# =============================================================================

def run(df: pd.DataFrame) -> None:
    """
    Run the full unsupervised latent robustness stream.

    1. Build normalized item matrix
    2. PCA — loadings, EVR, scree, alignment heatmap
    3. EFA — loadings, alignment heatmap
    4. Linear autoencoder (default seed):
         reconstruction loss bar, alignment heatmap, COR correlation matrix
    5. Seed stability (30 seeds): box plots, summary table
    6. Bottleneck sweep (n ∈ {3,4,5}): MSE + alignment vs size
    7. Combined alignment table + best-match summary
    """
    paths.LATENT_TABLES.mkdir(parents=True, exist_ok=True)
    paths.LATENT_FIGURES.mkdir(parents=True, exist_ok=True)

    # ── Item matrix ────────────────────────────────────────────────────────────
    try:
        item_mat, item_names = build_item_matrix(df)
    except ValueError as e:
        log.error("Cannot build item matrix: %s", e)
        return

    if item_mat.shape[0] < 30:
        log.warning("Too few complete-case rows (%d) for latent analysis", item_mat.shape[0])
        return

    all_alignment: list[pd.DataFrame] = []

    # ── PCA ───────────────────────────────────────────────────────────────────
    log.info("Running PCA...")
    pca_res = run_pca(item_mat)
    _save_csv(pca_res["evr"], "pca_explained_variance")
    _save_csv(pca_res["loadings"].reset_index().rename(columns={"index": "variable"}),
              "pca_loadings")
    _plot_pca_scree(pca_res["evr"])

    pca_align = compute_alignment(df.loc[item_mat.index], pca_res["scores"], "PCA")
    all_alignment.append(pca_align)
    _plot_alignment_heatmap(pca_align, "PCA")

    # ── EFA ───────────────────────────────────────────────────────────────────
    log.info("Running EFA...")
    efa_res = run_efa_latent(item_mat)
    _save_csv(efa_res["loadings"].reset_index().rename(columns={"index": "variable"}),
              "efa_loadings")
    efa_align = compute_alignment(df.loc[item_mat.index], efa_res["scores"], "EFA")
    all_alignment.append(efa_align)
    _plot_alignment_heatmap(efa_align, "EFA")

    # ── Linear autoencoder (default seed) ─────────────────────────────────────
    log.info("Running linear autoencoder (seed=%d)...", config.TF_SEED)
    ae_res = run_linear_autoencoder(item_mat)
    if ae_res:
        _plot_recon_metrics(ae_res)

        ae_align = compute_alignment(df.loc[item_mat.index], ae_res["scores"], "Linear_AE")
        all_alignment.append(ae_align)
        _plot_alignment_heatmap(ae_align, "Linear_AE")
        _plot_ae_cor_heatmap(ae_align)

        # Reconstruction metrics table
        recon_df = pd.DataFrame([{
            "seed":       config.TF_SEED,
            "n_latent":   ae_res["n_latent"],
            "recon_mse":  ae_res["recon_mse"],
            "train_mse":  ae_res["train_mse"],
            "val_mse":    ae_res["val_mse"],
            "mse_gap":    ae_res["val_mse"] - ae_res["train_mse"],
        }])
        _save_csv(recon_df, "ae_reconstruction_metrics")

    # ── Seed stability ─────────────────────────────────────────────────────────
    log.info("Running seed stability (%d seeds)...", config.AE_N_SEEDS)
    detail_df, summary_df = run_seed_stability(item_mat, df)
    if not detail_df.empty:
        _save_csv(detail_df,  "ae_seed_stability_detail")
        _save_csv(summary_df, "ae_seed_stability_summary")
        _plot_seed_stability(detail_df, summary_df)

    # ── Bottleneck sweep ───────────────────────────────────────────────────────
    log.info("Running bottleneck sweep %s...", config.AE_BOTTLENECK_SIZES)
    sweep_df = run_bottleneck_sweep(item_mat, df)
    if not sweep_df.empty:
        _save_csv(sweep_df, "ae_bottleneck_sweep")
        _plot_bottleneck_sweep(sweep_df)

    # ── Save per-household latent scores (consumed by ML pipeline) ────────────
    pca_res["scores"].to_csv(paths.LATENT_SCORES_PCA)
    log.info("Saved PCA scores: %s", paths.LATENT_SCORES_PCA.name)
    efa_res["scores"].to_csv(paths.LATENT_SCORES_EFA)
    log.info("Saved EFA scores: %s", paths.LATENT_SCORES_EFA.name)
    if ae_res:
        ae_res["scores"].to_csv(paths.LATENT_SCORES_AE)
        log.info("Saved AE scores: %s", paths.LATENT_SCORES_AE.name)

    # ── Combined alignment table ───────────────────────────────────────────────
    if all_alignment:
        combined = pd.concat(all_alignment, ignore_index=True)
        _save_csv(combined, "unsupervised_latent_alignment")

        best_rows = []
        for short in list(_CONSTRUCT_SCORES.keys()):
            sub = combined[combined["construct_short"] == short]
            if sub.empty:
                continue
            for method in sub["method"].unique():
                m_sub = sub[sub["method"] == method]
                best  = m_sub.loc[m_sub["pearson_r"].abs().idxmax()]
                best_rows.append({
                    "construct":        short,
                    "method":           method,
                    "best_latent":      best["latent_dim"],
                    "pearson_r":        best["pearson_r"],
                    "spearman_r":       best["spearman_r"],
                    "strong_alignment": best["strong_alignment"],
                })
        _save_csv(pd.DataFrame(best_rows), "latent_best_match_summary")
        log.info("Unsupervised latent robustness stream complete.")
