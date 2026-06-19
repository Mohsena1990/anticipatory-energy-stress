"""
unsupervised_latent.py
───────────────────────
Robustness stream: data-driven latent structure vs. theory-driven COR constructs.

Purpose
───────
The COR-derived household vulnerability dimensions are theory-specified.
This stream asks: are those same dimensions recoverable from the empirical
structure of the item-level data, without imposing the COR framework?

Methods
───────
  1. PCA            — linear variance decomposition
  2. EFA            — factor_analyzer / PCA fallback
  3. Linear autoencoder — Keras encoder with linear activations and a
                          narrow (4-neuron) bottleneck

Comparison
──────────
  corr(COR_composite_scores, ML_latent_scores)

  A high Pearson/Spearman correlation between a COR construct score and a
  data-driven latent dimension indicates that the theory-specified construct
  is empirically recoverable from the items.

  Professional note: the unsupervised ML stream does NOT replace COR theory.
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
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from src import config, paths
from src.construct_mapping import ALL_CONSTRUCT_ITEMS, CONSTRUCT_REGISTRY

log = logging.getLogger(__name__)


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
    plt.close(fig)
    log.info("Saved figure %s", p.name)


# =============================================================================
# Item matrix preparation
# =============================================================================

def build_item_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Collect all available normalized item columns (n_{item}) and return a
    matrix with listwise-complete rows.

    Returns (item_matrix, item_names)
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
    """Run PCA on the normalized item matrix."""
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
        "component":           [f"PC{i+1}" for i in range(n_components)],
        "explained_variance":  pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_evr":      np.cumsum(pca.explained_variance_ratio_),
    })

    log.info("PCA: first %d components explain %.1f%% of variance",
             n_components, 100 * pca.explained_variance_ratio_[:n_components].sum())
    return {"scores": scores_df, "loadings": loadings_df, "evr": evr_df, "pca": pca}


# =============================================================================
# EFA
# =============================================================================

def run_efa_latent(item_matrix: pd.DataFrame, n_factors: int = None) -> dict:
    """Run EFA and return factor scores approximated via regression."""
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
        scores,
        index=item_matrix.index,
        columns=[f"EFA{i+1}" for i in range(n_factors)],
    )
    loadings_df = pd.DataFrame(
        loadings,
        index=[c.replace("n_", "") for c in item_matrix.columns],
        columns=[f"EFA{i+1}" for i in range(n_factors)],
    )
    evr_df = pd.DataFrame({
        "factor": [f"EFA{i+1}" for i in range(n_factors)],
        "eigenvalue":       ev[:n_factors],
        "explained_var_ratio": evr,
    })
    log.info("EFA: %d factors, explaining %.1f%% variance",
             n_factors, 100 * evr.sum())
    return {"scores": scores_df, "loadings": loadings_df, "evr": evr_df, "method": "efa"}


# =============================================================================
# Linear autoencoder
# =============================================================================

def run_linear_autoencoder(
    item_matrix: pd.DataFrame,
    n_latent: int = None,
    epochs: int = None,
    lr: float = None,
) -> dict:
    """
    Fit a linear autoencoder (encoder → linear bottleneck → decoder).

    Linear activations are used throughout so that the bottleneck
    dimensions are interpretable as linear projections of the items,
    similar in spirit to PCA but trained end-to-end.

    Returns latent bottleneck scores for all respondents.
    """
    if n_latent is None: n_latent = config.N_LATENT_DIMS
    if epochs is None:   epochs   = config.AE_EPOCHS
    if lr is None:       lr       = config.AE_LEARNING_RATE

    try:
        import tensorflow as tf
        tf.random.set_seed(config.TF_SEED)
    except ImportError:
        log.warning("TensorFlow not available — skipping autoencoder")
        return {}

    from sklearn.preprocessing import StandardScaler

    X_raw = item_matrix.values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    n_features = X.shape[1]

    # Build encoder / decoder with linear activations
    tf.keras.backend.clear_session()
    inp     = tf.keras.Input(shape=(n_features,))
    encoded = tf.keras.layers.Dense(n_latent, activation="linear", name="bottleneck")(inp)
    decoded = tf.keras.layers.Dense(n_features, activation="linear", name="reconstruction")(encoded)

    autoencoder = tf.keras.Model(inp, decoded)
    encoder     = tf.keras.Model(inp, encoded)

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
    )

    autoencoder.fit(
        X, X,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            patience=30, restore_best_weights=True)],
    )

    latent = encoder.predict(X, verbose=0)
    scores_df = pd.DataFrame(
        latent,
        index=item_matrix.index,
        columns=[f"AE{i+1}" for i in range(n_latent)],
    )
    recon    = autoencoder.predict(X, verbose=0)
    recon_mse = float(np.mean((X - recon) ** 2))
    log.info("Linear autoencoder: reconstruction MSE = %.5f", recon_mse)

    return {
        "scores":   scores_df,
        "recon_mse": recon_mse,
        "n_latent": n_latent,
        "method":   "linear_autoencoder",
    }


# =============================================================================
# Alignment: ML latents vs. COR construct scores
# =============================================================================

def compute_alignment(
    df: pd.DataFrame,
    latent_scores_df: pd.DataFrame,
    method_name: str,
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations between each ML latent dimension
    and each COR construct composite score.

    Returns a tidy DataFrame with one row per (latent_dim, construct_score) pair.
    """
    construct_scores = {
        "FCP":  "fcp_score",
        "AEMC": "aemc_score",
        "BLI":  "bli_score",
        "TCR":  "tcr_score",
    }

    # Align indices
    common_idx = df.index.intersection(latent_scores_df.index)
    df_sub     = df.loc[common_idx]
    lat_sub    = latent_scores_df.loc[common_idx]

    rows = []
    for lat_col in latent_scores_df.columns:
        for short, score_col in construct_scores.items():
            if score_col not in df_sub.columns:
                continue
            mask = df_sub[score_col].notna() & lat_sub[lat_col].notna()
            if mask.sum() < 10:
                continue
            x = lat_sub.loc[mask, lat_col].values
            y = df_sub.loc[mask, score_col].values
            r_pearson, p_pearson   = stats_pearson(x, y)
            r_spearman, p_spearman = spearmanr(x, y)
            rows.append({
                "method":          method_name,
                "latent_dim":      lat_col,
                "construct_short": short,
                "construct_label": CONSTRUCT_REGISTRY.get(short, {}).get("label", short),
                "construct_score": score_col,
                "pearson_r":       round(float(r_pearson), 4),
                "pearson_p":       round(float(p_pearson), 4),
                "spearman_r":      round(float(r_spearman), 4),
                "spearman_p":      round(float(p_spearman), 4),
                "n":               int(mask.sum()),
                "strong_alignment": bool(abs(r_pearson) >= 0.40),
            })
    return pd.DataFrame(rows)


def stats_pearson(x: np.ndarray, y: np.ndarray):
    from scipy.stats import pearsonr
    try:
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


# =============================================================================
# Figures
# =============================================================================

def _plot_alignment_heatmap(alignment_df: pd.DataFrame, method: str) -> None:
    if alignment_df.empty:
        return
    sub    = alignment_df[alignment_df["method"] == method]
    if sub.empty:
        return
    pivot  = sub.pivot(index="latent_dim", columns="construct_short", values="pearson_r")
    n_lat  = len(pivot.index)
    n_con  = len(pivot.columns)

    fig, ax = plt.subplots(figsize=(max(5, n_con + 1), max(4, n_lat)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_con))
    ax.set_yticks(range(n_lat))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    for i in range(n_lat):
        for j in range(n_con):
            v = pivot.values[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"Latent alignment: {method} dimensions vs. COR construct scores")
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


# =============================================================================
# Main entry point
# =============================================================================

def run(df: pd.DataFrame) -> None:
    """
    Run the full unsupervised latent robustness stream.

    1. Build normalized item matrix
    2. PCA
    3. EFA
    4. Linear autoencoder
    5. Compare each method's latent dimensions to COR construct scores
    6. Export alignment tables and heatmap figures
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

    all_alignment = []

    # ── PCA ───────────────────────────────────────────────────────────────────
    log.info("Running PCA...")
    pca_res = run_pca(item_mat)
    _save_csv(pca_res["evr"], "pca_explained_variance")
    pca_loads = pca_res["loadings"].reset_index().rename(columns={"index": "variable"})
    _save_csv(pca_loads, "pca_loadings")
    _plot_pca_scree(pca_res["evr"])

    # Align PCA latents to full df (re-predict on all rows with any items)
    full_n_cols = [f"n_{c}" for c in item_names if f"n_{c}" in df.columns]
    # For alignment: use complete-case rows of item_mat only
    pca_align = compute_alignment(
        df.loc[item_mat.index], pca_res["scores"], "PCA"
    )
    all_alignment.append(pca_align)
    _plot_alignment_heatmap(pca_align, "PCA")

    # ── EFA ───────────────────────────────────────────────────────────────────
    log.info("Running EFA...")
    efa_res = run_efa_latent(item_mat)
    _save_csv(efa_res["loadings"].reset_index().rename(columns={"index": "variable"}),
              "efa_loadings")
    efa_align = compute_alignment(
        df.loc[item_mat.index], efa_res["scores"], "EFA"
    )
    all_alignment.append(efa_align)
    _plot_alignment_heatmap(efa_align, "EFA")

    # ── Linear autoencoder ────────────────────────────────────────────────────
    log.info("Running linear autoencoder...")
    ae_res = run_linear_autoencoder(item_mat)
    if ae_res:
        ae_align = compute_alignment(
            df.loc[item_mat.index], ae_res["scores"], "Linear_AE"
        )
        all_alignment.append(ae_align)
        _plot_alignment_heatmap(ae_align, "Linear_AE")

    # ── Combined alignment table ──────────────────────────────────────────────
    if all_alignment:
        combined = pd.concat(all_alignment, ignore_index=True)
        _save_csv(combined, "unsupervised_latent_alignment")

        # Best-match summary: for each construct, which latent dim aligns best?
        best_rows = []
        for short in ["FCP", "AEMC", "BLI", "TCR"]:
            sub = combined[combined["construct_short"] == short]
            if sub.empty:
                continue
            for method in sub["method"].unique():
                m_sub = sub[sub["method"] == method]
                best  = m_sub.loc[m_sub["pearson_r"].abs().idxmax()]
                best_rows.append({
                    "construct":     short,
                    "method":        method,
                    "best_latent":   best["latent_dim"],
                    "pearson_r":     best["pearson_r"],
                    "spearman_r":    best["spearman_r"],
                    "strong_alignment": best["strong_alignment"],
                })
        _save_csv(pd.DataFrame(best_rows), "latent_best_match_summary")
        log.info("Unsupervised latent robustness stream complete.")
