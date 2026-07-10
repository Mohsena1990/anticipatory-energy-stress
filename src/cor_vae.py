"""
cor_vae.py
──────────
Route 3 (COR-Informed VAE Route) — deep generative latent estimation of the
four COR constructs via a theory-informed Variational Autoencoder.

Architecture
────────────
  Raw survey items (n_{item} columns, the same 25-item set used by Routes 1
  and 2, see `src.unsupervised_latent.build_item_matrix`)
      → Encoder (Dense/ReLU, config.VAE_HIDDEN_DIMS) → (mu, logvar)
      → Reparameterize: z = mu + exp(0.5*logvar) * eps,  eps ~ N(0, I)
      → Decoder (mirror of encoder) → reconstruction
      → Prediction head (small MLP): z → HighAEV (sigmoid)

A plain VAE's four latent dimensions are not identifiable with any specific
COR construct. This module forces interpretability via a COR-alignment
penalty against Route 1's composite scores (`src.enable_preprocessing`):
z1≈FCP, z2≈AEMC, z3≈BLI, z4≈TCR.

Loss (all four terms, jointly optimized)
─────────────────────────────────────────
  L = reconstruction_MSE
    + beta   * KL[N(mu,sigma) || N(0,I)]
    + lambda * COR_alignment_loss
    + gamma  * BCE(prediction_head(z), high_aev)

  COR_alignment_loss = mean_i( 1 - |corr(mu_i, cor_composite_i)| ),  i=1..4

Note this is an intentional, documented dependency of Route 3 on Route 1's
composite scores (used only as the alignment target and prediction label,
both computed from the TRAINING split only when used inside
`fit_vae_train_test` — see leakage note there). Training uses full-batch
gradient descent (config.VAE_BATCH_SIZE=0): the sample is small (~500-750
complete cases) and the alignment term's Pearson correlation is only stable
when computed over a large-enough batch, so mini-batching is avoided.

Because a custom four-term joint loss (reconstruction + KL + a correlation-
based alignment term + a supervised prediction term) does not fit into a
single scalar `tf.keras.Model.fit()` loss function cleanly, training uses a
manual `tf.GradientTape` loop (functional-API sub-models for encoder /
decoder / prediction head, same `tf.device`/`AE_FORCE_CPU`/seed conventions
as `src.unsupervised_latent`'s linear autoencoder).

Usage
─────
  from src.cor_vae import run
  run(df)   # df from enable_preprocessing.run(), after Route 1 has already
            # populated fcp_score/aemc_score/bli_score/tcr_score/high_aev
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

warnings.filterwarnings("ignore")

from src import config, paths
from src.unsupervised_latent import build_item_matrix, compute_alignment
from src.route_utils import (
    compute_route_aev,
    route_high_aev,
    sign_canonicalize_against_route1,
    minmax_scale_route_scores,
)

log = logging.getLogger(__name__)

_ROUTE1_COLS = ["fcp_score", "aemc_score", "bli_score", "tcr_score"]
_Z_COLS      = ["z1", "z2", "z3", "z4"]
_PALETTE = {"z1": "#E74C3C", "z2": "#2980B9", "z3": "#E67E22", "z4": "#8E44AD"}

# Persisted/returned column names — Route 3 learns COR-aligned deep latent
# REPRESENTATIONS through a VAE (z1≈FCP, z2≈AEMC, z3≈BLI, z4≈TCR per the
# COR-alignment loss below), as opposed to Route 1's formative composites
# or Route 2's reflective CFA/SEM latent variables.
_RENAME_TO_VAE = {
    "z1": "vae_fcp_latent", "z2": "vae_aemc_latent",
    "z3": "vae_bli_latent", "z4": "vae_tcr_latent",
}
_CONSTRUCT_COLS = ["FCP", "AEMC", "BLI", "TCR"]


# =============================================================================
# I/O helpers
# =============================================================================

def _save_csv(df: pd.DataFrame, name: str) -> None:
    paths.COR_VAE_TABLES.mkdir(parents=True, exist_ok=True)
    p = paths.COR_VAE_TABLES / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p.name)


def _save_fig(fig: plt.Figure, name: str) -> None:
    paths.COR_VAE_FIGURES.mkdir(parents=True, exist_ok=True)
    p = paths.COR_VAE_FIGURES / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(p, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


# =============================================================================
# Model construction
# =============================================================================

def _build_vae_components(n_features: int, n_latent: int, hidden_dims: tuple, seed: int):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)

    inp = tf.keras.Input(shape=(n_features,))
    h = inp
    for h_dim in hidden_dims:
        h = tf.keras.layers.Dense(h_dim, activation="relu")(h)
    mu     = tf.keras.layers.Dense(n_latent, name="mu")(h)
    logvar = tf.keras.layers.Dense(n_latent, name="logvar")(h)
    encoder = tf.keras.Model(inp, [mu, logvar], name="encoder")

    z_in = tf.keras.Input(shape=(n_latent,))
    d = z_in
    for h_dim in reversed(hidden_dims):
        d = tf.keras.layers.Dense(h_dim, activation="relu")(d)
    recon = tf.keras.layers.Dense(n_features, activation="linear")(d)
    decoder = tf.keras.Model(z_in, recon, name="decoder")

    p_in = tf.keras.Input(shape=(n_latent,))
    p = tf.keras.layers.Dense(max(4, n_latent * 2), activation="relu")(p_in)
    pred = tf.keras.layers.Dense(1, activation="sigmoid")(p)
    predictor = tf.keras.Model(p_in, pred, name="predictor")

    return encoder, decoder, predictor


# =============================================================================
# Core training loop (full-batch, custom 4-term loss)
# =============================================================================

def _train_vae(
    X: np.ndarray,
    C: np.ndarray,
    y: np.ndarray,
    n_latent: int,
    hidden_dims: tuple,
    epochs: int,
    lr: float,
    beta: float,
    lam: float,
    gamma: float,
    val_split: float,
    seed: int,
) -> dict:
    """
    Fit the VAE via full-batch gradient descent with early stopping on the
    combined validation loss.

    Parameters
    ----------
    X : (n, n_features) standardized item matrix
    C : (n, 4) standardized COR composite scores, column order [FCP,AEMC,BLI,TCR]
    y : (n,) binary HighAEV labels

    Returns dict with fitted sub-models, mu/logvar/z for all rows, and metrics.
    """
    import tensorflow as tf

    device = "/CPU:0" if config.AE_FORCE_CPU else "/GPU:0"
    with tf.device(device):
        np.random.seed(seed)
        n, n_features = X.shape
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_split))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        X_t   = tf.constant(X, dtype=tf.float32)
        X_tr  = tf.constant(X[tr_idx], dtype=tf.float32)
        X_val = tf.constant(X[val_idx], dtype=tf.float32)
        C_tr  = tf.constant(C[tr_idx], dtype=tf.float32)
        C_val = tf.constant(C[val_idx], dtype=tf.float32)
        y_tr  = tf.constant(y[tr_idx].reshape(-1, 1), dtype=tf.float32)
        y_val = tf.constant(y[val_idx].reshape(-1, 1), dtype=tf.float32)

        encoder, decoder, predictor = _build_vae_components(n_features, n_latent, hidden_dims, seed)
        trainable = encoder.trainable_variables + decoder.trainable_variables + predictor.trainable_variables
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        def _forward(Xb, training):
            mu, logvar = encoder(Xb, training=training)
            eps = tf.random.normal(shape=tf.shape(mu))
            z = mu + tf.exp(0.5 * logvar) * eps
            recon = decoder(z, training=training)
            pred = predictor(z, training=training)
            return mu, logvar, z, recon, pred

        def _align_loss(mu, Cb):
            terms = []
            for i in range(n_latent):
                zi = mu[:, i] - tf.reduce_mean(mu[:, i])
                ci = Cb[:, i] - tf.reduce_mean(Cb[:, i])
                num = tf.reduce_sum(zi * ci)
                den = tf.sqrt(tf.reduce_sum(zi ** 2) * tf.reduce_sum(ci ** 2) + 1e-8)
                terms.append(1.0 - tf.abs(num / den))
            return tf.add_n(terms) / n_latent

        def _losses(Xb, Cb, yb, training):
            mu, logvar, z, recon, pred = _forward(Xb, training)
            recon_loss = tf.reduce_mean(tf.square(Xb - recon))
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
            )
            align = _align_loss(mu, Cb)
            bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(yb, pred))
            total = recon_loss + beta * kl + lam * align + gamma * bce
            return total, recon_loss, kl, align, bce

        best_val = np.inf
        best_weights = None
        patience, wait = 30, 0
        history = []

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                total, recon_l, kl_l, align_l, bce_l = _losses(X_tr, C_tr, y_tr, training=True)
            grads = tape.gradient(total, trainable)
            opt.apply_gradients(zip(grads, trainable))

            v_total, v_recon, v_kl, v_align, v_bce = _losses(X_val, C_val, y_val, training=False)
            history.append({
                "epoch": epoch, "train_total": float(total), "val_total": float(v_total),
                "train_recon": float(recon_l), "val_recon": float(v_recon),
                "train_kl": float(kl_l), "val_kl": float(v_kl),
                "train_align": float(align_l), "val_align": float(v_align),
                "train_bce": float(bce_l), "val_bce": float(v_bce),
            })

            if float(v_total) < best_val - 1e-6:
                best_val = float(v_total)
                best_weights = [w.numpy().copy() for w in trainable]
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_weights is not None:
            for w, bw in zip(trainable, best_weights):
                w.assign(bw)

        mu_all, logvar_all, z_all, recon_all, pred_all = _forward(X_t, training=False)
        final_total, final_recon, final_kl, final_align, final_bce = _losses(
            X_t, tf.constant(C, dtype=tf.float32), tf.constant(y.reshape(-1, 1), dtype=tf.float32),
            training=False,
        )

    return {
        "encoder": encoder, "decoder": decoder, "predictor": predictor,
        "mu": mu_all.numpy(), "logvar": logvar_all.numpy(), "z": z_all.numpy(),
        "recon": recon_all.numpy(), "pred": pred_all.numpy().ravel(),
        "history": pd.DataFrame(history),
        "n_epochs_trained": len(history),
        "final_recon_mse": float(final_recon), "final_kl": float(final_kl),
        "final_align_loss": float(final_align), "final_bce": float(final_bce),
        "final_total_loss": float(final_total),
        "tr_idx": tr_idx, "val_idx": val_idx,
    }


# =============================================================================
# Data preparation
# =============================================================================

def _prepare_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """
    Build the aligned (item matrix, standardized items, standardized COR
    composites, HighAEV label) tuple over the common complete-case index.
    """
    from sklearn.preprocessing import StandardScaler

    item_mat, _ = build_item_matrix(df)
    needed = _ROUTE1_COLS + ["high_aev"]
    aux = df[needed].dropna()
    common_idx = item_mat.index.intersection(aux.index)
    item_mat = item_mat.loc[common_idx]
    aux = aux.loc[common_idx]

    X = StandardScaler().fit_transform(item_mat.values.astype(np.float32))
    C = StandardScaler().fit_transform(aux[_ROUTE1_COLS].values.astype(np.float32))
    y = aux["high_aev"].values.astype(np.float32)

    log.info("Route 3 (VAE) input: n=%d complete-case rows, %d items", len(item_mat), item_mat.shape[1])
    return item_mat, X, C, y, common_idx


# =============================================================================
# Leakage-free train/test VAE fitting (for src.ml_classification)
# =============================================================================

def fit_vae_train_test(
    item_mat_tr: pd.DataFrame,
    item_mat_te: pd.DataFrame,
    cor_scores_tr: pd.DataFrame,
    y_tr: pd.Series,
    seed: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit the VAE on the TRAINING split only (items, Route 1 composite scores,
    and HighAEV labels all restricted to the training rows) then encode both
    splits via the trained encoder's deterministic mean `mu` (not the
    stochastic sample `z`) for stable downstream ML features. The test
    split's composite scores / labels are never used for fitting.

    Returns (scores_tr, scores_te) with columns ["z1","z2","z3","z4"].
    """
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    if seed is None:
        seed = config.TF_SEED

    item_scaler = StandardScaler().fit(item_mat_tr.values)
    X_tr = item_scaler.transform(item_mat_tr.values).astype(np.float32)
    X_te = item_scaler.transform(item_mat_te.values).astype(np.float32)

    cor_scaler = StandardScaler().fit(cor_scores_tr[_ROUTE1_COLS].values)
    C_tr = cor_scaler.transform(cor_scores_tr[_ROUTE1_COLS].values).astype(np.float32)
    y_tr_arr = y_tr.values.astype(np.float32)

    try:
        result = _train_vae(
            X_tr, C_tr, y_tr_arr,
            n_latent=config.N_LATENT_DIMS, hidden_dims=config.VAE_HIDDEN_DIMS,
            epochs=config.VAE_EPOCHS_FAST, lr=config.VAE_LEARNING_RATE,
            beta=config.VAE_BETA, lam=config.VAE_LAMBDA_ALIGN, gamma=config.VAE_GAMMA_PRED,
            val_split=config.VAE_VAL_SPLIT, seed=seed,
        )
    except Exception as e:
        log.warning("cor_vae: within-split VAE fit failed (%s) — latent scores set to NaN", e)
        vae_cols = list(_RENAME_TO_VAE.values())
        return (pd.DataFrame(np.nan, index=item_mat_tr.index, columns=vae_cols),
                pd.DataFrame(np.nan, index=item_mat_te.index, columns=vae_cols))

    device = "/CPU:0" if config.AE_FORCE_CPU else "/GPU:0"
    with tf.device(device):
        mu_tr, _ = result["encoder"](tf.constant(X_tr, dtype=tf.float32), training=False)
        mu_te, _ = result["encoder"](tf.constant(X_te, dtype=tf.float32), training=False)

    scores_tr = pd.DataFrame(mu_tr.numpy(), columns=_Z_COLS, index=item_mat_tr.index).rename(columns=_RENAME_TO_VAE)
    scores_te = pd.DataFrame(mu_te.numpy(), columns=_Z_COLS, index=item_mat_te.index).rename(columns=_RENAME_TO_VAE)
    return scores_tr, scores_te


# =============================================================================
# Seed stability (Hungarian-aligned |r|, mirrors unsupervised_latent's pattern)
# =============================================================================

def _hungarian_align(corr_mat: np.ndarray) -> np.ndarray:
    row_ind, col_ind = linear_sum_assignment(-corr_mat)
    assigned = np.full(corr_mat.shape[1], np.nan)
    for r, c in zip(row_ind, col_ind):
        assigned[c] = corr_mat[r, c]
    return assigned


def run_seed_stability(item_mat: pd.DataFrame, X: np.ndarray, C: np.ndarray, y: np.ndarray,
                        n_seeds: int = None) -> pd.DataFrame:
    if n_seeds is None:
        n_seeds = config.VAE_N_SEEDS
    constructs = ["FCP", "AEMC", "BLI", "TCR"]
    records = []
    for seed in range(n_seeds):
        res = _train_vae(
            X, C, y, n_latent=config.N_LATENT_DIMS, hidden_dims=config.VAE_HIDDEN_DIMS,
            epochs=config.VAE_EPOCHS_FAST, lr=config.VAE_LEARNING_RATE,
            beta=config.VAE_BETA, lam=config.VAE_LAMBDA_ALIGN, gamma=config.VAE_GAMMA_PRED,
            val_split=config.VAE_VAL_SPLIT, seed=seed,
        )
        mu = res["mu"]
        corr_mat = np.zeros((mu.shape[1], len(constructs)))
        for i in range(mu.shape[1]):
            for j in range(len(constructs)):
                r = np.corrcoef(mu[:, i], C[:, j])[0, 1]
                corr_mat[i, j] = abs(r) if np.isfinite(r) else 0.0
        assigned = _hungarian_align(corr_mat)
        rec = {"seed": seed, "recon_mse": res["final_recon_mse"], "kl": res["final_kl"],
               "align_loss": res["final_align_loss"], "bce": res["final_bce"]}
        for k, c in enumerate(constructs):
            rec[c] = assigned[k]
        records.append(rec)
        log.info("VAE seed %2d | recon=%.4f kl=%.4f align=%.4f bce=%.4f | %s",
                  seed, res["final_recon_mse"], res["final_kl"], res["final_align_loss"], res["final_bce"],
                  "  ".join(f"{c}={rec[c]:.3f}" for c in constructs))
    return pd.DataFrame(records)


# =============================================================================
# Prediction metrics + SHAP
# =============================================================================

def _eval_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, average_precision_score,
    )
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision":         round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":            round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":                round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":           round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc":            round(average_precision_score(y_true, y_prob), 4),
    }


def _shap_on_prediction_head(predictor, mu_val: pd.DataFrame, y_val: np.ndarray, y_prob_val: np.ndarray) -> None:
    """SHAP (KernelExplainer, framework-agnostic) on the 4-dim prediction head."""
    try:
        import tensorflow as tf
        import shap
        background = shap.sample(mu_val, min(50, len(mu_val)), random_state=config.RANDOM_SEED)
        device = "/CPU:0" if config.AE_FORCE_CPU else "/GPU:0"

        def _f(z):
            with tf.device(device):
                return predictor(z.astype(np.float32), training=False).numpy().ravel()

        explainer = shap.KernelExplainer(_f, background)
        shap_vals = explainer.shap_values(mu_val.values, nsamples="auto", silent=True)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        from src.shap_explainability import run as run_shap
        run_shap(
            model=None, X_test=mu_val, y_test=pd.Series(y_val), y_prob=y_prob_val,
            cat_idx=[], model_label="route3_vae", precomputed_shap_values=np.asarray(shap_vals),
        )
    except Exception as e:
        log.warning("Route 3 SHAP failed: %s", e)


# =============================================================================
# Figures
# =============================================================================

def _plot_training_curves(history: pd.DataFrame) -> None:
    if history.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [("total", axes[0, 0]), ("recon", axes[0, 1]), ("kl", axes[1, 0]), ("align", axes[1, 1])]
    for key, ax in panels:
        ax.plot(history["epoch"], history[f"train_{key}"], label="train", color="#2980B9")
        ax.plot(history["epoch"], history[f"val_{key}"], label="val", color="#E74C3C", ls="--")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    fig.suptitle("Route 3: VAE training curves (loss components)", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, "vae_training_curves")


def _plot_alignment_heatmap(alignment_df: pd.DataFrame) -> None:
    if alignment_df.empty:
        return
    pivot = alignment_df.pivot(index="latent_dim", columns="construct_short", values="pearson_r")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.6 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Route 3: VAE latent dims vs. Route 1 COR composites")
    fig.tight_layout()
    _save_fig(fig, "vae_cor_alignment_heatmap")


def _plot_seed_stability(detail_df: pd.DataFrame) -> None:
    if detail_df.empty:
        return
    constructs = [c for c in ["FCP", "AEMC", "BLI", "TCR"] if c in detail_df.columns]
    data = [detail_df[c].dropna().values for c in constructs]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, labels=constructs, patch_artist=True)
    for patch, c in zip(bp["boxes"], constructs):
        patch.set_facecolor(_PALETTE.get(f"z{constructs.index(c)+1}", "#999"))
        patch.set_alpha(0.7)
    ax.axhline(0.40, color="red", ls="--", lw=1, label="|r|=0.40")
    ax.set_ylabel("|Pearson r| (Hungarian-aligned)")
    ax.set_title(f"Route 3: VAE seed stability across {len(detail_df)} seeds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, "vae_seed_stability")


def _plot_prediction_curves(y_val: np.ndarray, y_prob: np.ndarray) -> None:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    axes[0].plot(fpr, tpr, color="#2980B9", lw=2, label=f"AUC={auc(fpr, tpr):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("Route 3: prediction head ROC")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_val, y_prob)
    axes[1].plot(rec, prec, color="#27AE60", lw=2, label=f"AP={average_precision_score(y_val, y_prob):.3f}")
    axes[1].set_title("Route 3: prediction head PR curve")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].legend()
    fig.tight_layout()
    _save_fig(fig, "vae_prediction_curves")


# =============================================================================
# Main entry point
# =============================================================================

def run(df: pd.DataFrame) -> dict:
    """
    Route 3 full pipeline: fit the COR-informed VAE on the full complete-case
    sample, report reconstruction/KL/alignment/prediction metrics, seed
    stability, and SHAP on the prediction head. Requires Route 1's
    fcp_score/aemc_score/bli_score/tcr_score/high_aev columns to already
    exist in `df` (i.e. `enable_preprocessing.run()` must have run first).
    """
    paths.COR_VAE_TABLES.mkdir(parents=True, exist_ok=True)
    paths.COR_VAE_FIGURES.mkdir(parents=True, exist_ok=True)

    missing = [c for c in _ROUTE1_COLS + ["high_aev"] if c not in df.columns]
    if missing:
        log.error("Route 3 (VAE) requires Route 1 columns %s — run enable_preprocessing first.", missing)
        return {}

    item_mat, X, C, y, common_idx = _prepare_inputs(df)
    if len(item_mat) < 30:
        log.warning("Too few complete-case rows (%d) for Route 3 VAE", len(item_mat))
        return {}

    log.info("Training COR-informed VAE (seed=%d, epochs<=%d)...", config.TF_SEED, config.VAE_EPOCHS)
    try:
        result = _train_vae(
            X, C, y, n_latent=config.N_LATENT_DIMS, hidden_dims=config.VAE_HIDDEN_DIMS,
            epochs=config.VAE_EPOCHS, lr=config.VAE_LEARNING_RATE,
            beta=config.VAE_BETA, lam=config.VAE_LAMBDA_ALIGN, gamma=config.VAE_GAMMA_PRED,
            val_split=config.VAE_VAL_SPLIT, seed=config.TF_SEED,
        )
    except Exception as e:
        log.error("Route 3 (VAE) training failed: %s — Route 3 unavailable this run.", e)
        _save_csv(pd.DataFrame([{"stage": "vae_training", "error": str(e)}]),
                  "cor_vae_failure_diagnostic")
        return {}
    log.info(
        "VAE trained (%d epochs) | recon_MSE=%.5f  KL=%.5f  align_loss=%.5f  BCE=%.5f",
        result["n_epochs_trained"], result["final_recon_mse"], result["final_kl"],
        result["final_align_loss"], result["final_bce"],
    )
    _plot_training_curves(result["history"])
    _save_csv(result["history"], "vae_training_history")

    # ── Reconstruction metrics ────────────────────────────────────────────
    recon_mae = float(np.mean(np.abs(X - result["recon"])))
    metrics_row = {
        "n_complete_case": len(item_mat), "n_epochs_trained": result["n_epochs_trained"],
        "recon_mse": round(result["final_recon_mse"], 5), "recon_mae": round(recon_mae, 5),
        "kl_divergence": round(result["final_kl"], 5),
        "cor_alignment_loss": round(result["final_align_loss"], 5),
        "prediction_bce": round(result["final_bce"], 5),
    }
    _save_csv(pd.DataFrame([metrics_row]), "vae_reconstruction_metrics")

    # ── COR alignment table (reuses unsupervised_latent.compute_alignment) ──
    mu_df = pd.DataFrame(result["mu"], columns=_Z_COLS, index=item_mat.index)
    alignment_df = compute_alignment(df.loc[item_mat.index], mu_df, "COR_VAE")
    _save_csv(alignment_df, "vae_cor_alignment")
    _plot_alignment_heatmap(alignment_df)

    # ── Prediction-head metrics on the held-out validation split ───────────
    val_idx = result["val_idx"]
    y_val = y[val_idx]
    y_prob_val = result["pred"][val_idx]
    pred_metrics = _eval_metrics(y_val, y_prob_val)
    pred_metrics["n_val"] = int(len(val_idx))
    _save_csv(pd.DataFrame([pred_metrics]), "vae_prediction_metrics")
    log.info("VAE prediction head (val split): %s", pred_metrics)
    _plot_prediction_curves(y_val, y_prob_val)

    # ── Seed stability ──────────────────────────────────────────────────────
    log.info("Running VAE seed stability (%d seeds)...", config.VAE_N_SEEDS)
    stability_df = run_seed_stability(item_mat, X, C, y, n_seeds=config.VAE_N_SEEDS)
    if not stability_df.empty:
        _save_csv(stability_df, "vae_seed_stability")
        _plot_seed_stability(stability_df)

    # ── SHAP on the prediction head (validation rows, z1–z4 latent dims) ────
    mu_val_df = mu_df.iloc[val_idx].reset_index(drop=True)
    _shap_on_prediction_head(result["predictor"], mu_val_df, y_val, y_prob_val)

    # ── vae_fcp_latent / vae_aemc_latent / vae_bli_latent / vae_tcr_latent
    #    + vae_aev_score / vae_high_aev (route-comparison outcome proxy) ─────
    scores_named = mu_df.rename(columns=_RENAME_TO_VAE)
    if all(c in df.columns for c in _ROUTE1_COLS):
        mu_construct = mu_df.rename(columns={"z1": "FCP", "z2": "AEMC", "z3": "BLI", "z4": "TCR"})
        route1 = df.loc[mu_df.index, _ROUTE1_COLS].rename(columns={
            "fcp_score": "FCP", "aemc_score": "AEMC", "bli_score": "BLI", "tcr_score": "TCR",
        })
        canon = sign_canonicalize_against_route1(mu_construct, route1)
        canon01 = minmax_scale_route_scores(canon, _CONSTRUCT_COLS)
        aev = compute_route_aev(canon01, _CONSTRUCT_COLS)
        scores_named["vae_aev_score"] = aev
        scores_named["vae_high_aev"]  = route_high_aev(aev, config.AEV_QUANTILE)
    else:
        log.warning("Route 1 composite scores absent from df — vae_aev_score/vae_high_aev not computed.")

    # ── Save per-household latent scores (consumed by src.ml_classification) ─
    scores_named.to_csv(paths.LATENT_SCORES_ROUTE3_VAE)
    log.info("Saved Route 3 VAE latent scores: %s", paths.LATENT_SCORES_ROUTE3_VAE.name)

    log.info("Route 3 (COR-VAE) stream complete.")
    return {
        "result": result, "metrics": metrics_row, "alignment": alignment_df,
        "prediction_metrics": pred_metrics, "seed_stability": stability_df,
        "scores": mu_df, "scores_named": scores_named,
    }
