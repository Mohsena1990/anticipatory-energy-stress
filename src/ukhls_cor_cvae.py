"""
ukhls_cor_cvae.py
───────────────────
Stage 2c — FES-conditioned COR-CVAE: a Conditional Variational Autoencoder
whose 4-dimensional latent space is regularised to align with Stage 2b's
COR-SEM factor scores (OBJECT/CONDITION/PERSONAL/ENERGY), with realised
energy-price stress (FES) as a genuine conditioning variable on both the
encoder and decoder.

  encoder(concat(items, FES)) -> mu, logvar        (n_latent=4)
  z = mu + exp(0.5*logvar) * eps
  decoder(concat(z, FES))     -> reconstruction (items only)
  predictor(z)                -> high_fuel_vulnerable (sigmoid)

  Loss = recon_MSE + beta*KL + lambda*align(mu, SEM factor scores)
                             + gamma*BCE(predictor(z), high_fuel_vulnerable)

Architecture/training loop adapted from `src.cor_vae` (full-batch gradient
descent via a manual `tf.GradientTape` loop, same 4-term ELBO structure);
the only structural change is that FES is concatenated into both the
encoder's and decoder's input instead of being left out.

Item preparation reuses `src.ukhls_cor_sem._standardize` (signed-log for
heavy-tailed monetary columns, then z-score) so the CVAE's item space and
the SEM's are on the same scale -- important for the alignment loss to be
meaningful. Unlike the SEM (which uses FIML to handle structural
missingness natively), the VAE needs a complete numeric input matrix, so
remaining missing items are median-imputed here -- a real, documented
simplification relative to Stage 2b, not swept under the rug.

Counterfactual scenario query
──────────────────────────────
`counterfactual_fes_shift()` re-encodes a household's real item vector
under its own realised FES exposure ("current", `fes_current`) vs. the
shared forecasted FES shock ("forecast", `fes_magnitude` -- see
`src.ukhls_preprocessing.attach_fes_delta`) and reports the shift in
predicted vulnerability probability and the latent Mahalanobis distance
from a "resilient anchor" (mean z of the top-quartile-by-
fuel_to_income_ratio households). This is a SIMULATION on the trained
model, not an observed outcome -- must be reported as such, the same
discipline the project already applies to FES/HighAEV framing generally.

Usage
─────
  from src.ukhls_cor_cvae import run
  result = run(df, sem_scores)   # sem_scores from ukhls_cor_sem.run()['scores']
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src import config, paths
from src.logging_utils import get_logger
from src.ukhls_mapping import COR_FACTOR_ITEMS
from src.ukhls_cor_sem import _standardize

log = get_logger("ukhls_cor_cvae")

_SEM_FACTOR_COLS = ["object_score", "condition_score", "personal_score", "energy_score"]
_Z_COLS = ["z1", "z2", "z3", "z4"]
_PALETTE = {"z1": "#E74C3C", "z2": "#2980B9", "z3": "#E67E22", "z4": "#8E44AD"}


# =============================================================================
# I/O helpers
# =============================================================================

def _save_csv(df: pd.DataFrame, name: str) -> None:
    paths.UKHLS_CVAE_TABLES.mkdir(parents=True, exist_ok=True)
    p = paths.UKHLS_CVAE_TABLES / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p.name)


def _save_fig(fig: plt.Figure, name: str) -> None:
    paths.UKHLS_CVAE_FIGURES.mkdir(parents=True, exist_ok=True)
    p = paths.UKHLS_CVAE_FIGURES / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(p, dpi=config.DPI, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


# =============================================================================
# Data preparation
# =============================================================================

def build_item_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Standardized (signed-log + z-score, same as ukhls_cor_sem) item
    matrix, median-imputed for the VAE's complete-input requirement."""
    all_cols = [c for cols in COR_FACTOR_ITEMS.values() for c in cols]
    all_cols = [c for c in all_cols if c in df.columns]
    std_df = _standardize(df, all_cols)
    item_mat = std_df[all_cols]
    n_missing = int(item_mat.isna().sum().sum())
    if n_missing:
        log.info("Median-imputing %d missing item values for CVAE input "
                  "(SEM handles this natively via FIML; the VAE needs a "
                  "complete numeric matrix)", n_missing)
        item_mat = item_mat.fillna(item_mat.median())
    return item_mat, all_cols


def _prepare_inputs(
    df: pd.DataFrame, sem_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index, dict]:
    """Aligned (item matrix, standardized items X, FES condition, SEM factor
    scores C, HighAEV-analogue label y) over the common valid-target index."""
    from sklearn.preprocessing import StandardScaler

    item_mat, item_cols = build_item_matrix(df)

    # fes_delta (row-varying: FES Magnitude minus this household-wave's own
    # realised exposure) is the conditioning signal -- NOT fes_magnitude,
    # which is constant this run and would produce std=0 -> NaN if scaled.
    # See src.ukhls_preprocessing.attach_fes_delta for the full rationale.
    needed = pd.concat([sem_scores[_SEM_FACTOR_COLS], df["fes_delta"],
                         df["high_fuel_vulnerable"]], axis=1)
    aux = needed.dropna()
    common_idx = item_mat.index.intersection(aux.index)
    item_mat = item_mat.loc[common_idx]
    aux = aux.loc[common_idx]

    item_scaler = StandardScaler().fit(item_mat.values)
    X = item_scaler.transform(item_mat.values).astype(np.float32)

    fes_scaler = StandardScaler().fit(aux[["fes_delta"]].values)
    Fc = fes_scaler.transform(aux[["fes_delta"]].values).astype(np.float32)

    sem_scaler = StandardScaler().fit(aux[_SEM_FACTOR_COLS].values)
    C = sem_scaler.transform(aux[_SEM_FACTOR_COLS].values).astype(np.float32)

    y = aux["high_fuel_vulnerable"].values.astype(np.float32)

    log.info("Stage 2c (CVAE) input: n=%d complete-case rows, %d items", len(item_mat), len(item_cols))
    scalers = {"item": item_scaler, "fes": fes_scaler, "sem": sem_scaler}
    return item_mat, X, Fc, C, y, common_idx, scalers


# =============================================================================
# Model construction (adapted from src.cor_vae._build_vae_components)
# =============================================================================

def _build_cvae_components(n_features: int, n_latent: int, hidden_dims: tuple, seed: int):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)

    inp = tf.keras.Input(shape=(n_features + 1,))  # items + FES condition
    h = inp
    for h_dim in hidden_dims:
        h = tf.keras.layers.Dense(h_dim, activation="relu")(h)
    mu     = tf.keras.layers.Dense(n_latent, name="mu")(h)
    logvar = tf.keras.layers.Dense(n_latent, name="logvar")(h)
    encoder = tf.keras.Model(inp, [mu, logvar], name="cvae_encoder")

    z_in = tf.keras.Input(shape=(n_latent + 1,))  # z + FES condition
    d = z_in
    for h_dim in reversed(hidden_dims):
        d = tf.keras.layers.Dense(h_dim, activation="relu")(d)
    recon = tf.keras.layers.Dense(n_features, activation="linear")(d)
    decoder = tf.keras.Model(z_in, recon, name="cvae_decoder")

    p_in = tf.keras.Input(shape=(n_latent,))
    p = tf.keras.layers.Dense(max(4, n_latent * 2), activation="relu")(p_in)
    pred = tf.keras.layers.Dense(1, activation="sigmoid")(p)
    predictor = tf.keras.Model(p_in, pred, name="cvae_predictor")

    return encoder, decoder, predictor


# =============================================================================
# Training loop (adapted from src.cor_vae._train_vae — full-batch, custom loss)
# =============================================================================

def _train_cvae(
    X: np.ndarray, Fc: np.ndarray, C: np.ndarray, y: np.ndarray,
    n_latent: int, hidden_dims: tuple, epochs: int, lr: float,
    beta: float, lam: float, gamma: float, val_split: float, seed: int,
) -> dict:
    import tensorflow as tf

    device = "/CPU:0" if config.AE_FORCE_CPU else "/GPU:0"
    with tf.device(device):
        np.random.seed(seed)
        n, n_features = X.shape
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_split))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        def _cat(a, b):
            return tf.constant(np.concatenate([a, b], axis=1), dtype=tf.float32)

        XF_t   = _cat(X, Fc)
        XF_tr  = _cat(X[tr_idx], Fc[tr_idx])
        XF_val = _cat(X[val_idx], Fc[val_idx])
        Fc_tr  = tf.constant(Fc[tr_idx], dtype=tf.float32)
        Fc_val = tf.constant(Fc[val_idx], dtype=tf.float32)
        X_tr   = tf.constant(X[tr_idx], dtype=tf.float32)
        X_val  = tf.constant(X[val_idx], dtype=tf.float32)
        C_tr   = tf.constant(C[tr_idx], dtype=tf.float32)
        C_val  = tf.constant(C[val_idx], dtype=tf.float32)
        y_tr   = tf.constant(y[tr_idx].reshape(-1, 1), dtype=tf.float32)
        y_val  = tf.constant(y[val_idx].reshape(-1, 1), dtype=tf.float32)

        encoder, decoder, predictor = _build_cvae_components(n_features, n_latent, hidden_dims, seed)
        trainable = encoder.trainable_variables + decoder.trainable_variables + predictor.trainable_variables
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        def _forward(XFb, Fb, training):
            mu, logvar = encoder(XFb, training=training)
            eps = tf.random.normal(shape=tf.shape(mu))
            z = mu + tf.exp(0.5 * logvar) * eps
            zf = tf.concat([z, Fb], axis=1)
            recon = decoder(zf, training=training)
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

        def _losses(XFb, Fb, Xb, Cb, yb, training):
            mu, logvar, z, recon, pred = _forward(XFb, Fb, training)
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
                total, recon_l, kl_l, align_l, bce_l = _losses(XF_tr, Fc_tr, X_tr, C_tr, y_tr, training=True)
            grads = tape.gradient(total, trainable)
            opt.apply_gradients(zip(grads, trainable))

            v_total, v_recon, v_kl, v_align, v_bce = _losses(XF_val, Fc_val, X_val, C_val, y_val, training=False)
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

        mu_all, logvar_all, z_all, recon_all, pred_all = _forward(XF_t, tf.constant(Fc, dtype=tf.float32), training=False)
        final_total, final_recon, final_kl, final_align, final_bce = _losses(
            XF_t, tf.constant(Fc, dtype=tf.float32),
            tf.constant(X, dtype=tf.float32), tf.constant(C, dtype=tf.float32),
            tf.constant(y.reshape(-1, 1), dtype=tf.float32), training=False,
        )

    return {
        "encoder": encoder, "decoder": decoder, "predictor": predictor,
        "mu": mu_all.numpy(), "logvar": logvar_all.numpy(), "z": z_all.numpy(),
        "recon": recon_all.numpy(), "pred": pred_all.numpy().ravel(),
        "history": pd.DataFrame(history), "n_epochs_trained": len(history),
        "final_recon_mse": float(final_recon), "final_kl": float(final_kl),
        "final_align_loss": float(final_align), "final_bce": float(final_bce),
        "final_total_loss": float(final_total),
        "tr_idx": tr_idx, "val_idx": val_idx,
    }


# =============================================================================
# Counterfactual FES scenario query
# =============================================================================

def counterfactual_fes_shift(
    result: dict, X: np.ndarray, scalers: dict, index: pd.Index,
    fes_raw_current: np.ndarray | float, fes_raw_forecast: float,
    resilient_anchor_z: np.ndarray,
    id_cols: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    For every household's real item vector, re-encode under its own
    realised FES exposure ("current", `fes_raw_current` -- a per-row array
    aligned to `index`, or a scalar broadcast to every row) vs. the shared
    forecasted FES shock ("forecast", `fes_raw_forecast` -- always a
    scalar, since FES Magnitude is constant this run), both scaled through
    the SAME fitted fes_scaler used in training, decode the prediction head
    under each, and report the probability shift + Mahalanobis-style
    |z - resilient_anchor| shift.

    THIS IS A SIMULATION on the trained model, not an observation: no
    household was actually surveyed under both price regimes.
    """
    import tensorflow as tf

    fes_scaler = scalers["fes"]
    n = X.shape[0]
    fes_raw_current = np.broadcast_to(
        np.asarray(fes_raw_current, dtype=np.float64).reshape(-1), (n,)
    ).astype(np.float32)
    Fc_current  = fes_scaler.transform(fes_raw_current.reshape(-1, 1)).astype(np.float32)
    fc_forecast = fes_scaler.transform([[fes_raw_forecast]]).astype(np.float32)[0, 0]
    Fc_forecast = np.full((n, 1), fc_forecast, dtype=np.float32)

    device = "/CPU:0" if config.AE_FORCE_CPU else "/GPU:0"
    with tf.device(device):
        XF_current  = tf.constant(np.concatenate([X, Fc_current],  axis=1), dtype=tf.float32)
        XF_forecast = tf.constant(np.concatenate([X, Fc_forecast], axis=1), dtype=tf.float32)
        mu_current,  _ = result["encoder"](XF_current,  training=False)
        mu_forecast, _ = result["encoder"](XF_forecast, training=False)
        pred_current  = result["predictor"](mu_current,  training=False).numpy().ravel()
        pred_forecast = result["predictor"](mu_forecast, training=False).numpy().ravel()

    mu_current_np, mu_forecast_np = mu_current.numpy(), mu_forecast.numpy()
    dist_current  = np.linalg.norm(mu_current_np  - resilient_anchor_z, axis=1)
    dist_forecast = np.linalg.norm(mu_forecast_np - resilient_anchor_z, axis=1)

    out = pd.DataFrame({
        "row_index": index,
        "pred_prob_current": pred_current, "pred_prob_forecast": pred_forecast,
        "pred_prob_shift": pred_forecast - pred_current,
        "dist_from_resilient_current": dist_current, "dist_from_resilient_forecast": dist_forecast,
        "dist_shift": dist_forecast - dist_current,
    })
    if id_cols is not None:
        out = pd.concat([id_cols.reset_index(drop=True), out], axis=1)
    return out


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
    fig.suptitle("Stage 2c: FES-conditioned CVAE training curves", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, "cvae_training_curves")


def _plot_alignment_heatmap(mu_df: pd.DataFrame, sem_scores: pd.DataFrame) -> None:
    rows = []
    for zc in _Z_COLS:
        for sc in _SEM_FACTOR_COLS:
            common = mu_df[zc].notna() & sem_scores[sc].notna()
            r = np.corrcoef(mu_df.loc[common, zc], sem_scores.loc[common, sc])[0, 1] if common.sum() > 2 else np.nan
            rows.append({"latent_dim": zc, "sem_factor": sc, "pearson_r": r})
    align_df = pd.DataFrame(rows)
    _save_csv(align_df, "cvae_sem_alignment")

    pivot = align_df.pivot(index="latent_dim", columns="sem_factor", values="pearson_r")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.6 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Stage 2c: CVAE latent dims vs. Stage 2b SEM factor scores")
    fig.tight_layout()
    _save_fig(fig, "cvae_sem_alignment_heatmap")
    return align_df


def plot_alignment_polar(align_df: pd.DataFrame) -> None:
    """
    Stacked polar bar chart, the CVAE's own "regression evaluation
    metrics" companion to Stage 1's forecasting-model polar charts: slices
    = the 4 latent dimensions (z1-z4), each stacked with |pearson_r|
    against all 4 SEM factors -- a wide, evenly-stacked slice means that
    latent dimension aligns broadly with the whole COR structure; a slice
    dominated by one segment means it aligns narrowly with a single
    factor. Distinct from `_plot_alignment_heatmap` (a precise per-pair
    number grid) -- this is the at-a-glance composition view.
    """
    if align_df.empty:
        return
    dims = [d for d in _Z_COLS if d in align_df["latent_dim"].unique()]
    n = len(dims)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    angles = np.array([i * 2 * np.pi / n for i in range(n)])
    slice_width = (2 * np.pi / n) * 0.65
    item_cmap = plt.get_cmap("inferno")
    n_factors = len(_SEM_FACTOR_COLS)
    factor_colors = {
        f: matplotlib.colors.to_hex(item_cmap(0.12 + 0.76 * i / max(n_factors - 1, 1)))
        for i, f in enumerate(_SEM_FACTOR_COLS)
    }

    max_total = 0.0
    for angle, dim in zip(angles, dims):
        sub = align_df[align_df["latent_dim"] == dim]
        bottom = 0.0
        for factor in _SEM_FACTOR_COLS:
            row = sub[sub["sem_factor"] == factor]
            if row.empty or pd.isna(row["pearson_r"].iloc[0]):
                continue
            val = abs(float(row["pearson_r"].iloc[0]))
            ax.bar(angle, val, width=slice_width, bottom=bottom, color=factor_colors[factor],
                   edgecolor="white", linewidth=1.2, zorder=3)
            bottom += val
        max_total = max(max_total, bottom)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(dims, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max_total * 1.1 if max_total else 1.0)
    ax.set_rlabel_position(45)  # between z1 and z2 spokes, clear of bars
    ax.tick_params(axis="y", labelsize=8, labelcolor="#555555")
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:.1f}")
    ax.grid(color="#EAECEE", linewidth=0.9, zorder=0)
    ax.spines["polar"].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor=factor_colors[f], edgecolor="white", label=f)
        for f in _SEM_FACTOR_COLS
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.12, 0.5),
              fontsize=9, framealpha=0.95, title="SEM factor (stacked)", title_fontsize=9.5)
    ax.set_title(
        "COR-CVAE: Latent-SEM Alignment Composition\n"
        "(stacked |Pearson r|, taller = broader alignment across all 4 COR factors)",
        fontsize=12, fontweight="bold", pad=20,
    )
    _save_fig(fig, "cvae_alignment_polar")


# =============================================================================
# Main entry point
# =============================================================================

def run(df: pd.DataFrame, sem_scores: pd.DataFrame) -> dict:
    """
    Stage 2c full pipeline: fit the FES-conditioned COR-CVAE, report
    reconstruction/KL/alignment/prediction metrics, and run the
    counterfactual FES scenario query.

    Requires Stage 2b's SEM factor scores (object_score/condition_score/
    personal_score/energy_score) already computed (src.ukhls_cor_sem.run).
    Resilient to failure: writes a diagnostic CSV and returns {} if
    training fails, so the pipeline can continue without Stage 2c.
    """
    paths.UKHLS_CVAE_TABLES.mkdir(parents=True, exist_ok=True)
    paths.UKHLS_CVAE_FIGURES.mkdir(parents=True, exist_ok=True)

    missing = [c for c in _SEM_FACTOR_COLS if c not in sem_scores.columns]
    if missing:
        log.error("Stage 2c (CVAE) requires SEM factor scores %s — run ukhls_cor_sem first.", missing)
        return {}

    item_mat, X, Fc, C, y, common_idx, scalers = _prepare_inputs(df, sem_scores)
    if len(item_mat) < 30:
        log.warning("Too few complete-case rows (%d) for Stage 2c CVAE", len(item_mat))
        return {}

    log.info("Training FES-conditioned COR-CVAE (n=%d, epochs<=%d)...", len(item_mat), config.UKHLS_VAE_EPOCHS)
    try:
        result = _train_cvae(
            X, Fc, C, y,
            n_latent=config.UKHLS_N_LATENT_DIMS, hidden_dims=config.UKHLS_VAE_HIDDEN_DIMS,
            epochs=config.UKHLS_VAE_EPOCHS, lr=config.UKHLS_VAE_LEARNING_RATE,
            beta=config.UKHLS_VAE_BETA, lam=config.UKHLS_VAE_LAMBDA_ALIGN, gamma=config.UKHLS_VAE_GAMMA_PRED,
            val_split=config.UKHLS_VAE_VAL_SPLIT, seed=config.TF_SEED,
        )
    except Exception as e:
        log.error("Stage 2c (CVAE) training failed: %s", e)
        _save_csv(pd.DataFrame([{"stage": "cvae_training", "error": str(e)}]), "cor_cvae_failure_diagnostic")
        return {}

    log.info(
        "CVAE trained (%d epochs) | recon_MSE=%.5f  KL=%.5f  align_loss=%.5f  BCE=%.5f",
        result["n_epochs_trained"], result["final_recon_mse"], result["final_kl"],
        result["final_align_loss"], result["final_bce"],
    )
    _plot_training_curves(result["history"])
    _save_csv(result["history"], "cvae_training_history")

    mu_df = pd.DataFrame(result["mu"], columns=_Z_COLS, index=item_mat.index)
    sem_aligned = sem_scores.loc[item_mat.index, _SEM_FACTOR_COLS].reset_index(drop=True)
    mu_df_reset = mu_df.reset_index(drop=True)
    align_df = _plot_alignment_heatmap(mu_df_reset, sem_aligned)
    plot_alignment_polar(align_df)
    mean_abs_align = align_df["pearson_r"].abs().mean()
    log.info("CVAE-SEM alignment: mean |r| = %.3f (bar: >=0.40)", mean_abs_align)

    # ── Counterfactual FES scenario query: each household's own realised
    # exposure (fes_current) vs. the shared forecasted shock (fes_magnitude)
    # -- see src.ukhls_preprocessing.attach_fes_delta for the rationale. ──
    counterfactual_df = pd.DataFrame()
    if "fes_current" in df.columns and "fes_magnitude" in df.columns and len(item_mat.index) >= 10:
        fes_current_row = df.loc[item_mat.index, "fes_current"].values
        fes_magnitude    = float(df["fes_magnitude"].iloc[0])
        resilient_mask = (df.loc[item_mat.index, "fuel_to_income_ratio"]
                           <= df["fuel_to_income_ratio"].quantile(config.RESILIENT_QUANTILE)).values
        resilient_anchor = result["mu"][resilient_mask].mean(axis=0) if resilient_mask.sum() > 0 else result["mu"].mean(axis=0)
        id_cols = df.loc[item_mat.index, [c for c in ["hidp", "wave", "interview_year", "gor_dv"] if c in df.columns]]
        counterfactual_df = counterfactual_fes_shift(
            result, X, scalers, item_mat.index, fes_current_row, fes_magnitude, resilient_anchor,
            id_cols=id_cols,
        )
        _save_csv(counterfactual_df, "cvae_counterfactual_fes_shift")
        log.info(
            "Counterfactual FES shift (each household's own fes_current -> shared "
            "fes_magnitude=%.3f): mean prob shift=%.4f, mean distance-from-resilient "
            "shift=%.4f -- SIMULATION on the trained model, not an observation.",
            fes_magnitude, counterfactual_df["pred_prob_shift"].mean(), counterfactual_df["dist_shift"].mean(),
        )

    scores_named = mu_df.rename(columns={"z1": "cvae_object_z", "z2": "cvae_condition_z",
                                          "z3": "cvae_personal_z", "z4": "cvae_energy_z"})
    scores_named.to_csv(paths.UKHLS_CVAE_TABLES / "cvae_latent_scores.csv")
    log.info("Saved Stage 2c CVAE latent scores: %d rows", len(scores_named))

    log.info("Stage 2c (FES-conditioned COR-CVAE) complete.")
    return {
        "result": result, "scores": scores_named, "alignment": align_df,
        "counterfactual": counterfactual_df, "scalers": scalers,
    }
