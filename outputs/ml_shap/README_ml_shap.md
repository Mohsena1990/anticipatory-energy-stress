# ML + SHAP Stage — High Thermal Discomfort Prediction

## Overview

This stage uses **CatBoost** gradient boosting to predict whether a UK household
falls in the top quartile of thermal discomfort (`high_discomfort = 1`), based
on COR-derived behavioural scores and household characteristics.

**SHAP (SHapley Additive exPlanations)** then explains which features drive the
prediction for individual households and on average across the test set.

## Target variable

| Variable | Definition |
|----------|-----------|
| `high_discomfort` | 1 if `discomfort_score >= 75th percentile`, else 0 |

Rows with missing `discomfort_score` (18 of 1,015 in the UK sub-sample) are
excluded from training and evaluation.

## Predictors

| Group | Features |
|-------|---------|
| COR composite scores | `insecurity_score`, `preservation_score` |
| Energy poverty | `risk_category` (categorical), `low_income_flag`, `high_cost_flag` |
| Household controls | `S8`, `H1`, `H2`, `H3` |
| Normalised item scores | `n_E2A`, `n_E2B`, `n_E7A-E7E`, `n_H9`, `n_E5A1-E5A5`, `n_E5A9`, `n_E6A1-E6A8` |

**Excluded from ML features:**
- `fes_core / fes_macro / fes_actual` — annual UK-level constants (zero within-sample variance)
- `discomfort_score` — direct target leakage
- `n_H12A, n_H12B, n_H15A-E` — discomfort item scores (target leakage)
- `n_S8` — redundant with raw `S8` already included as control

## Model configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | CatBoostClassifier |
| Iterations | 600 (with early stopping, patience=60) |
| Learning rate | 0.03 |
| Tree depth | 4 |
| Eval metric | AUC |
| Class weights | Balanced by inverse frequency |
| Train/test split | 75 / 25 (stratified, seed=42) |

## Outputs

### Tables

| File | Description |
|------|-------------|
| `ml_feature_list.csv` | Features used with categorical flag |
| `catboost_performance.csv` | Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC |
| `confusion_matrix.csv` | 2x2 test-set confusion matrix |
| `classification_report.csv` | Per-class precision, recall, F1 |
| `shap_feature_importance.csv` | Features ranked by mean |SHAP| |

### Figures

| File | Description |
|------|-------------|
| `shap_beeswarm.png` | SHAP beeswarm — feature impact distribution |
| `shap_bar_importance.png` | Mean |SHAP| bar chart (top 15 features) |
| `shap_dependence_top3.png` | SHAP dependence plots for top-3 features |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `roc_curve.png` | ROC curve with AUC |
| `pr_curve.png` | Precision-Recall curve with Average Precision |
| `predicted_risk_distribution.png` | Probability distribution by true label |

## Interpretation

SHAP values represent each feature's contribution to the log-odds of the
prediction for each household.  A positive SHAP value pushes the model toward
predicting `high_discomfort = 1`; a negative SHAP value pushes toward 0.

The beeswarm plot shows the distribution of SHAP values across households for
each feature, colour-coded by feature value (red = high, blue = low).
