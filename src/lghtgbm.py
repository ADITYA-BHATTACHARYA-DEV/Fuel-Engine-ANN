#!/usr/bin/env python3
"""
stage2_lightgbm_stacking.py
Optimized LightGBM stacking using GRU feature embeddings (Stage 2)

Outputs:
  - Trained LightGBM models (per target)
  - Metrics summary CSV + JSON
  - Feature importance plot
  - Residual distribution plot
  - Optional: Correlation heatmap
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
STAGE1_OUTPUT = Path(r"/Ammonia/Stage1_Output_new")
STAGE2_OUTPUT = Path(r"/Ammonia/Stage2_Output_new")
(STAGE2_OUTPUT / "lgbm_models").mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --------------------------------------------------------------------
# METRICS
# --------------------------------------------------------------------
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100


# --------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------
print("📦 Loading Stage 1 outputs...")
data_path = STAGE1_OUTPUT / "stage1_features.npz"
if not data_path.exists():
    raise FileNotFoundError(f"Stage1 output file not found: {data_path}")

data = np.load(data_path)
X_train = data["X_train_features"]
X_test = data["X_test_features"]
Y_train_scaled = data["Y_train_scaled"]
Y_test_scaled = data["Y_test_scaled"]

target_names = open(STAGE1_OUTPUT / "target_names.txt").read().splitlines()
scaler_Y = joblib.load(STAGE1_OUTPUT / "scaler_Y.joblib")

Y_train = scaler_Y.inverse_transform(Y_train_scaled)
Y_test = scaler_Y.inverse_transform(Y_test_scaled)

print(f"✅ Loaded: X_train={X_train.shape}, X_test={X_test.shape}, Y_train={Y_train.shape}")

# --------------------------------------------------------------------
# TRAIN LIGHTGBM PER TARGET
# --------------------------------------------------------------------
results = []
feature_importance_df = []

print("\n🚀 Training LightGBM models for each target...")

for i, target in enumerate(tqdm(target_names, desc="Training targets", unit="target")):
    y_tr = Y_train[:, i]
    y_te = Y_test[:, i]

    dtrain = lgb.Dataset(X_train, label=y_tr)
    dvalid = lgb.Dataset(X_test, label=y_te, reference=dtrain)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": RANDOM_SEED,
        "learning_rate": 0.02,
        "num_leaves": 96,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "min_data_in_leaf": 10,
    }

    # ✅ Compatible LightGBM training (works for all versions)
    model = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dvalid],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ],
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration or 1000)

    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    s_mape = smape(y_te, y_pred)

    results.append({
        "Target": target,
        "R2": r2,
        "MAE": mae,
        "SMAPE": s_mape,
    })

    model_path = STAGE2_OUTPUT / f"lgbm_models/{target.replace('/', '_')}.txt"
    model.save_model(str(model_path))

    imp_df = pd.DataFrame({
        "feature": [f"F{i}" for i in range(X_train.shape[1])],
        "importance": model.feature_importance(importance_type="gain"),
        "target": target
    })
    feature_importance_df.append(imp_df)

# --------------------------------------------------------------------
# METRICS SUMMARY
# --------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(STAGE2_OUTPUT / "metrics_per_target.csv", index=False)

mean_r2 = results_df["R2"].mean()
mean_mae = results_df["MAE"].mean()
mean_smape = results_df["SMAPE"].mean()

overall_metrics = {
    "Mean_R2": float(mean_r2),
    "Mean_MAE": float(mean_mae),
    "Mean_SMAPE": float(mean_smape),
}
json.dump(overall_metrics, open(STAGE2_OUTPUT / "overall_metrics.json", "w"), indent=4)

print("\n📊 Stage 2 Metrics Summary:")
print(results_df)
print("\n🔹 Mean R²:", round(mean_r2, 4))
print("🔹 Mean MAE:", round(mean_mae, 4))
print("🔹 Mean SMAPE:", round(mean_smape, 2), "%")

# --------------------------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------------------------
print("\n📈 Generating visualizations...")

# --- Feature importance ---
feature_importance_df = pd.concat(feature_importance_df)
mean_imp = (
    feature_importance_df.groupby("feature")["importance"]
    .mean()
    .sort_values(ascending=False)
    .head(30)
)
plt.figure(figsize=(10, 6))
sns.barplot(x=mean_imp.values, y=mean_imp.index, palette="viridis")
plt.title("Top 30 LightGBM Feature Importances")
plt.tight_layout()
plt.savefig(STAGE2_OUTPUT / "lgbm_feature_importances.png", dpi=300)
plt.close()

# --------------------------------------------------------------------
# RESIDUAL DISTRIBUTION
# --------------------------------------------------------------------
print("\n📈 Generating residual distribution plot...")

all_preds, all_true = [], []
valid_targets = []

for i, target in enumerate(target_names):
    model_path = STAGE2_OUTPUT / f"lgbm_models/{target.replace('/', '_')}.txt"
    if not model_path.exists():
        continue

    try:
        model = lgb.Booster(model_file=str(model_path))
        y_pred = model.predict(X_test)
        all_preds.append(y_pred)
        all_true.append(Y_test[:, i])
        valid_targets.append(target)
    except Exception as e:
        print(f"⚠️ Could not load model for {target}: {e}")
        continue

if len(all_preds) == 0:
    print("⚠️ No valid predictions found — skipping residual plot.")
else:
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    residuals = (all_true - all_preds).flatten()

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=40, kde=True)
    plt.title(f"Residual Distribution ({len(valid_targets)} Targets)")
    plt.xlabel("Residual (True - Predicted)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(STAGE2_OUTPUT / "lgbm_residuals.png", dpi=300)
    plt.close()

    print(f"✅ Residual plot saved to: {STAGE2_OUTPUT / 'lgbm_residuals.png'}")

# --------------------------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------------------------
if len(valid_targets) > 1:
    df_preds = pd.DataFrame(np.array(all_preds).T, columns=valid_targets)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_preds.corr(), cmap="coolwarm", annot=False)
    plt.title("Prediction Correlation Across Targets")
    plt.tight_layout()
    plt.savefig(STAGE2_OUTPUT / "lgbm_target_correlation.png", dpi=300)
    plt.close()
    print(f"✅ Target correlation heatmap saved to: {STAGE2_OUTPUT / 'lgbm_target_correlation.png'}")

print("\n✅ Stage 2 completed successfully! Models and metrics saved to:", STAGE2_OUTPUT)
