"""
Stage 2 + Stage 3: LightGBM + CatBoost Stacking Ensemble
--------------------------------------------------------
This script:
 - Loads Stage 1 GRU features
 - Trains both LightGBM and CatBoost regressors per target
 - Blends their predictions (ensemble averaging)
 - Evaluates all metrics
 - Generates plots for interpretability
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

# ===============================================================
# CONFIG
# ===============================================================
STAGE1_OUTPUT = Path(r"/Ammonia/Stage1_Output_new")
STAGE2_OUTPUT = Path(r"/Ammonia/Stage2_3_Ensemble_Output")
(STAGE2_OUTPUT / "models_lightgbm").mkdir(parents=True, exist_ok=True)
(STAGE2_OUTPUT / "models_catboost").mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ===============================================================
# METRICS
# ===============================================================
def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100


# ===============================================================
# LOAD DATA
# ===============================================================
print("📦 Loading Stage 1 outputs...")
data = np.load(STAGE1_OUTPUT / "stage1_features.npz")
X_train = data["X_train_features"]
X_test = data["X_test_features"]
Y_train_scaled = data["Y_train_scaled"]
Y_test_scaled = data["Y_test_scaled"]

target_names = open(STAGE1_OUTPUT / "target_names.txt").read().splitlines()
scaler_Y = joblib.load(STAGE1_OUTPUT / "scaler_Y.joblib")

Y_train = scaler_Y.inverse_transform(Y_train_scaled)
Y_test = scaler_Y.inverse_transform(Y_test_scaled)

print(f"✅ Loaded: X_train={X_train.shape}, X_test={X_test.shape}, Y_train={Y_train.shape}")

# ===============================================================
# TRAINING + ENSEMBLE INFERENCE
# ===============================================================
results = []
feature_importances = []

print("\n🚀 Training LightGBM + CatBoost hybrid ensemble per target...")

for i, target in enumerate(tqdm(target_names, desc="Training targets", unit="target")):
    y_tr = Y_train[:, i]
    y_te = Y_test[:, i]

    # ---------------------- LightGBM --------------------------
    dtrain = lgb.Dataset(X_train, label=y_tr)
    dvalid = lgb.Dataset(X_test, label=y_te, reference=dtrain)

    params_lgb = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": RANDOM_SEED,
        "learning_rate": 0.02,
        "num_leaves": 96,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "lambda_l1": 0.2,
        "lambda_l2": 0.3,
        "min_child_samples": 10,
    }

    model_lgb = lgb.train(
        params_lgb,
        dtrain,
        valid_sets=[dtrain, dvalid],
        num_boost_round=2000,
    )

    y_pred_lgb = model_lgb.predict(X_test)

    # ---------------------- CatBoost --------------------------
    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5,
        loss_function="RMSE",
        random_seed=RANDOM_SEED,
        verbose=False
    )
    model_cat.fit(X_train, y_tr, eval_set=(X_test, y_te))

    y_pred_cat = model_cat.predict(X_test)

    # ---------------------- Ensemble --------------------------
    y_pred_ensemble = (y_pred_lgb + y_pred_cat) / 2

    # ---------------------- Metrics --------------------------
    r2 = r2_score(y_te, y_pred_ensemble)
    mae = mean_absolute_error(y_te, y_pred_ensemble)
    s_mape = smape(y_te, y_pred_ensemble)

    results.append({
        "Target": target,
        "R2": r2,
        "MAE": mae,
        "SMAPE": s_mape
    })

    # Save models
    model_lgb.save_model(str(STAGE2_OUTPUT / f"models_lightgbm/{target.replace('/', '_')}.txt"))
    model_cat.save_model(str(STAGE2_OUTPUT / f"models_catboost/{target.replace('/', '_')}.cbm"))

    # Feature importances (from LightGBM)
    imp_df = pd.DataFrame({
        "feature": [f"F{i}" for i in range(X_train.shape[1])],
        "importance": model_lgb.feature_importance(importance_type="gain"),
        "target": target
    })
    feature_importances.append(imp_df)

# ===============================================================
# METRIC SUMMARY
# ===============================================================
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

print("\n📊 Stage 2 + 3 Ensemble Summary:")
print(results_df)
print("\n🔹 Mean R²:", round(mean_r2, 4))
print("🔹 Mean MAE:", round(mean_mae, 4))
print("🔹 Mean SMAPE:", round(mean_smape, 2), "%")

# ===============================================================
# VISUALIZATIONS
# ===============================================================
print("\n📈 Generating visualizations...")

# --- Feature importances ---
feature_importance_df = pd.concat(feature_importances)
mean_imp = (
    feature_importance_df.groupby("feature")["importance"]
    .mean()
    .sort_values(ascending=False)
    .head(30)
)
plt.figure(figsize=(10, 6))
sns.barplot(x=mean_imp.values, y=mean_imp.index, hue=mean_imp.index, legend=False, palette="viridis")
plt.title("Top 30 LightGBM Feature Importances")
plt.tight_layout()
plt.savefig(STAGE2_OUTPUT / "ensemble_feature_importances.png", dpi=300)

# --- R² per target ---
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="R2", y="Target", hue="Target", legend=False, palette="coolwarm")
plt.title("R² per Target - LightGBM + CatBoost Ensemble")
plt.tight_layout()
plt.savefig(STAGE2_OUTPUT / "ensemble_r2_per_target.png", dpi=300)

# --- Correlation Heatmap ---
try:
    preds_all = []
    for target in target_names:
        lgb_model = lgb.Booster(model_file=str(STAGE2_OUTPUT / f"models_lightgbm/{target.replace('/', '_')}.txt"))
        cat_model = CatBoostRegressor()
        cat_model.load_model(str(STAGE2_OUTPUT / f"models_catboost/{target.replace('/', '_')}.cbm"))
        preds_all.append((lgb_model.predict(X_test) + cat_model.predict(X_test)) / 2)

    preds_all = np.array(preds_all).T
    corr = np.corrcoef(preds_all.T)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, xticklabels=target_names, yticklabels=target_names, cmap="coolwarm", annot=False)
    plt.title("Predicted Target Correlation Heatmap (Ensemble)")
    plt.tight_layout()
    plt.savefig(STAGE2_OUTPUT / "ensemble_target_correlation.png", dpi=300)
except Exception as e:
    print(f"⚠️ Could not generate correlation heatmap: {e}")

print(f"\n✅ Stage 2+3 Ensemble completed! Results saved to:\n{STAGE2_OUTPUT}")
