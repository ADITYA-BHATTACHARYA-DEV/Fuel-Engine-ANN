"""
Stage 3: Neural Stacking Ensemble (LightGBM + CatBoost + Neural Network)
Combines gradient boosting models with a neural meta-learner for improved regression accuracy.
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
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
STAGE1_OUTPUT = Path(r"F:\KASHYAP PROJECT\Fuel_Engine\Ammonia\Stage1_Output_new")
STAGE3_OUTPUT = Path(r"F:\KASHYAP PROJECT\Fuel_Engine\Ammonia\Stage3_Neural_Ensemble")
STAGE3_OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# --------------------------------------------------------------------
# METRICS
# --------------------------------------------------------------------
def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100

# --------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# TRAIN LGBM + CATBOOST (BASE MODELS)
# --------------------------------------------------------------------
print("\n🚀 Training base models (LightGBM + CatBoost)...")

pred_train_stack = []
pred_test_stack = []

for i, target in enumerate(tqdm(target_names, desc="Training targets", unit="target")):
    y_tr = Y_train[:, i]
    y_te = Y_test[:, i]

    # LightGBM
    dtrain = lgb.Dataset(X_train, label=y_tr)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": RANDOM_SEED,
        "learning_rate": 0.03,
        "num_leaves": 96,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
    }
    model_lgb = lgb.train(params, dtrain, num_boost_round=1500)
    y_pred_lgb_train = model_lgb.predict(X_train)
    y_pred_lgb_test = model_lgb.predict(X_test)

    # CatBoost
    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5,
        loss_function="RMSE",
        random_seed=RANDOM_SEED,
        verbose=False,
    )
    model_cat.fit(X_train, y_tr)
    y_pred_cat_train = model_cat.predict(X_train)
    y_pred_cat_test = model_cat.predict(X_test)

    # Combine base predictions
    y_pred_train = np.vstack([y_pred_lgb_train, y_pred_cat_train]).T
    y_pred_test = np.vstack([y_pred_lgb_test, y_pred_cat_test]).T

    pred_train_stack.append(y_pred_train)
    pred_test_stack.append(y_pred_test)

# Concatenate across all targets
X_meta_train = np.hstack(pred_train_stack)
X_meta_test = np.hstack(pred_test_stack)

print(f"\n🧠 Meta-training features shape: {X_meta_train.shape}")

# --------------------------------------------------------------------
# NEURAL META-LEARNER
# --------------------------------------------------------------------
print("\n⚡ Training Neural Network meta-learner...")

def build_meta_network(input_dim, output_dim):
    model = Sequential([
        Dense(512, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(output_dim, activation='linear'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mae',
        metrics=['mae']
    )
    return model

meta_model = build_meta_network(X_meta_train.shape[1], Y_train.shape[1])

callbacks = [
    EarlyStopping(patience=30, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-5),
]

meta_model.fit(
    X_meta_train, Y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

# Predictions
Y_pred_nn = meta_model.predict(X_meta_test)

# --------------------------------------------------------------------
# EVALUATION
# --------------------------------------------------------------------
print("\n📊 Evaluating Neural Ensemble...")
results = []
for i, name in enumerate(target_names):
    yt, yp = Y_test[:, i], Y_pred_nn[:, i]
    r2 = r2_score(yt, yp)
    mae = mean_absolute_error(yt, yp)
    s = smape(yt, yp)
    results.append({"Target": name, "R2": r2, "MAE": mae, "SMAPE": s})

results_df = pd.DataFrame(results)
results_df.to_csv(STAGE3_OUTPUT / "neural_ensemble_metrics.csv", index=False)

print("\n🔹 Neural Ensemble Results:")
print(results_df)
print("\nMean R²:", round(results_df["R2"].mean(), 4))
print("Mean MAE:", round(results_df["MAE"].mean(), 4))
print("Mean SMAPE:", round(results_df["SMAPE"].mean(), 2), "%")

# --------------------------------------------------------------------
# VISUALIZATION
# --------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="R2", y="Target", palette="mako")
plt.title("R² per Target (Neural Stacking Ensemble)")
plt.tight_layout()
plt.savefig(STAGE3_OUTPUT / "neural_r2_per_target.png", dpi=300)

print(f"\n✅ Neural Stacking Ensemble completed! Results saved to:\n{STAGE3_OUTPUT}")
