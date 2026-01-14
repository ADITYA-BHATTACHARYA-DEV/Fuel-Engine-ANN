import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import TheilSenRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# -------------------------
# CONFIG & PATHS
# -------------------------
STAGE1_OUTPUT = Path(r"F:\KASHYAP PROJECT\Fuel_Engine\Ammonia\Stage1_Output_new")
STAGE3_OUTPUT = Path(r"F:\KASHYAP PROJECT\Fuel_Engine\Ammonia\Stage3_PINN_Ensemble_Improved")
STAGE3_OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# -------------------------
# HELPERS
# -------------------------
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def smape(y, yhat):
    return np.mean(2*np.abs(y-yhat)/(np.abs(y)+np.abs(yhat)+1e-9))*100

def mae(y, yhat):
    return np.mean(np.abs(y-yhat))

# -------------------------
# LOAD DATA
# -------------------------
data = np.load(STAGE1_OUTPUT / "stage1_features.npz")
X_train, X_test = data["X_train_features"], data["X_test_features"]
Y_train_scaled, Y_test_scaled = data["Y_train_scaled"], data["Y_test_scaled"]
target_names = open(STAGE1_OUTPUT / "target_names.txt").read().splitlines()
scaler_Y = joblib.load(STAGE1_OUTPUT / "scaler_Y.joblib")
num_samples = X_train.shape[0]

print(f"Dataset: {num_samples} samples, Targets: {len(target_names)}")

# -------------------------
# STACKED BASE LEARNERS
# -------------------------
X_meta_train = np.zeros((X_train.shape[0], len(target_names) * 3))
X_meta_test = np.zeros((X_test.shape[0], len(target_names) * 3))

n_splits = min(5, num_samples)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

for i, target in enumerate(target_names):
    y = Y_train_scaled[:, i]
    selector = SelectKBest(score_func=mutual_info_regression, k=min(2, X_train.shape[1]))
    Xtr_reduced = selector.fit_transform(X_train, y)
    Xte_reduced = selector.transform(X_test)

    gpr = GaussianProcessRegressor(kernel=Matern(nu=1.5)+WhiteKernel(noise_level=0.1),
                                   normalize_y=True)
    tsr = TheilSenRegressor()
    nn_base = models.Sequential([
        layers.Input(shape=(Xtr_reduced.shape[1],)),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1)
    ])
    nn_base.compile(optimizer='adam', loss='mse')

    # K-Fold CV for meta-features
    for train_idx, val_idx in kf.split(Xtr_reduced):
        gpr.fit(Xtr_reduced[train_idx], y[train_idx])
        tsr.fit(Xtr_reduced[train_idx], y[train_idx])
        nn_base.fit(Xtr_reduced[train_idx], y[train_idx], epochs=20, verbose=0)
        X_meta_train[val_idx, i*3] = gpr.predict(Xtr_reduced[val_idx])
        X_meta_train[val_idx, i*3+1] = tsr.predict(Xtr_reduced[val_idx])
        X_meta_train[val_idx, i*3+2] = nn_base.predict(Xtr_reduced[val_idx]).flatten()

    # Full training for test predictions
    gpr.fit(Xtr_reduced, y)
    tsr.fit(Xtr_reduced, y)
    nn_base.fit(Xtr_reduced, y, epochs=50, verbose=0)
    X_meta_test[:, i*3] = gpr.predict(Xte_reduced)
    X_meta_test[:, i*3+1] = tsr.predict(Xte_reduced)
    X_meta_test[:, i*3+2] = nn_base.predict(Xte_reduced).flatten()

# -------------------------
# PINN META-LEARNER
# -------------------------
def build_pinn(input_dim, output_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(32, activation='tanh')(inp)
    x = layers.Dense(32, activation='tanh')(x)
    out = layers.Dense(output_dim, activation='linear')(x)
    return models.Model(inp, out)

pinn_meta = build_pinn(X_meta_train.shape[1], Y_train_scaled.shape[1])
optimizer = optimizers.Adam(1e-3)

# Example physics constraint: Torque -> BP relation (replace RPM=1000)
@tf.function
def pinn_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    Torque = y_pred[:, 0:1]
    BP = y_pred[:, 1:2]
    physics_loss = tf.reduce_mean(tf.square(BP - 2*np.pi*Torque*1000/60))
    return mse_loss + 0.1*physics_loss

# TRAIN PINN
epochs = 150
batch_size = min(4, X_meta_train.shape[0])
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = pinn_meta(X_meta_train, training=True)
        loss = pinn_loss(Y_train_scaled, y_pred)
    grads = tape.gradient(loss, pinn_meta.trainable_variables)
    optimizer.apply_gradients(zip(grads, pinn_meta.trainable_variables))
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")

# -------------------------
# PREDICTIONS
# -------------------------
Y_pred_scaled = pinn_meta.predict(X_meta_test)
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
Y_true = scaler_Y.inverse_transform(Y_test_scaled)

# -------------------------
# EVALUATION
# -------------------------
results = []
for i, name in enumerate(target_names):
    yt, yp = Y_true[:, i], Y_pred[:, i]
    results.append({
        "Target": name,
        "MAE": mae(yt, yp),
        "SMAPE_%": smape(yt, yp)
    })

df_results = pd.DataFrame(results)

# -------------------------
# PLOT GRAPHS FOR EACH TARGET
# -------------------------
for i, name in enumerate(target_names):
    yt, yp = Y_true[:, i], Y_pred[:, i]
    plt.figure(figsize=(6,4))
    plt.scatter(yt, yp, color='royalblue', edgecolor='k')
    plt.plot([yt.min(), yt.max()], [yt.min(), yt.max()], 'r--')
    plt.title(f"{name}")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(STAGE3_OUTPUT / f"{sanitize_filename(name)}_graph.png")
    plt.close()

# -------------------------
# PLOT TABLE AS IMAGE
# -------------------------
fig, ax = plt.subplots(figsize=(12, len(df_results)*0.5 + 1))
ax.axis('off')
tbl = ax.table(cellText=df_results.values,
               colLabels=df_results.columns,
               cellLoc='center',
               loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.5)
plt.tight_layout()
plt.savefig(STAGE3_OUTPUT / "results_table.png")
plt.close()

# -------------------------
# SAVE CSV
# -------------------------
df_results.to_csv(STAGE3_OUTPUT / "pinn_ensemble_final_results.csv", index=False)
print("\n✅ Done! Metrics and plots saved in:", STAGE3_OUTPUT)
print(df_results)
