# # import os
# # import re
# # import time
# # import sys
# # import joblib
# # import traceback
# # import numpy as np
# # import pandas as pd
# # from tqdm import tqdm
# # import tensorflow as tf
# # from tensorflow.keras.models import Model
# # from tensorflow.keras.layers import Input, GRU, Dense, Dropout, concatenate
# # from tensorflow.keras.preprocessing.sequence import pad_sequences
# # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.model_selection import train_test_split
# #
# # # ----------------------------------------------------------
# # # CONFIGURATION
# # # ----------------------------------------------------------
# # DATA_FOLDER_PATH = r"F:\KASHYAP PROJECT\Fuel_Engine\Ammonia\converted_csvs"
# # SAVE_DIR = r"F:\KASHYAP PROJECT\Fuel_Engine\Ammonia\stage1_output"
# #
# # TIME_SERIES_COLUMNS = [f'Cycle{i}' for i in range(1, 11)]
# # TARGET_COLUMNS = [
# #     'Torque (Nm)', 'I.P (kW)', 'B.P (kW)', 'F.P (kW)', 'I.M.E.P (bar)', 'B.M.E.P (bar)',
# #     'F.M.E.P (bar)', 'Air (mg/stroke)', 'Air (kg/h)', 'Fuel (mg/stroke)', 'Fuel (kg/h)',
# #     'H.S.F (kg/h)', 'S.F.C (kg/kWh)', 'A/F Ratio', 'Vol.Effi. (%)', 'I.T.Effi. (%)',
# #     'B.T.Effi. (%)', 'Mech.Effi. (%)', 'H.b.p (%)', 'H.e.j.c.w (%)', 'H. red (%)', 'H. gas (%)'
# # ]
# #
# # MAX_SEQUENCE_LENGTH = 10
# # EPOCHS = 100
# # BATCH_SIZE = 8
# # RANDOM_SEED = 42
# #
# # np.random.seed(RANDOM_SEED)
# # tf.random.set_seed(RANDOM_SEED)
# #
# # # ----------------------------------------------------------
# # # 1. Load & Pair Data
# # # ----------------------------------------------------------
# # def load_and_pair_data(folder_path):
# #     print("📁 Scanning folders for CSV files...")
# #     start_time = time.time()
# #     all_csvs = []
# #
# #     for root, _, files in os.walk(folder_path):
# #         for f in files:
# #             if f.lower().endswith(".csv"):
# #                 all_csvs.append(os.path.join(root, f))
# #
# #     if not all_csvs:
# #         raise FileNotFoundError(f"No CSV files found in {folder_path}")
# #
# #     print(f"Found {len(all_csvs)} CSV files.\n")
# #
# #     file_info = []
# #     for fpath in tqdm(all_csvs, desc="Classifying files", unit="file"):
# #         try:
# #             df_sample = pd.read_csv(fpath, nrows=3, low_memory=False)
# #             df_sample.columns = [c.strip() for c in df_sample.columns]
# #             cols = set(df_sample.columns)
# #             print(f"🔍 {os.path.basename(fpath)} → {list(cols)}")
# #
# #             ftype = None
# #             tokens = re.findall(r'\d{2,3}', os.path.basename(fpath))
# #             keys = [t.lstrip("0") for t in tokens if t]
# #
# #             if all(c in cols for c in TIME_SERIES_COLUMNS):
# #                 ftype = "time_series"
# #             elif any(c in cols for c in TARGET_COLUMNS):
# #                 ftype = "performance"
# #
# #             if ftype and keys:
# #                 for key in keys:
# #                     file_info.append({"key": key, "type": ftype, "path": fpath})
# #
# #         except Exception as e:
# #             print(f"⚠️ Could not classify {fpath}: {e}")
# #
# #     if not file_info:
# #         raise FileNotFoundError("No valid CSVs found matching expected columns.")
# #
# #     df_info = pd.DataFrame(file_info)
# #     print(f"\nDetected files summary: {df_info['type'].value_counts().to_dict()}")
# #
# #     ts_keys = set(df_info[df_info["type"] == "time_series"]["key"])
# #     perf_keys = set(df_info[df_info["type"] == "performance"]["key"])
# #     common_keys = sorted(ts_keys & perf_keys)
# #     print(f"\nDetected {len(common_keys)} shared load keys.\n")
# #
# #     paired = []
# #     for key in common_keys:
# #         ts_file = df_info[(df_info["key"] == key) & (df_info["type"] == "time_series")].iloc[0]["path"]
# #         perf_file = df_info[(df_info["key"] == key) & (df_info["type"] == "performance")].iloc[0]["path"]
# #         paired.append({"key": key, "x_path": ts_file, "y_path": perf_file})
# #
# #     if not paired:
# #         raise FileNotFoundError("No valid (time-series + performance) pairs found. Check filenames and columns.")
# #
# #     print(f"🔗 Found {len(paired)} valid file pairs. Loading and aligning data...\n")
# #
# #     all_X, all_X_static, all_Y = [], [], []
# #     for p in tqdm(paired, desc="Pairing & loading", unit="pair"):
# #         try:
# #             df_x = pd.read_csv(p["x_path"], low_memory=False)
# #             df_y = pd.read_csv(p["y_path"], low_memory=False)
# #
# #             df_x.columns = [c.strip() for c in df_x.columns]
# #             df_y.columns = [c.strip() for c in df_y.columns]
# #
# #             if not all(c in df_x.columns for c in TIME_SERIES_COLUMNS):
# #                 raise ValueError(f"Missing TS columns in {p['x_path']}")
# #             if not all(c in df_y.columns for c in TARGET_COLUMNS):
# #                 raise ValueError(f"Missing target cols in {p['y_path']}")
# #
# #             seq = df_x[TIME_SERIES_COLUMNS].apply(pd.to_numeric, errors='coerce').values.astype("float32")
# #             target = df_y[TARGET_COLUMNS].iloc[0].apply(pd.to_numeric, errors='coerce').values.astype("float32")
# #
# #             if np.isnan(seq).any() or np.isnan(target).any():
# #                 raise ValueError("NaN values found")
# #
# #             all_X.append(seq)
# #             all_X_static.append([float(p["key"])])
# #             all_Y.append(target)
# #         except Exception as e:
# #             print(f"⚠️ Skipping pair {p['key']}: {e}")
# #
# #     if not all_X:
# #         raise FileNotFoundError("No valid (X, Y) pairs could be loaded.")
# #
# #     X_padded = pad_sequences(all_X, maxlen=MAX_SEQUENCE_LENGTH, dtype="float32", padding="post", truncating="post")
# #     X_static = np.array(all_X_static, dtype="float32")
# #     Y = np.array(all_Y, dtype="float32")
# #
# #     elapsed = time.time() - start_time
# #     print(f"\n✅ Loaded {len(all_X)} valid pairs in {elapsed:.1f} seconds.")
# #     print(f"Shapes — X_seq: {X_padded.shape}, X_static: {X_static.shape}, Y: {Y.shape}\n")
# #
# #     return X_padded, X_static, Y
# #
# # # ----------------------------------------------------------
# # # 2. Scaling
# # # ----------------------------------------------------------
# # def scale_data(X_pad, X_static, Y):
# #     print("🔄 Scaling features and targets...")
# #     scaler_Y = StandardScaler().fit(Y)
# #     Y_scaled = scaler_Y.transform(Y)
# #
# #     scaler_X_static = StandardScaler().fit(X_static)
# #     X_static_scaled = scaler_X_static.transform(X_static)
# #
# #     nsamples, nsteps, nfeatures = X_pad.shape
# #     X_flat = X_pad.reshape((nsamples * nsteps, nfeatures))
# #     scaler_X_seq = StandardScaler().fit(X_flat)
# #     X_scaled = scaler_X_seq.transform(X_flat).reshape((nsamples, nsteps, nfeatures))
# #
# #     print("✅ Scaling complete.\n")
# #     return X_scaled, X_static_scaled, Y_scaled, scaler_X_seq, scaler_X_static, scaler_Y
# #
# # # ----------------------------------------------------------
# # # 3. Build GRU Model
# # # ----------------------------------------------------------
# # def build_models(seq_shape, static_shape, output_shape):
# #     seq_input = Input(shape=seq_shape, name="Time_Series_Input")
# #     static_input = Input(shape=static_shape, name="Static_Input")
# #
# #     x = GRU(64, return_sequences=True)(seq_input)
# #     x = Dropout(0.2)(x)
# #     x = GRU(64, return_sequences=False)(x)
# #     x = Dropout(0.2)(x)
# #
# #     static_out = Dense(16, activation='relu')(static_input)
# #     combined = concatenate([x, static_out])
# #
# #     h = Dense(32, activation='relu')(combined)
# #     h = Dropout(0.2)(h)
# #     h = Dense(16, activation='relu')(h)
# #     output = Dense(output_shape, activation='linear')(h)
# #
# #     model = Model(inputs=[seq_input, static_input], outputs=output)
# #     model.compile(optimizer='adam', loss='mean_squared_error')
# #     return model
# #
# # # ----------------------------------------------------------
# # # 4. Main Execution
# # # ----------------------------------------------------------
# # def main():
# #     try:
# #         X_pad, X_static, Y = load_and_pair_data(DATA_FOLDER_PATH)
# #         X_seq_scaled, X_static_scaled, Y_scaled, s_seq, s_static, s_Y = scale_data(X_pad, X_static, Y)
# #
# #         X_seq_train, X_seq_test, X_static_train, X_static_test, Y_train, Y_test = train_test_split(
# #             X_seq_scaled, X_static_scaled, Y_scaled, test_size=0.2, random_state=RANDOM_SEED
# #         )
# #
# #         model = build_models(
# #             seq_shape=(MAX_SEQUENCE_LENGTH, X_seq_scaled.shape[2]),
# #             static_shape=(X_static_scaled.shape[1],),
# #             output_shape=Y_scaled.shape[1]
# #         )
# #
# #         os.makedirs(SAVE_DIR, exist_ok=True)
# #         ckpt_path = os.path.join(SAVE_DIR, "best_gru_model.keras")
# #         callbacks = [
# #             EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
# #             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1),
# #             ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)
# #         ]
# #
# #         print("\n🚀 Starting GRU training...\n")
# #         start = time.time()
# #         model.fit(
# #             [X_seq_train, X_static_train], Y_train,
# #             validation_data=([X_seq_test, X_static_test], Y_test),
# #             epochs=EPOCHS,
# #             batch_size=BATCH_SIZE,
# #             callbacks=callbacks,
# #             verbose=1
# #         )
# #         print(f"\n✅ Training complete in {(time.time() - start) / 60:.1f} minutes.\n")
# #
# #         # Save trained model and scalers
# #         model.save(os.path.join(SAVE_DIR, "gru_final_model.keras"))
# #         joblib.dump(s_seq, os.path.join(SAVE_DIR, "scaler_X_seq.joblib"))
# #         joblib.dump(s_static, os.path.join(SAVE_DIR, "scaler_X_static.joblib"))
# #         joblib.dump(s_Y, os.path.join(SAVE_DIR, "scaler_Y.joblib"))
# #
# #         print(f"💾 All artifacts saved in {SAVE_DIR}")
# #
# #     except Exception as e:
# #         print("\n❌ ERROR during Stage 1:", type(e).__name__, e)
# #         traceback.print_exc()
# #         sys.exit(1)
# #
# # if __name__ == "__main__":
# #     main()
# #
# #
#
#
# """
# stage1_train_gru_full.py
#
# Stage 1 pipeline:
#  - auto-detect & pair time-series (Cycle1..Cycle10) and performance CSVs
#  - preprocess & scale
#  - train GRU + static branch
#  - save models/scalers
#  - evaluate: R2, MAE, SMAPE
#  - save plots: training history, parity plots, residual histograms, target correlation heatmap
#  - produce confusion-matrix-like heatmaps by binning continuous outputs into quantiles
#  - save predictions vs actuals CSV & metrics JSON
# """
#
# import os
# import re
# import time
# import sys
# import json
# import joblib
# import traceback
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
#
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, GRU, Dense, Dropout, concatenate
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix
#
# # ---------------------------
# # CONFIG
# # ---------------------------
# DATA_FOLDER_PATH = Path(r"F:\KASHYAP PROJECT\Fuel_Engine\Ammonia\converted_csvs")
# SAVE_DIR = Path(r"F:\KASHYAP PROJECT\Fuel_Engine\Ammonia\stage1_output")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)
#
# # Your time-series and target columns (adapted to your last message)
# TIME_SERIES_COLUMNS = [f"Cycle{i}" for i in range(1, 11)]
# TARGET_COLUMNS = [
#     'Torque (Nm)', 'I.P (kW)', 'B.P (kW)', 'F.P (kW)', 'I.M.E.P (bar)', 'B.M.E.P (bar)',
#     'F.M.E.P (bar)', 'Air (mg/stroke)', 'Air (kg/h)', 'Fuel (mg/stroke)', 'Fuel (kg/h)',
#     'H.S.F (kg/h)', 'S.F.C (kg/kWh)', 'A/F Ratio', 'Vol.Effi. (%)', 'I.T.Effi. (%)',
#     'B.T.Effi. (%)', 'Mech.Effi. (%)', 'H.b.p (%)', 'H.e.j.c.w (%)', 'H. red (%)', 'H. gas (%)'
# ]
#
# MAX_SEQUENCE_LENGTH = 10  # because Cycle1..Cycle10
# EPOCHS = 80
# BATCH_SIZE = 8
# RANDOM_SEED = 42
# N_BINS_FOR_CONFUSION = 5  # quantile bins for "confusion matrix" view
#
# np.random.seed(RANDOM_SEED)
# tf.random.set_seed(RANDOM_SEED)
#
# # ---------------------------
# # HELPERS
# # ---------------------------
# def smape(y_true, y_pred):
#     y_true = np.asarray(y_true, dtype=np.float64)
#     y_pred = np.asarray(y_pred, dtype=np.float64)
#     denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
#     mask = denom == 0
#     # avoid division by zero
#     denom = denom[~mask]
#     num = np.abs(y_pred - y_true)[~mask]
#     if denom.size == 0:
#         return 0.0
#     return np.mean(num / denom) * 100.0
#
# def ensure_lower_strip(cols):
#     return [str(c).strip().lower() for c in cols]
#
# def detect_file_type(path, ts_cols_lower, target_cols_lower):
#     """Return 'time_series' or 'performance' or None based on header inspection (case-insensitive)."""
#     try:
#         df = pd.read_csv(path, nrows=3, low_memory=False)
#         cols = ensure_lower_strip(df.columns)
#         cols_set = set(cols)
#         if all(c in cols_set for c in ts_cols_lower):
#             return "time_series"
#         if any(c in cols_set for c in target_cols_lower):
#             return "performance"
#     except Exception:
#         return None
#     return None
#
# # ---------------------------
# # LOAD & PAIR
# # ---------------------------
# def load_and_pair_data(folder_path: Path):
#     print("Scanning folder tree for CSV files...")
#     start = time.time()
#     all_paths = list(folder_path.rglob("*.csv"))
#     if not all_paths:
#         raise FileNotFoundError(f"No CSVs found under {folder_path}")
#
#     ts_cols_lower = [c.lower() for c in TIME_SERIES_COLUMNS]
#     target_cols_lower = [c.lower() for c in TARGET_COLUMNS]
#
#     file_info = []
#     for p in tqdm(all_paths, desc="Classifying files", unit="file"):
#         ftype = detect_file_type(p, ts_cols_lower, target_cols_lower)
#         if ftype:
#             # extract numeric tokens as possible keys
#             basename = p.stem.lower()
#             tokens = re.findall(r'\d{1,3}', basename)
#             # if none, fallback to entire stem as key
#             keys = [t.lstrip("0") or "0" for t in tokens] if tokens else [basename]
#             for k in keys:
#                 file_info.append({"key": str(k), "type": ftype, "path": str(p)})
#
#     if not file_info:
#         raise FileNotFoundError("No files classified as time-series or performance based on headers.")
#
#     df_info = pd.DataFrame(file_info)
#     counts = df_info['type'].value_counts().to_dict()
#     print("Detected:", counts)
#
#     ts_keys = set(df_info[df_info['type'] == 'time_series']['key'])
#     perf_keys = set(df_info[df_info['type'] == 'performance']['key'])
#     common_keys = sorted(list(ts_keys & perf_keys))
#     print(f"Found {len(common_keys)} common keys (pairs).")
#
#     if not common_keys:
#         raise FileNotFoundError("No matching keys between time-series and performance files.")
#
#     X_seqs, X_static, Y_list = [], [], []
#     skipped = []
#     for key in tqdm(common_keys, desc="Pairing loads", unit="key"):
#         try:
#             ts_row = df_info[(df_info['key'] == key) & (df_info['type'] == 'time_series')].iloc[0]
#             perf_row = df_info[(df_info['key'] == key) & (df_info['type'] == 'performance')].iloc[0]
#             x_path = ts_row['path']
#             y_path = perf_row['path']
#
#             df_x = pd.read_csv(x_path, low_memory=False)
#             df_y = pd.read_csv(y_path, low_memory=False)
#
#             # normalize columns to lower+strip
#             df_x.columns = ensure_lower_strip(df_x.columns)
#             df_y.columns = ensure_lower_strip(df_y.columns)
#
#             # pick TS columns present (in order)
#             ts_present = [c for c in ts_cols_lower if c in df_x.columns]
#             tgt_present = [c for c in target_cols_lower if c in df_y.columns]
#
#             if len(ts_present) == 0 or len(tgt_present) == 0:
#                 skipped.append(key)
#                 continue
#
#             seq = df_x[ts_present].apply(pd.to_numeric, errors='coerce').values.astype('float32')
#             # if fewer columns than expected, pad columns with zeros
#             if seq.shape[1] < len(TIME_SERIES_COLUMNS):
#                 pad_width = len(TIME_SERIES_COLUMNS) - seq.shape[1]
#                 seq = np.pad(seq, ((0,0),(0,pad_width)), mode='constant', constant_values=0.0)
#
#             # take mean of perf file rows for target columns (robust to multi-row reports)
#             target_vals = df_y[tgt_present].apply(pd.to_numeric, errors='coerce').mean(axis=0).values.astype('float32')
#
#             # convert nan -> zero (or consider interpolation)
#             seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
#             target_vals = np.nan_to_num(target_vals, nan=0.0, posinf=0.0, neginf=0.0)
#
#             # ensure target vector length equals len(TARGET_COLUMNS) by aligning positions:
#             # create full-length vector where unknown targets are NaN->0
#             full_target = np.zeros(len(TARGET_COLUMNS), dtype='float32')
#             for i, tgt in enumerate(TARGET_COLUMNS):
#                 if tgt.lower() in tgt_present:
#                     j = tgt_present.index(tgt.lower())
#                     full_target[i] = target_vals[j]
#                 else:
#                     full_target[i] = 0.0
#
#             X_seqs.append(seq)
#             # static feature: numeric key if available
#             numeric_key = re.sub(r'\D','', key)
#             static_val = float(numeric_key) if numeric_key else 0.0
#             X_static.append([static_val])
#             Y_list.append(full_target)
#
#         except Exception as e:
#             skipped.append(key)
#             print(f"Skipping key {key}: {e}")
#
#     if len(X_seqs) == 0:
#         raise FileNotFoundError("No valid X/Y pairs loaded after checking files. Skipped keys: " + str(skipped))
#
#     # pad sequences to MAX_SEQUENCE_LENGTH (post)
#     X_padded = pad_sequences(X_seqs, maxlen=MAX_SEQUENCE_LENGTH, dtype='float32', padding='post', truncating='post')
#     X_static = np.array(X_static, dtype='float32')
#     Y = np.array(Y_list, dtype='float32')
#
#     print(f"Loaded pairs: {len(X_seqs)} | X_seq shape: {X_padded.shape} | X_static: {X_static.shape} | Y: {Y.shape}")
#     return X_padded, X_static, Y
#
# # ---------------------------
# # SCALING
# # ---------------------------
# def scale_data(X_pad, X_static, Y):
#     scaler_Y = StandardScaler().fit(Y)
#     Y_scaled = scaler_Y.transform(Y)
#
#     scaler_X_static = StandardScaler().fit(X_static)
#     X_static_scaled = scaler_X_static.transform(X_static)
#
#     ns, nsteps, nfeat = X_pad.shape
#     flat = X_pad.reshape((ns * nsteps, nfeat))
#     scaler_X_seq = StandardScaler().fit(flat)
#     flat_scaled = scaler_X_seq.transform(flat)
#     X_scaled = flat_scaled.reshape((ns, nsteps, nfeat))
#
#     return X_scaled, X_static_scaled, Y_scaled, scaler_X_seq, scaler_X_static, scaler_Y
#
# # ---------------------------
# # MODEL
# # ---------------------------
# def build_model(seq_shape, static_shape, output_shape):
#     seq_input = Input(shape=seq_shape, name="seq_in")
#     static_input = Input(shape=static_shape, name="static_in")
#
#     x = GRU(64, return_sequences=True)(seq_input)
#     x = Dropout(0.25)(x)
#     x = GRU(64)(x)
#     x = Dropout(0.25)(x)
#
#     s = Dense(16, activation='relu')(static_input)
#     merged = concatenate([x, s])
#
#     h = Dense(64, activation='relu')(merged)
#     h = Dropout(0.25)(h)
#     out = Dense(output_shape, activation='linear')(h)
#
#     model = Model([seq_input, static_input], out)
#     model.compile(optimizer='adam', loss='mse')
#     return model
#
# # ---------------------------
# # PLOTTING & METRICS
# # ---------------------------
# def plot_history(history, out_dir: Path):
#     fig, ax = plt.subplots(1,1, figsize=(8,5))
#     ax.plot(history.history.get('loss', []), label='train_loss')
#     ax.plot(history.history.get('val_loss', []), label='val_loss')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Loss (MSE)')
#     ax.legend()
#     ax.grid(True)
#     fig.savefig(out_dir / "training_history.png", bbox_inches='tight')
#     plt.close(fig)
#
# def parity_and_residual_plots(Y_true, Y_pred, target_names, out_dir: Path):
#     n = len(target_names)
#     cols = min(3, n)
#     rows = int(np.ceil(n/cols))
#     fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
#     axes = axes.flatten()
#     metrics = {}
#     for i, name in enumerate(target_names):
#         ax = axes[i]
#         yt = Y_true[:, i]
#         yp = Y_pred[:, i]
#         ax.scatter(yt, yp, s=10, alpha=0.6)
#         mn = min(yt.min(), yp.min())
#         mx = max(yt.max(), yp.max())
#         ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
#         ax.set_title(name)
#         ax.set_xlabel('Actual')
#         ax.set_ylabel('Predicted')
#         ax.grid(True)
#         # metrics per target
#         r2 = r2_score(yt, yp)
#         mae = mean_absolute_error(yt, yp)
#         sm = smape(yt, yp)
#         metrics[name] = {'r2': float(r2), 'mae': float(mae), 'smape_pct': float(sm)}
#     # remove unused axes
#     for j in range(n, len(axes)):
#         fig.delaxes(axes[j])
#     fig.tight_layout()
#     fig.savefig(out_dir / "parity_plots.png", bbox_inches='tight')
#     plt.close(fig)
#     # Residual histogram
#     fig2, ax2 = plt.subplots(figsize=(8,5))
#     residuals = (Y_pred - Y_true).ravel()
#     ax2.hist(residuals, bins=60)
#     ax2.set_title('Residuals histogram (all targets flattened)')
#     ax2.set_xlabel('Residual (pred - true)')
#     ax2.set_ylabel('Count')
#     fig2.savefig(out_dir / "residuals_hist.png", bbox_inches='tight')
#     plt.close(fig2)
#     return metrics
#
# def target_corr_heatmap(Y_true, out_dir: Path, target_names):
#     df = pd.DataFrame(Y_true, columns=target_names)
#     corr = df.corr()
#     fig, ax = plt.subplots(figsize=(10,8))
#     sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
#     ax.set_title('Target correlation matrix')
#     fig.savefig(out_dir / "target_correlation.png", bbox_inches='tight')
#     plt.close(fig)
#
# def binned_confusion_heatmaps(Y_true, Y_pred, target_names, out_dir: Path, n_bins=5):
#     os.makedirs(out_dir, exist_ok=True)
#     cms = {}
#     for i, name in enumerate(target_names):
#         yt = Y_true[:, i]
#         yp = Y_pred[:, i]
#         # bin into quantiles
#         try:
#             bins = np.quantile(yt, np.linspace(0,1,n_bins+1))
#             # ensure strictly increasing
#             bins = np.unique(bins)
#             if bins.size <= 1:
#                 continue
#             yb = np.digitize(yt, bins, right=False) - 1
#             pb = np.digitize(yp, bins, right=False) - 1
#             # clip
#             yb = np.clip(yb, 0, n_bins-1)
#             pb = np.clip(pb, 0, n_bins-1)
#             cm = confusion_matrix(yb, pb, labels=list(range(n_bins)))
#             cms[name] = cm
#             # plot heatmap
#             fig, ax = plt.subplots(figsize=(5,4))
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
#             ax.set_xlabel('Predicted bin')
#             ax.set_ylabel('Actual bin')
#             ax.set_title(f'Binned confusion (quantiles) - {name}')
#             fig.savefig(out_dir / f"confusion_binned_{i}_{name.replace(' ','_').replace('/','_')}.png", bbox_inches='tight')
#             plt.close(fig)
#         except Exception as e:
#             print(f"Could not compute binned confusion for {name}: {e}")
#     return cms
#
# # ---------------------------
# # MAIN
# # ---------------------------
# def main():
#     try:
#         X_pad, X_static, Y = load_and_pair_data(DATA_FOLDER_PATH)
#
#         X_seq_scaled, X_static_scaled, Y_scaled, scaler_X_seq, scaler_X_static, scaler_Y = scale_data(X_pad, X_static, Y)
#
#         # split
#         X_seq_tr, X_seq_te, X_static_tr, X_static_te, Y_tr, Y_te = train_test_split(
#             X_seq_scaled, X_static_scaled, Y_scaled, test_size=0.2, random_state=RANDOM_SEED
#         )
#
#         model = build_model(seq_shape=(MAX_SEQUENCE_LENGTH, X_seq_scaled.shape[2]),
#                             static_shape=(X_static_scaled.shape[1],),
#                             output_shape=Y_scaled.shape[1])
#
#         SAVE_DIR.mkdir(parents=True, exist_ok=True)
#         ckpt = SAVE_DIR / "best_gru_model.keras"
#         callbacks = [
#             EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
#             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1),
#             ModelCheckpoint(str(ckpt), monitor='val_loss', save_best_only=True, verbose=1)
#         ]
#
#         print("\nTraining model...")
#         history = model.fit([X_seq_tr, X_static_tr], Y_tr,
#                             validation_data=([X_seq_te, X_static_te], Y_te),
#                             epochs=EPOCHS, batch_size=BATCH_SIZE,
#                             callbacks=callbacks, verbose=1)
#
#         # save models & scalers
#         model.save(SAVE_DIR / "gru_final_model.keras")
#         joblib.dump(scaler_X_seq, SAVE_DIR / "scaler_X_seq.joblib")
#         joblib.dump(scaler_X_static, SAVE_DIR / "scaler_X_static.joblib")
#         joblib.dump(scaler_Y, SAVE_DIR / "scaler_Y.joblib")
#
#         # extract features (optional)
#         # create feature extractor that outputs merged hidden representation
#         # We'll rebuild small extractor that shares model weights up to penultimate Dense:
#         # For simplicity, just use model.predict to get predictions and evaluate
#
#         # Predictions
#         Y_pred_scaled = model.predict([X_seq_te, X_static_te], batch_size=BATCH_SIZE)
#         Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
#         Y_true = scaler_Y.inverse_transform(Y_te)
#
#         # Save predictions vs actuals
#         df_preds = pd.DataFrame(Y_pred, columns=TARGET_COLUMNS)
#         df_true = pd.DataFrame(Y_true, columns=TARGET_COLUMNS)
#         df_comb = pd.concat([df_true.add_prefix("actual_"), df_preds.add_prefix("pred_")], axis=1)
#         df_comb.to_csv(SAVE_DIR / "predictions_vs_actuals.csv", index=False)
#
#         # Metrics (per-target + overall)
#         metrics = {}
#         overall_r2 = r2_score(Y_true, Y_pred, multioutput='uniform_average')
#         overall_mae = mean_absolute_error(Y_true, Y_pred)
#         overall_smape = smape(Y_true, Y_pred)
#         metrics['overall'] = {'r2': float(overall_r2), 'mae': float(overall_mae), 'smape_pct': float(overall_smape)}
#
#         per_target = {}
#         for i, name in enumerate(TARGET_COLUMNS):
#             yt = Y_true[:, i]
#             yp = Y_pred[:, i]
#             per_target[name] = {
#                 'r2': float(r2_score(yt, yp)) if len(yt) > 1 else None,
#                 'mae': float(mean_absolute_error(yt, yp)),
#                 'smape_pct': float(smape(yt, yp))
#             }
#         metrics['by_target'] = per_target
#
#         # Save metrics
#         with open(SAVE_DIR / "metrics.json", "w") as fh:
#             json.dump(metrics, fh, indent=2)
#
#         # Plots
#         plot_history(history, SAVE_DIR)
#         per_target_metrics = parity_and_residual_plots(Y_true, Y_pred, TARGET_COLUMNS, SAVE_DIR)
#         target_corr_heatmap(Y_true, SAVE_DIR, TARGET_COLUMNS)
#         cms = binned_confusion_heatmaps(Y_true, Y_pred, TARGET_COLUMNS, SAVE_DIR, n_bins=N_BINS_FOR_CONFUSION)
#
#         print("\n--- RESULTS SUMMARY ---")
#         print(f"Saved model: {SAVE_DIR / 'gru_final_model.keras'}")
#         print(f"Saved scalers: scaler_X_seq.joblib, scaler_X_static.joblib, scaler_Y.joblib")
#         print(f"Saved preds CSV: {SAVE_DIR / 'predictions_vs_actuals.csv'}")
#         print(f"Saved metrics JSON: {SAVE_DIR / 'metrics.json'}")
#         print(f"Saved plots: training_history.png, parity_plots.png, residuals_hist.png, target_correlation.png")
#         print(f"Saved binned confusion heatmaps: {len(cms)} (one per target where available)")
#
#         # Print short metrics to console
#         print("\nOverall metrics:")
#         print(f"  R2 (macro): {metrics['overall']['r2']:.4f}")
#         print(f"  MAE   (macro): {metrics['overall']['mae']:.4f}")
#         print(f"  SMAPE (%) : {metrics['overall']['smape_pct']:.4f}")
#
#         print("\nPer-target sample (first 3):")
#         for t in list(per_target.keys())[:3]:
#             print(f" - {t}: R2={per_target[t]['r2']}, MAE={per_target[t]['mae']:.4f}, SMAPE%={per_target[t]['smape_pct']:.4f}")
#
#     except Exception as e:
#         print("\nERROR:", type(e).__name__, e)
#         traceback.print_exc()
#         sys.exit(1)
#
# if __name__ == "__main__":
#     main()









"""
stage1_train_gru_full_improved.py
Improved GRU + feature extraction for LightGBM stacking (Stage 2)
"""

import os
import re
import time
import sys
import json
import joblib
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, concatenate, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
DATA_FOLDER_PATH = Path(r"/Ammonia/converted_csvs")
SAVE_DIR = Path(r"/Ammonia/Stage1_Output_new")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TIME_SERIES_COLUMNS = [f"Cycle{i}" for i in range(1, 11)]
TARGET_COLUMNS = [
    'Torque (Nm)', 'I.P (kW)', 'B.P (kW)', 'F.P (kW)', 'I.M.E.P (bar)', 'B.M.E.P (bar)',
    'F.M.E.P (bar)', 'Air (mg/stroke)', 'Air (kg/h)', 'Fuel (mg/stroke)', 'Fuel (kg/h)',
    'H.S.F (kg/h)', 'S.F.C (kg/kWh)', 'A/F Ratio', 'Vol.Effi. (%)', 'I.T.Effi. (%)',
    'B.T.Effi. (%)', 'Mech.Effi. (%)', 'H.b.p (%)', 'H.e.j.c.w (%)', 'H. red (%)', 'H. gas (%)'
]

MAX_SEQUENCE_LENGTH = 10
EPOCHS = 120
BATCH_SIZE = 16
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom == 0
    denom = denom[~mask]
    num = np.abs(y_pred - y_true)[~mask]
    return np.mean(num / denom) * 100.0 if denom.size else 0.0


def ensure_lower_strip(cols):
    return [str(c).strip().lower() for c in cols]


def detect_file_type(path, ts_cols_lower, target_cols_lower):
    """Detect if CSV is time-series or performance file."""
    try:
        df = pd.read_csv(path, nrows=3, low_memory=False)
        cols = ensure_lower_strip(df.columns)
        cols_set = set(cols)
        if all(c in cols_set for c in ts_cols_lower):
            return "time_series"
        if any(c in cols_set for c in target_cols_lower):
            return "performance"
    except Exception:
        return None
    return None

# -------------------------------------------------------------------
# LOAD & PAIR
# -------------------------------------------------------------------
def load_and_pair_data(folder_path: Path):
    print("\n📁 Scanning folder tree for CSV files...")
    all_paths = list(folder_path.rglob("*.csv"))
    if not all_paths:
        raise FileNotFoundError(f"No CSVs found under {folder_path}")

    ts_cols_lower = [c.lower() for c in TIME_SERIES_COLUMNS]
    target_cols_lower = [c.lower() for c in TARGET_COLUMNS]

    file_info = []
    for p in tqdm(all_paths, desc="🔍 Classifying files", unit="file"):
        ftype = detect_file_type(p, ts_cols_lower, target_cols_lower)
        if ftype:
            basename = p.stem.lower()
            tokens = re.findall(r'\d{1,3}', basename)
            keys = [t.lstrip("0") or "0" for t in tokens] if tokens else [basename]
            for k in keys:
                file_info.append({"key": str(k), "type": ftype, "path": str(p)})

    if not file_info:
        raise FileNotFoundError("❌ No valid CSVs detected as time-series or performance.")

    df_info = pd.DataFrame(file_info)
    print("✅ Detected files summary:", df_info['type'].value_counts().to_dict())

    ts_keys = set(df_info[df_info['type'] == 'time_series']['key'])
    perf_keys = set(df_info[df_info['type'] == 'performance']['key'])
    common_keys = sorted(list(ts_keys & perf_keys))
    print(f"🔗 Found {len(common_keys)} matching keys (time-series ↔ performance)")

    X_seqs, X_static, Y_list = [], [], []
    for key in tqdm(common_keys, desc="📦 Pairing loads", unit="pair"):
        try:
            ts_row = df_info[(df_info['key'] == key) & (df_info['type'] == 'time_series')].iloc[0]
            perf_row = df_info[(df_info['key'] == key) & (df_info['type'] == 'performance')].iloc[0]
            df_x = pd.read_csv(ts_row['path'], low_memory=False)
            df_y = pd.read_csv(perf_row['path'], low_memory=False)

            df_x.columns = ensure_lower_strip(df_x.columns)
            df_y.columns = ensure_lower_strip(df_y.columns)

            ts_present = [c for c in ts_cols_lower if c in df_x.columns]
            tgt_present = [c for c in target_cols_lower if c in df_y.columns]
            if not ts_present or not tgt_present:
                continue

            seq = df_x[ts_present].apply(pd.to_numeric, errors='coerce').values.astype('float32')
            seq = np.nan_to_num(seq)
            seq = np.pad(seq, ((0,0),(0,MAX_SEQUENCE_LENGTH - seq.shape[1])),'constant') if seq.shape[1]<MAX_SEQUENCE_LENGTH else seq[:,:MAX_SEQUENCE_LENGTH]

            tgt_vals = df_y[tgt_present].apply(pd.to_numeric, errors='coerce').mean(axis=0).values
            tgt_vals = np.nan_to_num(tgt_vals)
            full_target = np.zeros(len(TARGET_COLUMNS))
            for i, tgt in enumerate(TARGET_COLUMNS):
                if tgt.lower() in tgt_present:
                    j = tgt_present.index(tgt.lower())
                    full_target[i] = tgt_vals[j]

            X_seqs.append(seq)
            static_val = float(re.sub(r'\D','', key) or 0)
            X_static.append([static_val])
            Y_list.append(full_target)

        except Exception as e:
            print(f"⚠️ Skipping {key}: {e}")

    if not X_seqs:
        raise FileNotFoundError("❌ No valid pairs found. Please verify CSV structure.")

    X_pad = pad_sequences(X_seqs, maxlen=MAX_SEQUENCE_LENGTH, dtype='float32', padding='post')
    X_static = np.array(X_static, dtype='float32')
    Y = np.array(Y_list, dtype='float32')

    print(f"✅ Loaded pairs: {len(X_seqs)} | X_seq: {X_pad.shape} | X_static: {X_static.shape} | Y: {Y.shape}")
    return X_pad, X_static, Y

# -------------------------------------------------------------------
# SCALING
# -------------------------------------------------------------------
def scale_data(X_pad, X_static, Y):
    scaler_Y = StandardScaler().fit(Y)
    Y_scaled = scaler_Y.transform(Y)

    scaler_X_static = StandardScaler().fit(X_static)
    X_static_scaled = scaler_X_static.transform(X_static)

    ns, nsteps, nfeat = X_pad.shape
    flat = X_pad.reshape((ns * nsteps, nfeat))
    scaler_X_seq = StandardScaler().fit(flat)
    flat_scaled = scaler_X_seq.transform(flat)
    X_scaled = flat_scaled.reshape((ns, nsteps, nfeat))

    return X_scaled, X_static_scaled, Y_scaled, scaler_X_seq, scaler_X_static, scaler_Y

# -------------------------------------------------------------------
# MODEL
# -------------------------------------------------------------------
def build_model(seq_shape, static_shape, output_shape):
    seq_in = Input(shape=seq_shape, name="seq_input")
    static_in = Input(shape=static_shape, name="static_input")

    x = GRU(128, return_sequences=True)(seq_in)
    x = Dropout(0.3)(x)
    x = GRU(64)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    s = Dense(32, activation='relu')(static_in)
    s = BatchNormalization()(s)

    merged = concatenate([x, s])
    h = Dense(128, activation='relu')(merged)
    h = Dropout(0.3)(h)
    h = Dense(64, activation='relu')(h)
    out = Dense(output_shape, activation='linear')(h)

    model = Model([seq_in, static_in], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model, Model([seq_in, static_in], h)

# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
def main():
    try:
        X_pad, X_static, Y = load_and_pair_data(DATA_FOLDER_PATH)
        X_seq_scaled, X_static_scaled, Y_scaled, scaler_X_seq, scaler_X_static, scaler_Y = scale_data(X_pad, X_static, Y)

        X_seq_tr, X_seq_te, X_static_tr, X_static_te, Y_tr, Y_te = train_test_split(
            X_seq_scaled, X_static_scaled, Y_scaled, test_size=0.2, random_state=RANDOM_SEED
        )

        model, feature_extractor = build_model(
            seq_shape=(MAX_SEQUENCE_LENGTH, X_seq_scaled.shape[2]),
            static_shape=(X_static_scaled.shape[1],),
            output_shape=Y_scaled.shape[1]
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1),
            ModelCheckpoint(SAVE_DIR / "best_gru_model.keras", save_best_only=True)
        ]

        print("\n🚀 Training GRU model...")
        hist = model.fit(
            [X_seq_tr, X_static_tr], Y_tr,
            validation_data=([X_seq_te, X_static_te], Y_te),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=callbacks, verbose=1
        )

        model.save(SAVE_DIR / "gru_final_model.keras")
        joblib.dump(scaler_X_seq, SAVE_DIR / "scaler_X_seq.joblib")
        joblib.dump(scaler_X_static, SAVE_DIR / "scaler_X_static.joblib")
        joblib.dump(scaler_Y, SAVE_DIR / "scaler_Y.joblib")
        with open(SAVE_DIR / "target_names.txt", "w") as f:
            f.write("\n".join(TARGET_COLUMNS))

        # Extract features for Stage 2 stacking
        print("\n🔍 Extracting GRU latent features...")
        X_train_features = feature_extractor.predict([X_seq_tr, X_static_tr])
        X_test_features = feature_extractor.predict([X_seq_te, X_static_te])
        np.savez_compressed(
            SAVE_DIR / "stage1_features.npz",
            X_train_features=X_train_features,
            X_test_features=X_test_features,
            Y_train_scaled=Y_tr,
            Y_test_scaled=Y_te
        )
        print(f"💾 Saved feature file → {SAVE_DIR / 'stage1_features.npz'}")

        # Predictions
        Y_pred_scaled = model.predict([X_seq_te, X_static_te])
        Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
        Y_true = scaler_Y.inverse_transform(Y_te)

        r2 = r2_score(Y_true, Y_pred)
        mae = mean_absolute_error(Y_true, Y_pred)
        smp = smape(Y_true, Y_pred)
        print(f"\n📊 Performance — R2={r2:.4f}, MAE={mae:.4f}, SMAPE={smp:.2f}%")

        print("\n✅ Stage 1 completed successfully! Ready for Stage 2 stacking.")

    except Exception as e:
        print("\n❌ ERROR:", type(e).__name__, e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
