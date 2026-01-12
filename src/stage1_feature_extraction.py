"""
stage1_feature_extraction.py
FIXED: Added Debugging + Key Normalization (01 vs 1)
"""

import os
import re
import sys
import json
import joblib
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
DATA_FOLDER_PATH = Path(r"/Ammonia/converted_csvs")
SAVE_DIR = Path(r"/Ammonia/Stage1_Output_Enhanced")
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
# HELPER: PHYSICS FEATURE EXTRACTOR
# -------------------------------------------------------------------
def extract_physics_stats(sequences):
    mean_val = np.mean(sequences, axis=1)
    max_val = np.max(sequences, axis=1)
    min_val = np.min(sequences, axis=1)
    std_val = np.std(sequences, axis=1)
    ptp_val = np.ptp(sequences, axis=1)
    skew_val = np.nan_to_num(skew(sequences, axis=1))
    kurt_val = np.nan_to_num(kurtosis(sequences, axis=1))
    features = np.hstack([mean_val, max_val, min_val, std_val, ptp_val, skew_val, kurt_val])
    return features

# -------------------------------------------------------------------
# DATA LOADING (FIXED)
# -------------------------------------------------------------------
def ensure_lower_strip(cols):
    return [str(c).strip().lower() for c in cols]

def load_and_pair_data(folder_path):
    print(f"\n📁 Scanning {folder_path} for data...")
    all_paths = list(folder_path.rglob("*.csv"))

    # 1. Map files
    file_map = {}
    print("   -> Indexing files...")

    for p in all_paths:
        try:
            name = p.stem.lower()
            # FIX: Normalize key to integer to match "01" with "1"
            num_match = re.search(r'\d+', name)
            if not num_match:
                print(f"⚠️ Skipping {name}: No number found.")
                continue

            key = str(int(num_match.group())) # '01' -> '1'

            # Identify type
            try:
                df_check = pd.read_csv(p, nrows=1)
                cols = set(ensure_lower_strip(df_check.columns))
            except Exception as e:
                print(f"⚠️ Could not read {name}: {e}")
                continue

            # Check column matches
            ts_match = any(c in cols for c in [x.lower() for x in TIME_SERIES_COLUMNS])
            perf_match = any(c in cols for c in [x.lower() for x in TARGET_COLUMNS])

            if ts_match:
                ftype = "ts"
            elif perf_match:
                ftype = "perf"
            else:
                # DEBUG PRINT: Show why it failed if neither matches
                # print(f"⚠️ Skipping {name}: Columns didn't match known targets/cycles.")
                continue

            if key not in file_map: file_map[key] = {}
            file_map[key][ftype] = p

        except Exception as e:
            print(f"⚠️ Error processing file {p.name}: {e}")
            continue

    # 2. Process Pairs
    X_seqs, X_static, Y_list = [], [], []

    ts_cols_lower = [c.lower() for c in TIME_SERIES_COLUMNS]
    tgt_cols_lower = [c.lower() for c in TARGET_COLUMNS]

    print(f"   -> Found {len(file_map)} potential file groups.")

    for key, files in tqdm(file_map.items(), desc="Loading Pairs"):
        # Check if pair is complete
        if "ts" not in files:
            print(f"❌ Key {key}: Missing Time-Series file (Found only Performance)")
            continue
        if "perf" not in files:
            print(f"❌ Key {key}: Missing Performance file (Found only Time-Series)")
            continue

        try:
            # Load Data
            df_x = pd.read_csv(files["ts"])
            df_y = pd.read_csv(files["perf"])
            df_x.columns = ensure_lower_strip(df_x.columns)
            df_y.columns = ensure_lower_strip(df_y.columns)

            # Extract Sequence
            valid_ts = [c for c in ts_cols_lower if c in df_x.columns]
            if not valid_ts:
                print(f"❌ Key {key}: TS file loaded but no valid columns found.")
                continue

            seq = df_x[valid_ts].values.astype('float32')

            # Pad/Truncate
            if seq.shape[0] < MAX_SEQUENCE_LENGTH:
                pad_amt = MAX_SEQUENCE_LENGTH - seq.shape[0]
                seq = np.pad(seq, ((0, pad_amt), (0,0)), 'constant')
            else:
                seq = seq[:MAX_SEQUENCE_LENGTH, :]

            # Extract Target
            y_vals = []
            for t in tgt_cols_lower:
                val = df_y[t].mean() if t in df_y.columns else 0.0
                y_vals.append(val)

            X_seqs.append(seq)
            X_static.append([float(key)])
            Y_list.append(y_vals)

        except Exception as e:
            print(f"❌ Key {key}: Error reading CSVs - {e}")
            continue

    return np.array(X_seqs), np.array(X_static), np.array(Y_list)

# -------------------------------------------------------------------
# MODEL DEFINITION
# -------------------------------------------------------------------
def build_gru_extractor(seq_shape, static_shape, output_shape):
    seq_in = Input(shape=seq_shape, name="seq_in")
    x = GRU(128, return_sequences=True)(seq_in)
    x = GRU(64)(x)
    x = BatchNormalization()(x)

    static_in = Input(shape=static_shape, name="static_in")
    s = Dense(32, activation='relu')(static_in)

    merged = concatenate([x, s])
    h = Dense(128, activation='relu', name="latent_features")(merged)
    out = Dense(output_shape)(h)

    model = Model([seq_in, static_in], out)
    model.compile(optimizer='adam', loss='mse')
    extractor = Model([seq_in, static_in], h)
    return model, extractor

# -------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------
if __name__ == "__main__":
    X_seq, X_static, Y = load_and_pair_data(DATA_FOLDER_PATH)

    # --- CRITICAL CHECK ---
    if len(Y) == 0:
        print("\n⛔ FATAL ERROR: No valid data pairs were loaded.")
        print("Please check the '❌' error messages above.")
        print("Ensure your CSVs have matching IDs (e.g. Cycle1.csv and Performance1.csv)")
        print("Ensure column names match the TARGET_COLUMNS list exactly.")
        sys.exit(1)

    print(f"\n✅ Successfully loaded {len(Y)} samples.")

    # 2. Scale Data
    scaler_Y = StandardScaler().fit(Y)
    Y_scaled = scaler_Y.transform(Y)

    ns, nt, nf = X_seq.shape
    scaler_seq = StandardScaler().fit(X_seq.reshape(-1, nf))
    X_seq_scaled = scaler_seq.transform(X_seq.reshape(-1, nf)).reshape(ns, nt, nf)

    scaler_static = StandardScaler().fit(X_static)
    X_static_scaled = scaler_static.transform(X_static)

    # 3. Train/Test Split
    indices = np.arange(len(X_seq))
    tr_idx, te_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_seq_tr, X_seq_te = X_seq_scaled[tr_idx], X_seq_scaled[te_idx]
    X_stat_tr, X_stat_te = X_static_scaled[tr_idx], X_static_scaled[te_idx]
    Y_tr, Y_te = Y_scaled[tr_idx], Y_scaled[te_idx]

    # 4. Train GRU
    print("\n🧠 Pre-training GRU Feature Extractor...")
    model, extractor = build_gru_extractor(
        (nt, nf), (X_static.shape[1],), Y.shape[1]
    )

    model.fit(
        [X_seq_tr, X_stat_tr], Y_tr,
        validation_data=([X_seq_te, X_stat_te], Y_te),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
    )

    # 5. Extract & Save
    print("\n⛏️ Extracting Features...")
    gru_feat_tr = extractor.predict([X_seq_tr, X_stat_tr], verbose=0)
    gru_feat_te = extractor.predict([X_seq_te, X_stat_te], verbose=0)

    phys_feat_tr = extract_physics_stats(X_seq_tr)
    phys_feat_te = extract_physics_stats(X_seq_te)

    X_train_final = np.hstack([gru_feat_tr, phys_feat_tr, X_stat_tr])
    X_test_final = np.hstack([gru_feat_te, phys_feat_te, X_stat_te])

    np.savez_compressed(
        SAVE_DIR / "NEWstage1_enhanced_features.npz",
        X_train=X_train_final,
        X_test=X_test_final,
        Y_train_raw=Y[tr_idx],
        Y_test_raw=Y[te_idx]
    )

    joblib.dump(scaler_Y, SAVE_DIR / "scaler_Y.joblib")
    with open(SAVE_DIR / "target_names.txt", "w") as f:
        f.write("\n".join(TARGET_COLUMNS))

    print(f"\n🎉 Stage 1 Complete. Saved to {SAVE_DIR}")