"""
server/scratch/run_mask_comparison.py
====================================
Empirical experiment: Compare Unmasked/Zero-Imputed BiLSTM vs. Binary-Masked BiLSTM vs. CatBoost
on the exact sequence diagnosis task.
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from catboost import CatBoostClassifier

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SERVER_DIR)

from ml.lstm_model.train_model import preprocess_data, DATA_PATH

FEATURE_COLS = ['entry_age', 'CDGLOBAL', 'MMSCORE', 'TOTSCORE', 'visit_month']
TARGET_COL = 'DIAGNOSIS'
MAX_LEN = 3
LABEL_NAMES = ['CN', 'MCI', 'AD']

print("Loading data from", DATA_PATH)
df = pd.read_csv(DATA_PATH)
df_filtered = preprocess_data(df)

# Subject-level split
unique_subjects = np.array(list(df_filtered['subject_id'].unique()))
train_ids, test_ids = train_test_split(unique_subjects, test_size=0.2, random_state=SEED)
train_df = df_filtered[df_filtered['subject_id'].isin(train_ids)].copy()
test_df = df_filtered[df_filtered['subject_id'].isin(test_ids)].copy()

print(f"Train subjects: {len(train_ids)}, Test subjects: {len(test_ids)}")

def create_windows_with_masks(df):
    X_vals, X_masks, y, X_last_visit = [], [], [], []
    for pid, group in df.groupby('subject_id'):
        group = group.sort_values('visit_month').reset_index(drop=True)
        n = len(group)
        if n < 3:
            continue
        for i in range(2, n):
            start = max(0, i - MAX_LEN)
            window_df = group.iloc[start:i][FEATURE_COLS]
            
            # Values and masks
            raw_vals = window_df.values.copy()
            masks = (~np.isnan(raw_vals)).astype(float) # 1 if observed, 0 if NaN
            
            X_vals.append(raw_vals)
            X_masks.append(masks)
            y.append(group.iloc[i][TARGET_COL])
            # Last available visit for tabular CatBoost comparison
            last_row = group.iloc[i-1][FEATURE_COLS].to_dict()
            X_last_visit.append(last_row)
            
    return X_vals, X_masks, np.array(y, dtype=int), pd.DataFrame(X_last_visit)

print("Building windows with observation masks...")
X_train_v, X_train_m, y_train, X_train_last = create_windows_with_masks(train_df)
X_test_v, X_test_m, y_test, X_test_last = create_windows_with_masks(test_df)

print(f"Train windows: {len(y_train)}, Test windows: {len(y_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset A: Zero-Imputed BiLSTM (Current baseline in repo)
# ─────────────────────────────────────────────────────────────────────────────
X_tr_zero = pad_sequences(X_train_v, maxlen=MAX_LEN, dtype='float32', padding='pre', value=0.0)
X_te_zero = pad_sequences(X_test_v, maxlen=MAX_LEN, dtype='float32', padding='pre', value=0.0)

X_tr_zero = np.nan_to_num(X_tr_zero, nan=0.0)
X_te_zero = np.nan_to_num(X_te_zero, nan=0.0)

scaler_zero = MinMaxScaler()
tr_shape = X_tr_zero.shape
te_shape = X_te_zero.shape

X_tr_zero_scaled = scaler_zero.fit_transform(X_tr_zero.reshape(-1, tr_shape[-1])).reshape(tr_shape)
X_te_zero_scaled = scaler_zero.transform(X_te_zero.reshape(-1, te_shape[-1])).reshape(te_shape)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset B: Binary-Masked BiLSTM (5 normalized values + 5 binary masks = 10 channels)
# ─────────────────────────────────────────────────────────────────────────────
# Pad values with 0.0, pad masks with 0.0 (unobserved)
X_tr_v_padded = pad_sequences(X_train_v, maxlen=MAX_LEN, dtype='float32', padding='pre', value=0.0)
X_te_v_padded = pad_sequences(X_test_v, maxlen=MAX_LEN, dtype='float32', padding='pre', value=0.0)

X_tr_m_padded = pad_sequences(X_train_m, maxlen=MAX_LEN, dtype='float32', padding='pre', value=0.0)
X_te_m_padded = pad_sequences(X_test_m, maxlen=MAX_LEN, dtype='float32', padding='pre', value=0.0)

# Replace remaining NaNs in values with median/0
X_tr_v_padded = np.nan_to_num(X_tr_v_padded, nan=0.0)
X_te_v_padded = np.nan_to_num(X_te_v_padded, nan=0.0)

# Scale values
scaler_mask = MinMaxScaler()
X_tr_v_scaled = scaler_mask.fit_transform(X_tr_v_padded.reshape(-1, tr_shape[-1])).reshape(tr_shape)
X_te_v_scaled = scaler_mask.transform(X_te_v_padded.reshape(-1, te_shape[-1])).reshape(te_shape)

# Concatenate [scaled_values, binary_masks] -> 10 features per time step
X_tr_masked = np.concatenate([X_tr_v_scaled, X_tr_m_padded], axis=-1) # (N, 3, 10)
X_te_masked = np.concatenate([X_te_v_scaled, X_te_m_padded], axis=-1) # (N, 3, 10)

print(f"Masked BiLSTM input shape: {X_tr_masked.shape}")

# Class weights
classes = np.unique(y_train)
cw = compute_class_weight('balanced', classes=classes, y=y_train)
cw_dict = dict(zip(classes, cw))

def build_bilstm(in_dim):
    m = Sequential([
        Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.1),
                      input_shape=(MAX_LEN, in_dim)),
        BatchNormalization(),
        Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

# ── Train Model 1: Zero-Imputed BiLSTM ────────────────────────────────────────
print("\n[1/3] Training Baseline Zero-Imputed BiLSTM (5 features)...")
m1 = build_bilstm(5)
cb1 = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
       ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-5)]
m1.fit(X_tr_zero_scaled, y_train, validation_data=(X_te_zero_scaled, y_test),
       epochs=25, batch_size=32, class_weight=cw_dict, callbacks=cb1, verbose=0)

p1 = m1.predict(X_te_zero_scaled, verbose=0)
pred1 = np.argmax(p1, axis=1)
acc1 = accuracy_score(y_test, pred1)
f1_1 = f1_score(y_test, pred1, average='macro')
loss1 = log_loss(y_test, p1)

# ── Train Model 2: Binary-Masked BiLSTM ───────────────────────────────────────
print("\n[2/3] Training Binary-Masked BiLSTM (10 features: 5 values + 5 masks)...")
m2 = build_bilstm(10)
cb2 = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
       ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-5)]
m2.fit(X_tr_masked, y_train, validation_data=(X_te_masked, y_test),
       epochs=25, batch_size=32, class_weight=cw_dict, callbacks=cb2, verbose=0)

p2 = m2.predict(X_te_masked, verbose=0)
pred2 = np.argmax(p2, axis=1)
acc2 = accuracy_score(y_test, pred2)
f1_2 = f1_score(y_test, pred2, average='macro')
loss2 = log_loss(y_test, p2)

# ── Train Model 3: CatBoost (Natural Missingness, No Zero-Filling) ─────────────
print("\n[3/3] Training CatBoost with Natural Missingness on same task...")
# Flatten window features (up to 3 visits = 15 features) keeping NaNs natural!
def flatten_windows(X_vals_list):
    flat = []
    for w in X_vals_list:
        row = []
        # pad to 3
        pad_len = MAX_LEN - len(w)
        for _ in range(pad_len):
            row.extend([np.nan] * len(FEATURE_COLS))
        for v in w:
            row.extend(v)
        flat.append(row)
    return np.array(flat, dtype=float)

X_tr_cb = flatten_windows(X_train_v)
X_te_cb = flatten_windows(X_test_v)

cb_model = CatBoostClassifier(
    iterations=400, depth=6, learning_rate=0.05,
    loss_function='MultiClass', random_seed=SEED,
    class_weights=cw, verbose=0, early_stopping_rounds=40
)
cb_model.fit(X_tr_cb, y_train, eval_set=(X_te_cb, y_test))

p3 = cb_model.predict_proba(X_te_cb)
pred3 = np.argmax(p3, axis=1)
acc3 = accuracy_score(y_test, pred3)
f1_3 = f1_score(y_test, pred3, average='macro')
loss3 = log_loss(y_test, p3)

# ── Final Results Comparison ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EMPIRICAL COMPARISON RESULTS ON HELD-OUT TEST SPLIT (N = %d windows)" % len(y_test))
print("=" * 70)
print(f"{'Model Architecture':<32} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Log Loss':<10}")
print("-" * 70)
print(f"{'Zero-Imputed BiLSTM (5 feat)':<32} | {acc1:.4f}     | {f1_1:.4f}     | {loss1:.4f}")
print(f"{'Binary-Masked BiLSTM (10 feat)':<32} | {acc2:.4f}     | {f1_2:.4f}     | {loss2:.4f}")
print(f"{'CatBoost (Natural Missingness)':<32} | {acc3:.4f}     | {f1_3:.4f}     | {loss3:.4f}")
print("=" * 70)

print("\n--- Detailed Classification Reports ---")
print("\n1. Zero-Imputed BiLSTM:")
print(classification_report(y_test, pred1, target_names=LABEL_NAMES, digits=4))

print("\n2. Binary-Masked BiLSTM:")
print(classification_report(y_test, pred2, target_names=LABEL_NAMES, digits=4))

print("\n3. CatBoost (Natural Missingness):")
print(classification_report(y_test, pred3, target_names=LABEL_NAMES, digits=4))
