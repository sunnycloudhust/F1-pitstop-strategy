from model import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score
)

path = "C:/Nguyen Tri/Code/Statisanalyss/all_data.csv"

df = pd.read_csv(path)

categorical_cols = ['Compound', 'Team', 'Driver']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

continuous_cols = [
    'LapTime_Seconds', 'Position', 'LapNumber', 'Stint', 'TyreLife', 
    'TrackStatus', 'delta_laptime', 'CumulativeTimeStint', 'race_progress_fraction'
]

feature_cols = continuous_cols + [col for col in df_encoded.columns if any(c in col for c in categorical_cols)]

scaler = StandardScaler()
df_encoded[continuous_cols] = scaler.fit_transform(df_encoded[continuous_cols])

df_encoded = df_encoded.fillna(0)

def create_driver_sequences(data, feature_columns, target_column='HasPitStop', seq_length=10):
    X, y = [], []
    for _, group in data.groupby(['RaceID', 'DriverNumber']):
        feat = group[feature_columns].values
        targ = group[target_column].values
        if len(feat) < seq_length:
            continue
        for i in range(len(feat) - seq_length + 1):
            X.append(feat[i : i + seq_length])
            y.append(targ[i + seq_length - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

if 2025 in df_encoded['Year'].values:
    train_df = df_encoded[df_encoded['Year'] < 2025]
    test_df = df_encoded[df_encoded['Year'] == 2025]
else:
    latest_year = df_encoded['Year'].max()
    print(f"Warning: 2025 data not found. Using {latest_year} for testing instead.")
    train_df = df_encoded[df_encoded['Year'] < latest_year]
    test_df = df_encoded[df_encoded['Year'] == latest_year]

X_train, y_train = create_driver_sequences(train_df, feature_cols)
X_test, y_test = create_driver_sequences(test_df, feature_cols)

pit_stop_model = F1PitStopPredictor()

pit_stop_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(curve='PR', name='auc_pr')]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6
)

pit_stop_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=50,
    class_weight={0: 1.0, 1: 3.0},
    callbacks=[early_stopping, reduce_lr]
)

y_pred_probs = pit_stop_model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_probs)
auc_pr = average_precision_score(y_test, y_pred_probs)
balanced_acc = balanced_accuracy_score(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print("--- Evaluation Metrics ---")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1-Score:          {f1:.4f}")
print(f"Specificity:       {specificity:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"ROC-AUC:           {roc_auc:.4f}")
print(f"AUC-PR:            {auc_pr:.4f}")

print("\n--- Confusion Matrix ---")
print(f"True Negatives (TN):  {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP):  {tp}")