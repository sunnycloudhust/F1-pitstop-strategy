import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight

from model import F1PitStopPredictor

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "all_data.csv"
SEQ_LENGTH  = 10
BATCH_SIZE  = 64  
EPOCHS      = 50
LR          = 1e-3 
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Load & encode ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

categorical_cols = ['Compound', 'Team', 'Driver']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
df_encoded = df_encoded.fillna(0)

continuous_cols = [
    'LapTime_Seconds', 'Position', 'LapNumber', 'Stint', 'TyreLife',
    'TrackStatus', 'delta_laptime', 'CumulativeTimeStint', 'race_progress_fraction',
]
feature_cols = continuous_cols + [
    col for col in df_encoded.columns
    if any(c + '_' in col for c in categorical_cols)
]

# ── Train / test split ────────────────────────────────────────────────────────
if 2025 in df_encoded['Year'].values:
    test_year = 2025
else:
    test_year = df_encoded['Year'].max()

train_df = df_encoded[df_encoded['Year'] < test_year].copy()
test_df  = df_encoded[df_encoded['Year'] == test_year].copy()

# ── Scale continuous features ─────────────────────────────────────────────────
scaler = StandardScaler()
train_df[continuous_cols] = scaler.fit_transform(train_df[continuous_cols])
test_df[continuous_cols]  = scaler.transform(test_df[continuous_cols])
joblib.dump(scaler, 'scaler.pkl')

# ── Sequence builder ──────────────────────────────────────────────────────────
def create_driver_sequences(
    data: pd.DataFrame,
    feature_columns: list,
    target_column: str = 'HasPitStop',
    seq_length: int = SEQ_LENGTH,
):
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

X_train, y_train = create_driver_sequences(train_df, feature_cols)
X_test,  y_test  = create_driver_sequences(test_df,  feature_cols)


classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

print(f"Computed class weights: {class_weight_dict}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = F1PitStopPredictor()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(curve='PR', name='auc_pr'),
    ],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc_pr', patience=7,
        restore_best_weights=True, mode='max',
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras', monitor='val_auc_pr',
        save_best_only=True, mode='max', verbose=1,
    ),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weight_dict, 
    callbacks=callbacks,
)

# ── Vẽ biểu đồ Loss ────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss Over Time')
plt.ylabel('Loss (Binary Crossentropy)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png', bbox_inches='tight') 
print("\nLoss plot saved to 'loss_curve.png'.")

# ── Evaluation ────────────────────────────────────────────────────────────────
y_pred_probs = model.predict(X_test).flatten()

y_pred       = (y_pred_probs > 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\n─── Evaluation Metrics ───────────────────────────────────")
print(f"  Precision:         {precision_score(y_test, y_pred):.4f}")
print(f"  Recall:            {recall_score(y_test, y_pred):.4f}")
print(f"  F1-Score:          {f1_score(y_test, y_pred):.4f}")
print(f"  Specificity:       {tn / (tn + fp):.4f}")
print(f"  Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC:           {roc_auc_score(y_test, y_pred_probs):.4f}")
print(f"  AUC-PR:            {average_precision_score(y_test, y_pred_probs):.4f}")

print("\n─── Confusion Matrix ─────────────────────────────────────")
print(f"  True  Negatives : {tn:>6}")
print(f"  False Positives : {fp:>6}")
print(f"  False Negatives : {fn:>6}")
print(f"  True  Positives : {tp:>6}")
