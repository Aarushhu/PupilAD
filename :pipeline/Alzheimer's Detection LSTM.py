"""
PupilAD 2.0 — Stage 2: LC Integrity Scoring Model
Bidirectional LSTM consuming per-task pupillometric features
(output of ingest_tobii.py or ingest_plr.py) -> LC integrity score (0-100)
-> risk tier (normal / watch / MCI-likely)

-------------------------------------------------------------------
IMPORTANT — READ BEFORE USING RESULTS ANYWHERE IN YOUR PAPER
-------------------------------------------------------------------
This script currently trains against SYNTHETIC labels (derived from the
synthetic generator's "healthy" vs "mci_like" profile tags) purely to
prove the model architecture and training loop work end-to-end. This
is a PIPELINE VALIDATION exercise, not a scientific result. Once real
pilot data + a real outcome measure (e.g. clinical MCI diagnosis, MMSE/
MoCA score, or ADNI-style diagnostic label if you get access to a
labeled reference set) is available, retrain against that instead of
the synthetic labels used here.

-------------------------------------------------------------------
INPUT FORMAT
-------------------------------------------------------------------
Expects the feature CSV produced by ingest_tobii.py's process_batch():
one row per subject, columns like "nback_mean_dilation",
"stroop_peak_dilation", etc. (8 features x 4 tasks = 32 feature columns,
plus subject_id).

The LSTM treats the 4 tasks as a sequence (timesteps), so features are
reshaped to (n_subjects, n_tasks, n_features_per_task) rather than a
flat vector — this preserves the "how did the pattern evolve across
tasks" signal the bidirectional design is meant to capture.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TASKS = ["nback", "digit_span", "stroop", "semantic_fluency"]
FEATURES_PER_TASK = [
    "mean_dilation", "peak_dilation", "latency_to_peak_ms", "dilation_auc",
    "dilation_slope", "baseline_pupil_mm", "pupil_variability", "valid_sample_frac",
]

RISK_TIERS = [
    (70, 100, "normal"),
    (50, 69, "watch"),
    (0, 49, "MCI-likely"),
]


def score_to_tier(score: float) -> str:
    for low, high, label in RISK_TIERS:
        if low <= score <= high:
            return label
    return "unknown"


def reshape_features_to_sequence(feature_df: pd.DataFrame) -> np.ndarray:
    """
    Convert the flat feature CSV (one row per subject, columns like
    'nback_mean_dilation') into a (n_subjects, n_tasks, n_features_per_task)
    array suitable for an LSTM.
    """
    n_subjects = len(feature_df)
    n_tasks = len(TASKS)
    n_feats = len(FEATURES_PER_TASK)
    sequence = np.zeros((n_subjects, n_tasks, n_feats))

    for t_idx, task in enumerate(TASKS):
        for f_idx, feat in enumerate(FEATURES_PER_TASK):
            col = f"{task}_{feat}"
            if col in feature_df.columns:
                sequence[:, t_idx, f_idx] = feature_df[col].fillna(0).values
            # else: leave as zeros — missing task/feature, flagged by valid_sample_frac anyway

    return sequence


def build_lstm_model(n_tasks: int, n_features: int) -> tf.keras.Model:
    """Bidirectional LSTM -> single LC integrity score (0-100)."""
    inputs = layers.Input(shape=(n_tasks, n_features))
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(inputs)
    x = layers.Bidirectional(layers.LSTM(16))(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid")(x)  # 0-1, scaled to 0-100 later

    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_on_synthetic(feature_csv: str = "pupilad_features_synthetic.csv"):
    """
    Train + evaluate the LSTM using synthetic labels derived from subject_id
    naming convention (SYN_HEALTHY_* -> high score, SYN_MCILIKE_* -> low score).
    THIS IS FOR PIPELINE VALIDATION ONLY — see module docstring.
    """
    df = pd.read_csv(feature_csv)

    # Synthetic ground truth: healthy -> ~85 +/- noise, mci_like -> ~35 +/- noise
    rng = np.random.default_rng(42)
    labels = df["subject_id"].apply(
        lambda sid: rng.normal(85, 5) if "HEALTHY" in sid else rng.normal(35, 8)
    ).values
    labels = np.clip(labels, 0, 100)

    X = reshape_features_to_sequence(df)
    y = labels / 100.0  # scale to 0-1 for sigmoid output

    # Standardize features across the flattened (subject*task) axis
    n_subjects, n_tasks, n_feats = X.shape
    scaler = StandardScaler()
    X_flat = X.reshape(-1, n_feats)
    X_scaled = scaler.fit_transform(X_flat).reshape(n_subjects, n_tasks, n_feats)

    # With only 10 synthetic subjects, a real train/test split is too small to
    # be meaningful — this just demonstrates the mechanics work end-to-end.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    model = build_lstm_model(n_tasks, n_feats)
    print(model.summary())

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=2,
        verbose=1,
    )

    predictions = model.predict(X_scaled).flatten() * 100
    results = pd.DataFrame({
        "subject_id": df["subject_id"],
        "true_score_synthetic": y * 100,
        "predicted_lc_integrity_score": predictions,
        "predicted_tier": [score_to_tier(s) for s in predictions],
    })

    print("\n--- Results (synthetic data — pipeline validation only) ---")
    print(results.to_string(index=False))

    return model, scaler, results


if __name__ == "__main__":
    train_on_synthetic()
