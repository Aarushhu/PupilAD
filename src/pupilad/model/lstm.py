import numpy as np

from pupilad.tobii.ingest import FEATURE_NAMES, TASK_EVENTS

TASKS = list(TASK_EVENTS)
NORMAL_TIER_MIN_SCORE = 70
WATCH_TIER_MIN_SCORE = 50


def score_to_tier(score):
    if score >= NORMAL_TIER_MIN_SCORE:
        return "normal"
    if score >= WATCH_TIER_MIN_SCORE:
        return "watch"
    return "MCI-likely"


def features_to_sequence(feature_table):
    sequence = np.zeros((len(feature_table), len(TASKS), len(FEATURE_NAMES)))
    for task_index, task in enumerate(TASKS):
        for feature_index, feature in enumerate(FEATURE_NAMES):
            column = f"{task}_{feature}"
            if column in feature_table.columns:
                sequence[:, task_index, feature_index] = feature_table[column].fillna(0).to_numpy()
    return sequence


def build_lstm_model(n_tasks, n_features):
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(n_tasks, n_features))
    hidden = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(inputs)
    hidden = layers.Bidirectional(layers.LSTM(16))(hidden)
    hidden = layers.Dense(16, activation="relu")(hidden)
    hidden = layers.Dropout(0.2)(hidden)
    output = layers.Dense(1, activation="sigmoid")(hidden)
    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
