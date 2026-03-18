import os
from typing import cast

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# --- Configuration ---
FIXED_EPOCHS = 2500
ITERATIONS = 30
DATA_PATH = '../ObesityDataSet_raw_and_data_synthetic.csv'
TARGET_COL = 'NObeyesdad'
RESULTS_FILE = '../results/ctgan_xgboost_completedataset_result.csv'


def train_ctgan_and_sample(train_df):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)

    print(f"🧪 Training CTGAN on complete data (epochs={FIXED_EPOCHS})...")
    model = CTGANSynthesizer(metadata=metadata, epochs=FIXED_EPOCHS, verbose=False)
    model.fit(train_df)

    synthetic = model.sample(len(train_df)).reset_index(drop=True)
    print("📦 CTGAN sampling complete.")
    return synthetic


def ensure_known_targets(df, target_col, known_values):
    invalid_mask = ~df[target_col].isin(known_values)
    if invalid_mask.any():
        df.loc[invalid_mask, target_col] = known_values[0]
    return df


def _encode_for_xgb(train_df, test_df):
    X_train = pd.get_dummies(train_df, drop_first=False)
    X_test = pd.get_dummies(test_df, drop_first=False)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test


def prepare_features(train_df, test_df, target_col, label_encoder):
    y_train = label_encoder.transform(train_df[target_col])
    y_test = label_encoder.transform(test_df[target_col])

    X_train_raw = train_df.drop(columns=[target_col]).copy()
    X_test_raw = test_df.drop(columns=[target_col]).copy()

    X_train, X_test = _encode_for_xgb(X_train_raw, X_test_raw)
    num_classes = len(label_encoder.classes_)

    return X_train, y_train, X_test, y_test, num_classes


def run_iteration(X_train, y_train, X_test, y_test, num_classes):
    clf = XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        use_label_encoder=False,
        verbosity=0,
    )
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)
    preds = probas.argmax(axis=1)
    return accuracy_score(y_test, preds)


def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    data_full = pd.read_csv(DATA_PATH)
    D_train, D_test = train_test_split(
        data_full,
        test_size=0.2,
        stratify=data_full[TARGET_COL],
    )
    D_train = cast(pd.DataFrame, D_train)
    D_test = cast(pd.DataFrame, D_test)

    label_encoder = LabelEncoder()
    label_encoder.fit(D_train[TARGET_COL])

    results_log = []
    print(f"🚀 Starting CTGAN-XGBoost complete-data baseline for {ITERATIONS} iterations")

    for i in range(1, ITERATIONS + 1):
        print(f"\n--- Iteration {i}/{ITERATIONS}: CTGAN training ---")
        synthetic_train = train_ctgan_and_sample(D_train)
        synthetic_train = ensure_known_targets(synthetic_train, TARGET_COL, label_encoder.classes_)

        X_train, y_train, X_test, y_test, num_classes = prepare_features(
            synthetic_train,
            D_test,
            TARGET_COL,
            label_encoder,
        )

        acc = run_iteration(X_train, y_train, X_test, y_test, num_classes)
        results_log.append({
            'Missingness': 0.0,
            'Iteration': i,
            'Accuracy': acc,
        })

        print(f"   Iter {i}: Accuracy = {acc:.4f}")
        pd.DataFrame(results_log).to_csv(RESULTS_FILE, index=False)

    print(f"✅ Saved baseline results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
