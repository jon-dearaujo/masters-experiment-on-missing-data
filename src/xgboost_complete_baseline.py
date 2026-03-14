import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# --- Configuration ---
ITERATIONS = 30
DATA_PATH = '../ObesityDataSet_raw_and_data_synthetic.csv'
TARGET_COL = 'NObeyesdad'
RESULTS_FILE = '../results/xgboost_completedataset_result.csv'


def prepare_features(train_df, test_df, target_col):
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    X_train_enc = pd.get_dummies(X_train, drop_first=False)
    X_test_enc = pd.get_dummies(X_test, drop_first=False)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    num_classes = len(le.classes_)

    return X_train_enc, y_train_enc, X_test_enc, y_test_enc, num_classes


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

    X_train, y_train, X_test, y_test, num_classes = prepare_features(D_train, D_test, TARGET_COL)

    results_log = []
    print(f"🚀 Starting XGBoost complete-data baseline for {ITERATIONS} iterations")

    for i in range(1, ITERATIONS + 1):
        acc = run_iteration(X_train, y_train, X_test, y_test, num_classes)
        results_log.append({
            'Iteration': i,
            'Accuracy': acc,
        })

        print(f"   Iter {i}: Accuracy = {acc:.4f}")
        pd.DataFrame(results_log).to_csv(RESULTS_FILE, index=False)

    print(f"✅ Saved baseline results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
