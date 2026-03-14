import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# --- Configuration ---
ITERATIONS = 30
DATA_PATH = '../ObesityDataSet_raw_and_data_synthetic.csv'
TARGET_COL = 'NObeyesdad'
RESULTS_FILE = '../results/catboost_completedataset_result.csv'


def prepare_features(train_df, test_df, target_col):
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    cat_cols = X_train.select_dtypes(include='object').columns
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    return X_train, y_train_enc, X_test, y_test_enc, cat_indices


def run_iteration(X_train, y_train, X_test, y_test, cat_indices):
    clf = CatBoostClassifier(
        loss_function='MultiClass',
        verbose=False,
    )
    clf.fit(X_train, y_train, cat_features=cat_indices)
    preds = clf.predict(X_test).ravel()
    return accuracy_score(y_test, preds)


def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    data_full = pd.read_csv(DATA_PATH)
    D_train, D_test = train_test_split(
        data_full,
        test_size=0.2,
        stratify=data_full[TARGET_COL],
    )

    X_train, y_train, X_test, y_test, cat_indices = prepare_features(D_train, D_test, TARGET_COL)

    results_log = []
    print(f"🚀 Starting CatBoost complete-data baseline for {ITERATIONS} iterations")

    for i in range(1, ITERATIONS + 1):
        acc = run_iteration(X_train, y_train, X_test, y_test, cat_indices)
        results_log.append({
            'Iteration': i,
            'Accuracy': acc,
        })

        print(f"   Iter {i}: Accuracy = {acc:.4f}")
        pd.DataFrame(results_log).to_csv(RESULTS_FILE, index=False)

    print(f"✅ Saved baseline results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
