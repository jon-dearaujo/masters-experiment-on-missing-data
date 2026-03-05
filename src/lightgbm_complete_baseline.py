import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# --- Configuration ---
ITERATIONS = 30
DATA_PATH = '../ObesityDataSet_raw_and_data_synthetic.csv'
TARGET_COL = 'NObeyesdad'
RESULTS_FILE = '../results/lightgbm_completedataset_result.csv'


def prepare_features(train_df, test_df, target_col):
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Handle categoricals the same way as the missing-data script
    for col in X_train.select_dtypes(include='object').columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    return X_train, y_train_enc, X_test, y_test_enc


def run_iteration(X_train, y_train, X_test, y_test):
    clf = lgb.LGBMClassifier(verbose=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds)


def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    data_full = pd.read_csv(DATA_PATH)
    D_train, D_test = train_test_split(
        data_full,
        test_size=0.2,
        stratify=data_full[TARGET_COL],
    )

    X_train, y_train, X_test, y_test = prepare_features(D_train, D_test, TARGET_COL)

    results_log = []
    print(f"🚀 Starting LightGBM complete-data baseline for {ITERATIONS} iterations")

    for i in range(1, ITERATIONS + 1):
        acc = run_iteration(X_train, y_train, X_test, y_test)
        results_log.append({
            'Iteration': i,
            'Accuracy': acc,
        })

        print(f"   Iter {i}: Accuracy = {acc:.4f}")
        pd.DataFrame(results_log).to_csv(RESULTS_FILE, index=False)

    print(f"✅ Saved baseline results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
