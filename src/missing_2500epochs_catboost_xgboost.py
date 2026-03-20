import os
import numpy as np
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import multiprocessing as mp
import random
from datetime import datetime

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

# --- Configuration ---
FIXED_EPOCHS = 5000  # anteriormente 2500; atualizado para refletir melhor desempenho
ITERATIONS = 30      # repeticoes para significância
MISSINGNESS_LEVELS = [0.10, 0.20, 0.30, 0.40]
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE_CAT = f"../results/{TIMESTAMP}_final_missingness_catboost.csv"
RESULTS_FILE_XGB = f"../results/{TIMESTAMP}_final_missingness_xgboost.csv"
DATA_PATH = '../ObesityDataSet_raw_and_data_synthetic.csv'
TARGET_COL = 'NObeyesdad'
WORKERS = 4  # run levels in parallel (one per level)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def apply_mcar_missingness(df, level, target_col):
    data = df.copy()
    for col in df.columns:
        if col == target_col:
            continue
        mask = np.random.rand(len(data)) < level
        data.loc[mask, col] = np.nan
    return data


def evaluate_catboost(synthetic_data, real_test_data, target_col, seed: int):
    le = LabelEncoder()
    y_train = le.fit_transform(synthetic_data[target_col])
    y_test = le.transform(real_test_data[target_col])

    X_train = synthetic_data.drop(columns=[target_col]).copy()
    X_test = real_test_data.drop(columns=[target_col]).copy()

    cat_cols = X_train.select_dtypes(include="object").columns
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]

    # CatBoost requires categorical features to be strings (no NaN).
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str).fillna("nan")
        X_test[col] = X_test[col].astype(str).fillna("nan")

    model = CatBoostClassifier(
        loss_function="MultiClass",
        verbose=False,
        random_seed=seed,
    )
    model.fit(X_train, y_train, cat_features=cat_indices)

    preds = model.predict(X_test).ravel()
    return accuracy_score(y_test, preds)


def _encode_for_xgb(train_df, test_df):
    X_train = pd.get_dummies(train_df, drop_first=False)
    X_test = pd.get_dummies(test_df, drop_first=False)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test


def evaluate_xgboost(synthetic_data, real_test_data, target_col, seed: int):
    le = LabelEncoder()
    y_train = le.fit_transform(synthetic_data[target_col])
    y_test = le.transform(real_test_data[target_col])

    X_train_raw = synthetic_data.drop(columns=[target_col])
    X_test_raw = real_test_data.drop(columns=[target_col])

    X_train, X_test = _encode_for_xgb(X_train_raw, X_test_raw)

    num_classes = len(set(list(y_train)))  # type: ignore[arg-type]
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        verbosity=0,
        seed=seed,
    )
    model.fit(X_train, y_train)

    probas = model.predict_proba(X_test)
    preds = np.argmax(probas, axis=1)
    return accuracy_score(y_test, preds)


def _run_single_task(level, iteration):
    set_seed(iteration)
    data_full = pd.read_csv(DATA_PATH)

    D_train, D_test = train_test_split(
        data_full,
        test_size=0.2,
        random_state=iteration,
        stratify=data_full[TARGET_COL]
    )

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(D_train)

    D_incomplete = apply_mcar_missingness(D_train, level, TARGET_COL)

    model = CTGANSynthesizer(metadata=metadata, epochs=FIXED_EPOCHS, verbose=False)
    model.fit(D_incomplete)

    S_incomplete = model.sample(len(D_train))

    acc_cat = evaluate_catboost(S_incomplete, D_test, TARGET_COL, iteration)
    acc_xgb = evaluate_xgboost(S_incomplete, D_test, TARGET_COL, iteration)

    return {
        'Missingness': level,
        'Iteration': iteration,
        'CatBoost': acc_cat,
        'XGBoost': acc_xgb,
    }


def _task_wrapper(args):
    level, iteration = args
    return _run_single_task(level, iteration)


def run_experiment():
    os.makedirs(os.path.dirname(RESULTS_FILE_CAT), exist_ok=True)

    tasks = [(level, i) for level in MISSINGNESS_LEVELS for i in range(1, ITERATIONS + 1)]

    results_cat = []
    results_xgb = []

    print(f"🚀 Starting Main Experiment: Epochs={FIXED_EPOCHS}, Levels={MISSINGNESS_LEVELS}, Workers={WORKERS}")

    with mp.Pool(processes=WORKERS) as pool:
        for res in pool.imap_unordered(_task_wrapper, tasks):
            level = res['Missingness']
            iteration = res['Iteration']
            acc_cat = res['CatBoost']
            acc_xgb = res['XGBoost']

            print(f"   Level {int(level * 100)}% | Iter {iteration}: CatBoost = {acc_cat:.4f} | XGBoost = {acc_xgb:.4f}")

            results_cat.append({
                'Missingness': level,
                'Iteration': iteration,
                'Accuracy': acc_cat
            })
            results_xgb.append({
                'Missingness': level,
                'Iteration': iteration,
                'Accuracy': acc_xgb
            })

            pd.DataFrame(results_cat).to_csv(RESULTS_FILE_CAT, index=False)
            pd.DataFrame(results_xgb).to_csv(RESULTS_FILE_XGB, index=False)


if __name__ == "__main__":
    run_experiment()
