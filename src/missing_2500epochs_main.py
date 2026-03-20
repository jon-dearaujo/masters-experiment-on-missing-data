import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import os
import random
from datetime import datetime

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

# --- Configuration ---
FIXED_EPOCHS = 5000  # anteriormente 2500; atualizado com base nos testes pareados
ITERATIONS = 30      # para garantir significância estatística
MISSINGNESS_LEVELS = [0.10, 0.20, 0.30, 0.40] # 10% a 40%
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = f"../results/{TIMESTAMP}_final_missingness_impact.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

# --- Helper: Train & Evaluate LightGBM ---
def get_utility_score(synthetic_data, real_test_data, target_col, seed: int):
    # Prepare data for LightGBM (Encoding)
    le = LabelEncoder()

    X_train = synthetic_data.drop(columns=[target_col])
    y_train = le.fit_transform(synthetic_data[target_col])

    X_test = real_test_data.drop(columns=[target_col])
    y_test = le.transform(real_test_data[target_col]) # Use same encoder

    # Handle categoricals
    for col in X_train.select_dtypes(include='object').columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    # Train LightGBM
    clf = lgb.LGBMClassifier(verbose=-1, random_state=seed)
    clf.fit(X_train, y_train)

    # Predict & Score
    predictions = clf.predict(X_test)
    return accuracy_score(y_test, predictions)

# --- Main Experiment Loop ---
results_log = []

# 1. Load Real Data (ONCE)
data_full = pd.read_csv('../ObesityDataSet_raw_and_data_synthetic.csv')
target_col = 'NObeyesdad'

print(f"🚀 Starting Main Experiment: Epochs={FIXED_EPOCHS}, Levels={MISSINGNESS_LEVELS}")

for level in MISSINGNESS_LEVELS:
    print(f"\n--- Testing Missingness Level: {int(level*100)}% ---")

    for i in range(1, ITERATIONS + 1):
        set_seed(i)
        D_train_iter, D_test_iter = train_test_split(
            data_full,
            test_size=0.2,
            stratify=data_full[target_col],
            random_state=i
        )

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(D_train_iter)

        # A. Create Incomplete Data (Randomly each time!)
        D_incomplete = D_train_iter.copy()
        # Apply MCAR missingness
        for col in D_train_iter.columns:
            if col != target_col:
                mask = np.random.rand(len(D_incomplete)) < level
                D_incomplete.loc[mask, col] = np.nan

        # B. Train Generative Model
        model = CTGANSynthesizer(metadata=metadata, epochs=FIXED_EPOCHS, verbose=False)
        model.fit(D_incomplete)

        # C. Generate Synthetic Data
        S_incomplete = model.sample(len(D_train_iter))

        # D. Measure Utility (The "Score")
        acc = get_utility_score(S_incomplete, D_test_iter, target_col, i)

        print(f"   Iter {i}: Accuracy = {acc:.4f}")

        results_log.append({
            'Missingness': level,
            'Iteration': i,
            'Accuracy': acc
        })

        # Save incrementally (only to timestamped file to preserve históricos)
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        pd.DataFrame(results_log).to_csv(RESULTS_FILE, index=False)
