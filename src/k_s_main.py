import itertools
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, f_oneway
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
ITERATIONS = 30
ALL_EPOCHS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]  # 1000 to 5000 epochs
# Focus on continuous numerical columns for K-S test
NUMERICAL_COLS = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

def run_all_epochs_training_and_ks(i, results_file_path):
    print(f"--- Iteration {i}/{ITERATIONS} ---")
    # List to store results
    # --- 1. Load and Prep Data ---
    try:
        data_full = pd.read_csv('../ObesityDataSet_raw_and_data_synthetic.csv')
        # Standard split to ensure we train on the same basis as your main experiment
        target_column = 'NObeyesdad'
        X = data_full.drop(columns=[target_column])
        y = data_full[target_column]

        D_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        D_train = pd.concat([D_train, y.loc[D_train.index]], axis=1)

        # Metadata for SDV
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=D_train)
        scores = {'Iteration': i}
        for epochs in ALL_EPOCHS:
            print(f"▶️ Training CTGAN ({epochs} epochs)...")
            # Train
            model = CTGANSynthesizer(metadata=metadata, epochs=epochs, verbose=False)
            model.fit(D_train)

            # Generate (same size as training)
            synthetic_data = model.sample(len(D_train))

            # Calculate Average K-S Score across all numerical columns
            ks_values = []
            for col in NUMERICAL_COLS:
                ks_stat, _ = ks_2samp(D_train[col], synthetic_data[col])
                ks_values.append(ks_stat)

            avg_ks = np.mean(ks_values)
            scores[f'KS_{epochs}'] = avg_ks
            print(f"    -> Average K-S Score: {avg_ks:.4f}")
        # Save partial results (so you don't lose data if it crashes)
        is_first_iteration = (i == 1)
        pd.DataFrame([scores]).to_csv(results_file_path, index=False, mode='a', header=is_first_iteration)
    except Exception as e:
        return f"❌ Iteration {i} failed during data loading. {e}"
    return f"✅ Iteration {i} completed."

def k_s_main():
    timestamp_for_file_name = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file_dir = f"../results/ks-{timestamp_for_file_name}"
    results_file = f"{results_file_dir}/multi_epoch_validation_ks_scores.csv"

    # Ensure results directory exists
    print(f"📁 Setting up results directory {results_file_dir}...")
    os.makedirs(results_file_dir, exist_ok=True)

    print(f"🚀 Starting Quality Control: {ALL_EPOCHS} Epochs")
    print(f"🔄 Running {ITERATIONS} iterations to prove statistical significance...\n")
    with ProcessPoolExecutor(max_workers=5) as executor:
        results = executor.map(
            run_all_epochs_training_and_ks,
            range(1, ITERATIONS + 1),
            itertools.repeat(results_file)
        )

        print("\n--- Parallel Results ---")
        for result in results:
            print(result)

    print("\n🎉 All parallel sets have finished.")


    # --- 3. Analysis (ANOVA) ---
    print("\n" + "="*50)
    print("📊 SWEET SPOT ANALYSIS")
    print("="*50)

    df = pd.read_csv(results_file)

    # Prepare lists for ANOVA
    groups = []
    for ep in ALL_EPOCHS:
        col_data = df[f'KS_{ep}']
        groups.append(col_data)
        print(f"{ep} Epochs -> Mean K-S: {col_data.mean():.4f} (±{col_data.std():.4f})")

    # Run ANOVA
    f_stat, p_value = f_oneway(*groups)
    with open(f"{results_file_dir}/final.md", 'a') as f:
        print("\n🧪 One-Way ANOVA Results:")
        print(f"   P-Value: {p_value:.4e}")
        f.writelines("\n🧪 One-Way ANOVA Results:")
        f.writelines(f"   P-Value: {p_value:.4e}")

        if p_value < 0.05:
            best_epoch = ALL_EPOCHS[np.argmin([g.mean() for g in groups])]
            print("✅ Significant difference found between groups.")
            print(f"🏆 The best performing epoch count is: {best_epoch}")
            f.writelines("✅ Significant difference found between groups.")
            f.writelines(f"🏆 The best performing epoch count is: {best_epoch}")
        else:
            print("❌ No significant difference. The lower epoch counts are likely sufficient.")
            f.writelines("❌ No significant difference. The lower epoch counts are likely sufficient.")

if __name__ == "__main__": k_s_main()