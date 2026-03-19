import itertools
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, f_oneway, friedmanchisquare, wilcoxon
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sklearn.model_selection import train_test_split

try:
    import torch
except ImportError:  # pragma: no cover - torch might not be installed
    torch = None


PAIRED_ITERATIONS = 30
# PAIRED_EPOCHS = [2_000, 2_500, 5_000]
PAIRED_EPOCHS = [5_000, 7_500, 10_000]
NUMERICAL_COLS = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
DEFAULT_MAX_WORKERS = 4


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs for deterministic paired runs."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def run_paired_epochs_training_and_ks(iteration: int, results_file_path: str) -> str:
    print(f"--- Paired Iteration {iteration}/{PAIRED_ITERATIONS} (seed={iteration}) ---")
    set_seed(iteration)

    try:
        data_full = pd.read_csv('../ObesityDataSet_raw_and_data_synthetic.csv')
        target_column = 'NObeyesdad'
        X = data_full.drop(columns=[target_column])
        y = data_full[target_column]

        D_train, _, _, _ = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=iteration,
            stratify=y
        )
        D_train = pd.concat([D_train, y.loc[D_train.index]], axis=1)

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=D_train)
        scores = {'Iteration': iteration}

        for epochs in PAIRED_EPOCHS:
            print(f"▶️ Training CTGAN ({epochs} epochs)...")
            set_seed(iteration)
            model = CTGANSynthesizer(metadata=metadata, epochs=epochs, verbose=False)
            model.fit(D_train)

            synthetic_data = model.sample(len(D_train))

            ks_values = []
            for col in NUMERICAL_COLS:
                ks_stat, _ = ks_2samp(D_train[col], synthetic_data[col])
                ks_values.append(ks_stat)

            avg_ks = np.mean(ks_values)
            scores[f'KS_{epochs}'] = avg_ks
            print(f"    -> Average K-S Score: {avg_ks:.4f}")

        is_first_iteration = (iteration == 1)
        pd.DataFrame([scores]).to_csv(
            results_file_path,
            index=False,
            mode='a',
            header=is_first_iteration
        )

    except Exception as exc:  # pragma: no cover - logging path
        return f"❌ Iteration {iteration} failed. {exc}"

    return f"✅ Iteration {iteration} completed."


def paired_k_s_main() -> None:
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file_dir = f"../results/paired-ks-{timestamp}"
    results_file = f"{results_file_dir}/paired_multi_epoch_validation_ks_scores.csv"
    summary_file = f"{results_file_dir}/paired_final.md"

    print(f"📁 Setting up results directory {results_file_dir}...")
    os.makedirs(results_file_dir, exist_ok=True)

    print(f"🚀 Starting Paired KS Quality Control: {PAIRED_EPOCHS} Epochs")
    print(f"🔄 Running {PAIRED_ITERATIONS} paired iterations to prove statistical significance...\n")

    with ProcessPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
        results = executor.map(
            run_paired_epochs_training_and_ks,
            range(1, PAIRED_ITERATIONS + 1),
            itertools.repeat(results_file)
        )

        print("\n--- Parallel Results ---")
        for result in results:
            print(result)

    print("\n🎉 All paired parallel sets have finished.")

    lines = []

    print("\n" + "=" * 50)
    lines.append("\n" + "=" * 50)
    print("📊 SWEET SPOT ANALYSIS (Paired)")
    lines.append("📊 SWEET SPOT ANALYSIS (Paired)")
    print("=" * 50)
    lines.append("=" * 50)

    df = pd.read_csv(results_file)

    groups = []
    for ep in PAIRED_EPOCHS:
        col_data = df[f'KS_{ep}']
        groups.append(col_data)
        print(f"{ep} Epochs -> Mean K-S: {col_data.mean():.4f} (±{col_data.std():.4f})")
        lines.append(f"{ep} Epochs -> Mean K-S: {col_data.mean():.4f} (±{col_data.std():.4f})")


    friedman_stat, friedman_p = friedmanchisquare(*groups)
    lines.append("\n🧮 Friedman Test Results:")
    lines.append(f"   Chi-Square: {friedman_stat:.4f}")
    lines.append(f"   P-Value: {friedman_p:.4e}")

    w_5000_7500, p_5000_7500 = wilcoxon(df['KS_5000'], df['KS_7500'], alternative='two-sided')
    w_7500_10000, p_7500_10000 = wilcoxon(df['KS_7500'], df['KS_10000'], alternative='two-sided')
    lines.append("\n⚖️ Wilcoxon Signed-Rank Results (two-sided):")
    lines.append(f"   5000 vs 7500 -> Statistic: {w_5000_7500:.4f}, P-Value: {p_5000_7500:.4e}, Bonferroni corrected: {p_5000_7500 * 2:.4e}")
    lines.append(f"   7500 vs 10000 -> Statistic: {w_7500_10000:.4f}, P-Value: {p_7500_10000:.4e}, Bonferroni corrected: {p_7500_10000 * 2:.4e}")

    print("\n".join(lines))
    with open(summary_file, 'a') as summary:
        summary.writelines(line + "\n" for line in lines)


if __name__ == "__main__":
    paired_k_s_main()
