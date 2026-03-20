import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


UNPAIRED_RESULTS_FILE = "../results/ks-20251122_160745/multi_epoch_validation_ks_scores.csv"
PAIRED_MID_RESULTS_FILE = "../results/paired-ks-20260318_084543/paired_multi_epoch_validation_ks_scores.csv"
PAIRED_PLATEAU_RESULTS_FILE = "../results/paired-ks-20260319_105042/paired_multi_epoch_validation_ks_scores.csv"


def ensure_output_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("../results", "final_epoch_plots", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)


def plot_global_trend(df: pd.DataFrame, out_path: str) -> None:
    value_cols = sorted([col for col in df.columns if col.startswith('KS_')], key=lambda c: int(c.split('_')[1]))
    melted = df.melt(id_vars='Iteration', value_vars=value_cols, var_name='Época', value_name='Escore K-S')
    melted['Época'] = melted['Época'].str.replace('KS_', '').astype(int)

    summary = melted.groupby('Época')['Escore K-S'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(10, 6))
    plt.errorbar(summary['Época'], summary['mean'], yerr=summary['std'], fmt='-o', capsize=4)
    plt.title('Tendência global dos escores K-S (experimento não pareado)')
    plt.xlabel('Épocas de treinamento')
    plt.ylabel('Média do K-S (com desvio padrão)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_unpaired_boxplot(df: pd.DataFrame, out_path: str) -> None:
    value_cols = [col for col in df.columns if col.startswith('KS_')]
    melted = df.melt(id_vars='Iteration', value_vars=value_cols, var_name='Época', value_name='Escore K-S')
    melted['Época'] = melted['Época'].str.replace('KS_', '')

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=melted, x='Época', y='Escore K-S', palette='Blues')
    sns.swarmplot(data=melted, x='Época', y='Escore K-S', color='0.2', alpha=0.5, size=3)
    plt.title('Distribuição dos escores K-S por época (não pareado)')
    plt.xlabel('Épocas de treinamento')
    plt.ylabel('Escore K-S médio (menor é melhor)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_paired_lines(df: pd.DataFrame, epoch_cols: list[str], title: str, out_path: str) -> None:
    epochs = [int(col.split('_')[1]) for col in epoch_cols]
    plt.figure(figsize=(10, 6))

    for _, row in df.iterrows():
        plt.plot(epochs, row[epoch_cols], color='gray', alpha=0.4, linewidth=1)

    mean_values = df[epoch_cols].mean()
    plt.plot(epochs, mean_values, color='red', linewidth=2.5, marker='o', label='Média')
    plt.title(title)
    plt.xlabel('Épocas de treinamento')
    plt.ylabel('Escore K-S (menor é melhor)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_effect_size_deltas(df_mid: pd.DataFrame, df_plateau: pd.DataFrame, out_path: str) -> None:
    merged = df_mid.merge(df_plateau, on='Iteration', how='inner')
    deltas = []

    deltas.append(pd.DataFrame({
        'Comparação': '2500 - 2000',
        'Delta_KS': merged['KS_2500_mid'] - merged['KS_2000_mid']
    }))

    deltas.append(pd.DataFrame({
        'Comparação': '5000 - 2500',
        'Delta_KS': merged['KS_5000_mid'] - merged['KS_2500_mid']
    }))

    deltas.append(pd.DataFrame({
        'Comparação': '7500 - 5000',
        'Delta_KS': merged['KS_7500_plat'] - merged['KS_5000_plat']
    }))

    deltas.append(pd.DataFrame({
        'Comparação': '10000 - 7500',
        'Delta_KS': merged['KS_10000_plat'] - merged['KS_7500_plat']
    }))

    df_deltas = pd.concat(deltas, ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_deltas, x='Comparação', y='Delta_KS', palette='Set2')
    sns.stripplot(data=df_deltas, x='Comparação', y='Delta_KS', color='0.2', alpha=0.6, size=4)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Distribuição das diferenças de K-S (Δ)')
    plt.xlabel('Comparação entre épocas')
    plt.ylabel('Delta K-S (negativo = melhora)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def render_all_charts() -> None:
    output_dir = ensure_output_dir()
    print(f"📁 Salvando gráficos em: {output_dir}")

    df_unpaired = load_csv(UNPAIRED_RESULTS_FILE)
    df_paired_mid = load_csv(PAIRED_MID_RESULTS_FILE).rename(columns={
        'KS_2000': 'KS_2000_mid',
        'KS_2500': 'KS_2500_mid',
        'KS_5000': 'KS_5000_mid'
    })
    df_paired_plateau = load_csv(PAIRED_PLATEAU_RESULTS_FILE).rename(columns={
        'KS_5000': 'KS_5000_plat',
        'KS_7500': 'KS_7500_plat',
        'KS_10000': 'KS_10000_plat'
    })

    plot_global_trend(df_unpaired, os.path.join(output_dir, 'global_trend_unpaired.png'))
    print('✅ Tendência global gerada')

    plot_unpaired_boxplot(df_unpaired, os.path.join(output_dir, 'unpaired_boxplot.png'))
    print('✅ Boxplot não pareado gerado')

    plot_paired_lines(
        df_paired_mid,
        ['KS_2000_mid', 'KS_2500_mid', 'KS_5000_mid'],
        'Comparação pareada: 2000 × 2500 × 5000',
        os.path.join(output_dir, 'paired_lines_mid.png')
    )
    print('✅ Comparação pareada 2000–5000 gerada')

    plot_paired_lines(
        df_paired_plateau,
        ['KS_5000_plat', 'KS_7500_plat', 'KS_10000_plat'],
        'Validação do platô: 5000 × 7500 × 10000',
        os.path.join(output_dir, 'paired_lines_plateau.png')
    )
    print('✅ Comparação pareada 5000–10000 gerada')

    plot_effect_size_deltas(
        df_paired_mid,
        df_paired_plateau,
        os.path.join(output_dir, 'paired_effect_deltas.png')
    )
    print('✅ Gráfico de deltas gerado')


if __name__ == '__main__':
    render_all_charts()
