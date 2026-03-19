import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RESULTS_FILE = "results/paired-ks-20260318_084543/paired_multi_epoch_validation_ks_scores.csv"
OUTPUT_IMAGE = "results/paired-ks-20260318_084543/paired_ks_boxplot_comparacao.png"


if not os.path.exists(RESULTS_FILE):
    print(f"Erro: arquivo de resultados não encontrado em {RESULTS_FILE}")
    raise SystemExit(1)

df = pd.read_csv(RESULTS_FILE)

if 'Iteration' in df.columns:
    df = df.drop(columns=['Iteration'])

df_melted = df.melt(var_name='Épocas', value_name='Escore K-S')
df_melted['Épocas'] = df_melted['Épocas'].str.replace('KS_', '')

print("Dados carregados. Gerando gráfico pareado...")

plt.figure(figsize=(10, 6))

custom_palette = sns.color_palette("coolwarm", n_colors=len(df.columns))

sns.boxplot(
    data=df_melted,
    x='Épocas',
    y='Escore K-S',
    palette=custom_palette,
    width=0.5,
    linewidth=1.5,
    fliersize=5
)

sns.swarmplot(
    data=df_melted,
    x='Épocas',
    y='Escore K-S',
    color='0.25',
    alpha=0.6,
    size=4
)

plt.title('Distribuição dos escores K-S por épocas\nExperimento pareado (2000 × 2500 × 5000)', fontsize=14, pad=15)
plt.ylabel('Distância K-S média (menor é melhor)', fontsize=12)
plt.xlabel('Épocas de treinamento', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300)
print(f"✅ Gráfico salvo em: {OUTPUT_IMAGE}")
plt.show()
