import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ler CSV (primeira linha já é interpretada como cabeçalho)
df = pd.read_csv("../results/ks-20251122_160745/multi_epoch_validation_ks_scores.csv")

# Selecionar apenas as colunas de K-S
ks_cols = [col for col in df.columns if col.startswith("KS_")]

# Converter de formato wide para long
long_df = df[ks_cols].melt(var_name="Epoca", value_name="Distancia_KS")

# Extrair número da época (ex: KS_2500 -> 2500)
long_df["Epoca"] = long_df["Epoca"].str.replace("KS_", "", regex=False).astype(int)

# Ordenar para garantir a sequência correta no gráfico
long_df = long_df.sort_values("Epoca")

plt.figure(figsize=(9,5))

sns.pointplot(
    data=long_df,
    x="Epoca",
    y="Distancia_KS",
    errorbar="sd",
    capsize=0.2
)

plt.title("Distância K-S ao Longo das Épocas de Treinamento")
plt.xlabel("Número de Épocas de Treinamento")
plt.ylabel("Distância Kolmogorov-Smirnov (quanto menor, melhor)")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()