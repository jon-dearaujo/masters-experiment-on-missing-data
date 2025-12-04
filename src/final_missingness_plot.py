import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# --- Configuration ---
RESULTS_FILE = "../results/final_missingness_impact.csv"
OUTPUT_IMAGE = "../results/missingness_decay_trend.png"

# --- 1. Load Data ---
try:
    df = pd.read_csv(RESULTS_FILE)
except FileNotFoundError:
    print("Error: CSV file not found.")
    exit()

# Ensure Missingness is treated as a percentage for plotting
# If your CSV has 0.1, 0.2, multiply by 100 for prettier labels
if df['Missingness'].max() <= 1.0:
    df['Missingness'] = df['Missingness'] * 100

# --- 2. Plotting ---
plt.figure(figsize=(10, 6))

# Set style
sns.set_style("whitegrid")

# Create Line Plot with Standard Deviation (sd) as the error band
# 'marker="o"' adds dots at the mean points
sns.lineplot(
    data=df,
    x='Missingness',
    y='Accuracy',
    errorbar='sd',  # Draws the shaded region for Standard Deviation
    marker='o',
    markersize=8,
    linewidth=2.5,
    color='#C0392B' # A nice strong "decay" red color
)

# --- 3. Formatting ---
plt.title('Impact of Data Missingness on Model Utility (LightGBM)', fontsize=14, pad=15)
plt.ylabel('Downstream Accuracy', fontsize=12)
plt.xlabel('Source Data Missingness (%)', fontsize=12)

# Format X axis as percentages
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
# Set X ticks explicitly to your levels (10, 20, 30, 40)
plt.xticks([10, 20, 30, 40])

# Add limits if needed (e.g., 0 to 1 for accuracy)
# plt.ylim(0, 1)

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300)
print(f"✅ Chart saved to: {OUTPUT_IMAGE}")
plt.show()