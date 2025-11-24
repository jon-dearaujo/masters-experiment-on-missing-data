import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Update this filename if you named it differently
RESULTS_FILE = "../results/ks-20251122_160745/multi_epoch_validation_ks_scores.csv"
OUTPUT_IMAGE = "../results/ks-20251122_160745/ks_boxplot_comparison.png"

# --- 1. Load Data ---
if not os.path.exists(RESULTS_FILE):
    print(f"Error: Could not find results file at {RESULTS_FILE}")
    exit(1)

df = pd.read_csv(RESULTS_FILE)

# Remove the 'Iteration' column if it exists, as we only need the scores
if 'Iteration' in df.columns:
    df = df.drop(columns=['Iteration'])

# --- 2. Prepare Data for Plotting ---
# "Melt" the dataframe from wide format (cols: KS_1000, KS_2000) to long format
# This makes it easier for Seaborn to plot
df_melted = df.melt(var_name='Epochs', value_name='K-S Score')

# Clean up the labels (e.g., change "KS_1000" to "1000")
df_melted['Epochs'] = df_melted['Epochs'].str.replace('KS_', '')

print("Data loaded and prepared. Generating plot...")

# --- 3. Create the Box Plot ---
plt.figure(figsize=(10, 6))

# Define a custom color palette (optional, but looks nice)
custom_palette = sns.color_palette("Blues", n_colors=len(df.columns))

# Draw the box plot
sns.boxplot(
    data=df_melted,
    x='Epochs',
    y='K-S Score',
    palette=custom_palette,
    width=0.5,       # Width of the boxes
    linewidth=1.5,   # Thickness of the lines
    fliersize=5      # Size of outlier diamonds
)

# Add a swarmplot on top to show individual data points (optional but recommended)
sns.swarmplot(
    data=df_melted,
    x='Epochs',
    y='K-S Score',
    color='0.25',
    alpha=0.6,
    size=4
)

# --- 4. Styling ---
plt.title('Distribution of Kolmogorov-Smirnov Scores by Training Epochs\n(Lower Score = Better Statistical Fidelity)', fontsize=14, pad=15)
plt.ylabel('K-S Distance (Avg across numerical cols)', fontsize=12)
plt.xlabel('Training Epochs', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# --- 5. Save ---
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300) # 300 dpi is print-quality for thesis
print(f"✅ Box plot saved to: {OUTPUT_IMAGE}")
plt.show()