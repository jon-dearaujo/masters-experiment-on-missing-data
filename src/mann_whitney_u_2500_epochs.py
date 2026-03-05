import pandas as pd
from scipy.stats import mannwhitneyu

df = pd.read_csv("../results/ks-20251122_160745/multi_epoch_validation_ks_scores.csv")

ks_2500 = df["KS_2500"]
ks_2000 = df["KS_2000"]
ks_5000 = df["KS_5000"]

# 2500 better than 2000? (lower KS is better)
u_1, p_1 = mannwhitneyu(ks_2500, ks_2000, alternative="less")

# 2500 worse than 5000?
u_2, p_2 = mannwhitneyu(ks_2500, ks_5000, alternative="greater")

print("2500 vs 2000 (is 2500 better?) p =", p_1)
print("2500 vs 5000 (is 2500 worse?) p =", p_2)

'''
Results:

2500 vs 2000 (is 2500 better?) p = 0.13859125767474645
2500 vs 5000 (is 2500 worse?) p = 0.05767360051139876

Interpretation:
- The p-value of 0.1386 for the 2500 vs 2000 comparison suggests that there is no statistically significant evidence to conclude that 2500 epochs is better than 2000 epochs in terms of KS scores.

- The p-value of 0.0577 for the 2500 vs 5000 comparison is close to the conventional threshold of 0.05, but it does not quite reach statistical significance. This suggests that there is not strong evidence to conclude that 2500 epochs is worse than 5000 epochs, although it is borderline.
'''
