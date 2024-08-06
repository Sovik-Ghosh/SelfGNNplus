import numpy as np
from scipy.stats import ttest_rel, wilcoxon

# Sample data
data1 = [0.391, 0.240, 0.487, 0.264, 0.637, 0.437, 0.745, 0.465, 0.239, 0.128, 0.341, 0.154, 0.365, 0.201, 0.509, 0.237]
data2 = [0.393, 0.244, 0.492, 0.265, 0.675, 0.465, 0.780, 0.492, 0.257, 0.143, 0.373, 0.172, 0.381, 0.212, 0.529, 0.249]

# Convert to numpy arrays
data1 = np.array(data1)
data2 = np.array(data2)

# Calculate p-values
t_stat, p_val_ttest = ttest_rel(data1, data2)
stat_wilcoxon, p_val_wilcoxon = wilcoxon(data1, data2)

# Calculate differences
differences = data2 - data1

# Determine mean difference
mean_difference = np.mean(differences)

# Count positive and negative differences
positive_count = np.sum(differences > 0)
negative_count = np.sum(differences < 0)

# Output results
print(f"Paired T-Test P-Value: {p_val_ttest:.10f}")
print(f"Wilcoxon Signed-Rank Test P-Value: {p_val_wilcoxon:.10f}")
print(f"Mean Difference: {mean_difference:.10f}")
print(f"Positive Differences: {positive_count}")
print(f"Negative Differences: {negative_count}")

# Determine overall trend
if mean_difference > 0:
    print("Overall Trend: Positive")
else:
    print("Overall Trend: Negative")
