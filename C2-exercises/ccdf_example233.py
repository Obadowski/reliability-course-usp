import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add the folder to path if running from a different directory
# sys.path.insert(0, "/home/obad/Doutorado/phd-knowledge-base/courses/Reliability")
from weibull import weibull, weibull_mean

# --- Your empirical dataset ---
# Replace with your actual failure/event times
empirical_times = np.array([
    1643,
    1664,
    2083,
    3625,
    7230,
    9095,
    9968,
    11689,
    12989,
    13622,
    13953,
    14527,
    15263,
    15428,
    15503,
    15629,
    16342,
    16584,
    17374,
    18571,
    19739,
    19936,
    20102,
    20832,
    23378,
    23612,
    23678,
    23971,
    24341,
    26964
])

# Example dataset
# empirical_times = np.array([120, 340, 500, 720, 950, 1100, 1400, 1800])

# Sort and compute empirical CCDF (survival function)
t_sorted = np.sort(empirical_times)
n = len(t_sorted)
ccdf_empirical = 1 - np.arange(1, n + 1) / n   # 1 - empirical CDF

# --- Weibull fit (replace alpha/beta with fitted values) ---
alpha, beta = 18400, 1.5
t_range = np.linspace(0, t_sorted[-1] * 1.2, 500)
ccdf_weibull = weibull(t_range, alpha, beta)["R"]   # R(t) = 1 - F(t) = CCDF

# --- Output dataframe: empirical t vs empirical CCDF vs fitted Weibull CCDF ---
weibull_at_empirical = weibull(t_sorted, alpha, beta)["R"]
rank = np.arange(1, n + 1)
KS_rank = rank / n
KS_rank_minus = (rank - 1) / n

df = pd.DataFrame({
    "t":              t_sorted,
    "rank":           rank,
    "KS_rank":        KS_rank,
    "KS_rank_minus":  KS_rank_minus,
    "ccdf_weibull":   1-weibull_at_empirical,
    "KS_statistic_1": np.abs((1 - weibull_at_empirical) - KS_rank),
    "KS_statistic_2": np.abs((1 - weibull_at_empirical) - KS_rank_minus),
})
print(df.to_string(index=False))

# KS Statistics: max distance between empirical CCDF and Weibull CCDF at the empirical points
print(f"KS Statistic (1) = {df['KS_statistic_1'].max():.4f}")
print(f"KS Statistic (2) = {df['KS_statistic_2'].max():.4f}")

# Alpha critical obtained from KS tables on Modarres book, for n=30:
D_alpha02 = 0.190
D_alpha01 = 0.208
print(f"Critical D at α=0.02: {D_alpha02:.4f}")
print(f"Critical D at α=0.01: {D_alpha01:.4f}")

df.to_csv("ccdf_comparison.csv", index=False)

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.step(t_sorted, ccdf_empirical, where="post", label="Empirical CCDF", color="steelblue")
plt.plot(t_range, ccdf_weibull, "--", label=f"Weibull (α={alpha}, β={beta})", color="tomato")
plt.xlabel("t")
plt.ylabel("P(T > t)")
plt.title("Empirical vs Weibull CCDF")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
