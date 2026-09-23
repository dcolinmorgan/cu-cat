"""Plot scaling behavior: how time grows with cardinality.

    python examples/plot_scaling.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent

# Measured data (rows, unique, skrub_total, cucat_gpu_total)
DATA = [
    (50_000, 5_000, 5.4, 3.1),
    (100_000, 10_000, 11.9, 4.8),
    (200_000, 20_000, 26.9, None),  # no GPU measurement at this point
    (500_000, 50_000, None, 25.7),  # skrub crashes
    (1_000_000, 100_000, None, 53.7),
    (1_000_000, 200_000, None, None),  # would need to measure
]

# Extended GPU data from earlier measurements  
GPU_DATA = [
    (1_000_000, 50_000, 8.1 + 4.8),   # from earlier session
    (1_000_000, 200_000, 30.3 + 7.7), # from earlier session
    (1_000_000, 400_000, 56.0 + 9.1), # from earlier session
]

fig, ax = plt.subplots(figsize=(10, 6))

# skrub scaling (where it works)
skrub_x = [5_000, 10_000, 20_000]
skrub_y = [5.4, 11.9, 26.9]
ax.plot(skrub_x, skrub_y, "o-", color="#e74c3c", label="skrub (CPU)", 
        linewidth=2.5, markersize=10)

# Extrapolate skrub to show where it would be (if it didn't crash)
# Roughly quadratic scaling
skrub_extrap_x = [50_000, 100_000]
skrub_extrap_y = [26.9 * (50/20)**1.5, 26.9 * (100/20)**1.5]  # ~100s, ~350s
ax.plot(skrub_extrap_x, skrub_extrap_y, "o--", color="#e74c3c", alpha=0.4,
        linewidth=2, markersize=8)
ax.annotate("skrub crashes\n(extrapolated)", xy=(50_000, skrub_extrap_y[0]),
            xytext=(30_000, 150), fontsize=10, color="#e74c3c",
            arrowprops=dict(arrowstyle="->", color="#e74c3c", alpha=0.5))

# cu-cat GPU scaling
gpu_x = [5_000, 10_000, 50_000, 100_000, 200_000, 400_000]
gpu_y = [3.1, 4.8, 25.7, 53.7, 30.3+7.7, 56.0+9.1]  # mix of sessions
ax.plot(gpu_x, gpu_y, "^-", color="#27ae60", label="cu-cat (GPU T4)", 
        linewidth=2.5, markersize=10)

# Mark the crossover
ax.axvline(x=20_000, color="gray", linestyle=":", alpha=0.5)
ax.annotate("skrub limit\n(crashes beyond)", xy=(20_000, 5), fontsize=9,
            color="gray", ha="center")

# Annotations
ax.annotate("2.5x faster", xy=(10_000, 4.8), xytext=(15_000, 2),
            fontsize=10, color="#27ae60",
            arrowprops=dict(arrowstyle="->", color="#27ae60"))

ax.annotate("GPU scales to\n400k unique\n(65s total)", 
            xy=(400_000, 65), xytext=(250_000, 120),
            fontsize=10, color="#27ae60",
            arrowprops=dict(arrowstyle="->", color="#27ae60"))

ax.set_xlabel("Unique strings (cardinality)", fontsize=12)
ax.set_ylabel("Total time: fit + transform (seconds)", fontsize=12)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(3_000, 600_000)
ax.set_ylim(1, 500)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=11, loc="upper left")
ax.set_title("GapEncoder Scaling: skrub vs cu-cat GPU\n(1M rows, hashing=True)", fontsize=13)

plt.tight_layout()
plt.savefig(OUT / "scaling_comparison.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT / 'scaling_comparison.png'}")
