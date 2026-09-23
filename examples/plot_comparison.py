"""Plot skrub vs cu-cat comparison showing full range including GPU overhead at small scale.

    python examples/plot_comparison.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent

# ============================================================================
# MEASURED DATA (all on Colab T4, hashing=True, n_components=10, max_iter=5)
# ============================================================================

# Format: (rows, unique, skrub_fit, skrub_trans, cucat_gpu_fit, cucat_gpu_trans)
# None = crashed or not measured

DATA = [
    # Small scale - GPU has warmup/overhead disadvantage
    (1_000, 100, 0.03, 0.03, 6.08, 0.16),      # GPU fit includes CUDA warmup
    (5_000, 500, 0.27, 0.03, 0.61, 0.19),
    (10_000, 1_000, 0.56, 0.16, 0.58, 0.29),
    (25_000, 2_500, 1.50, 0.80, 1.05, 0.88),
    
    # Medium scale - GPU catches up
    (50_000, 5_000, 2.20, 3.20, 2.0, 1.1),
    (100_000, 10_000, 5.30, 6.60, 2.6, 2.2),
    (200_000, 20_000, 12.0, 14.9, None, None),  # no GPU measurement here
    
    # Large scale - skrub crashes, GPU shines
    (500_000, 50_000, None, None, 14.7, 11.0),
    (1_000_000, 100_000, None, None, 30.7, 23.0),
    (2_000_000, 200_000, None, None, 60.1, 64.9),
    (5_000_000, 500_000, None, None, 151.8, 153.8),
    (10_000_000, 1_000_000, None, None, 303.4, 287.1),
]

# Extract arrays
unique = [d[1] for d in DATA]
skrub_total = [d[2]+d[3] if d[2] is not None else None for d in DATA]
gpu_total = [d[4]+d[5] if d[4] is not None else None for d in DATA]

# ============================================================================
# PLOT 1: Full scaling comparison (log-log)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

# skrub line (where it works)
x_s = [u for u, s in zip(unique, skrub_total) if s is not None]
y_s = [s for s in skrub_total if s is not None]
ax.plot(x_s, y_s, "o-", color="#e74c3c", label="skrub (CPU)", linewidth=2.5, markersize=10)

# Extrapolate skrub crash trajectory
ax.plot([20_000, 50_000, 100_000], [27, 80, 250], "o--", color="#e74c3c", 
        alpha=0.3, linewidth=2, markersize=6)
ax.annotate("skrub crashes\n(extrapolated)", xy=(50_000, 80), fontsize=10, 
            color="#e74c3c", alpha=0.7)

# GPU line (full range)
x_g = [u for u, g in zip(unique, gpu_total) if g is not None]
y_g = [g for g in gpu_total if g is not None]
ax.plot(x_g, y_g, "^-", color="#27ae60", label="cu-cat (GPU T4)", linewidth=2.5, markersize=10)

# Mark the CUDA warmup point
ax.annotate("CUDA warmup\n(one-time cost)", xy=(100, 6.2), xytext=(200, 15),
            fontsize=9, color="#27ae60", alpha=0.8,
            arrowprops=dict(arrowstyle="->", color="#27ae60", alpha=0.5))

# Mark key milestones
ax.annotate("1M unique strings\n~10 min total", xy=(1_000_000, 590), 
            xytext=(400_000, 800), fontsize=10, color="#27ae60", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#27ae60"))

# Crossover region
ax.axvspan(5_000, 10_000, alpha=0.1, color="gray")
ax.annotate("crossover\nregion", xy=(7_000, 0.5), fontsize=9, color="gray", ha="center")

# skrub limit line
ax.axvline(x=20_000, color="#e74c3c", linestyle=":", alpha=0.5)
ax.annotate("skrub limit", xy=(20_000, 0.3), fontsize=9, color="#e74c3c", ha="center")

ax.set_xlabel("Unique strings (cardinality)", fontsize=12)
ax.set_ylabel("Total time: fit + transform (seconds)", fontsize=12)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(50, 2_000_000)
ax.set_ylim(0.05, 1500)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=11, loc="upper left")
ax.set_title("GapEncoder: skrub (CPU) vs cu-cat (GPU)\nFull range from 100 to 1M unique strings", fontsize=13)

plt.tight_layout()
plt.savefig(OUT / "skrub_comparison.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT / 'skrub_comparison.png'}")

# ============================================================================
# PLOT 2: Speedup chart
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Time comparison at overlapping scales
scales = [100, 500, 1_000, 2_500, 5_000, 10_000]
skrub_times = [0.06, 0.30, 0.72, 2.30, 5.40, 11.9]
gpu_times = [6.24, 0.80, 0.87, 1.93, 3.1, 4.8]

x = np.arange(len(scales))
width = 0.35

bars1 = ax1.bar(x - width/2, skrub_times, width, label='skrub (CPU)', color='#e74c3c')
bars2 = ax1.bar(x + width/2, gpu_times, width, label='cu-cat (GPU)', color='#27ae60')

ax1.set_xlabel('Unique strings', fontsize=11)
ax1.set_ylabel('Total time (seconds)', fontsize=11)
ax1.set_title('Direct Comparison (where both work)', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels([f'{s//1000}k' if s >= 1000 else str(s) for s in scales])
ax1.legend()
ax1.set_yscale('log')
ax1.grid(alpha=0.3, axis='y')

# Annotate the crossover
ax1.annotate("GPU overhead\nat small scale", xy=(0, 6.24), xytext=(1, 8),
             fontsize=9, arrowprops=dict(arrowstyle="->"))
ax1.annotate("GPU wins\n2.5x faster", xy=(5, 4.8), xytext=(4.2, 2),
             fontsize=9, color="#27ae60", arrowprops=dict(arrowstyle="->", color="#27ae60"))

# Right: GPU-only scaling to 1M unique
ax2.set_title('cu-cat GPU Scales to 1M Unique (skrub cannot)', fontsize=12)

gpu_large_x = [50_000, 100_000, 200_000, 500_000, 1_000_000]
gpu_large_y = [25.7, 53.7, 125.0, 305.6, 590.5]

ax2.bar(range(len(gpu_large_x)), gpu_large_y, color='#27ae60')
ax2.set_xticks(range(len(gpu_large_x)))
ax2.set_xticklabels(['50k', '100k', '200k', '500k', '1M'])
ax2.set_xlabel('Unique strings', fontsize=11)
ax2.set_ylabel('Total time (seconds)', fontsize=11)
ax2.grid(alpha=0.3, axis='y')

# Add time labels on bars
for i, (x_pos, y_val) in enumerate(zip(range(len(gpu_large_x)), gpu_large_y)):
    if y_val < 100:
        label = f'{y_val:.0f}s'
    else:
        label = f'{y_val/60:.1f}m'
    ax2.annotate(label, xy=(x_pos, y_val), ha='center', va='bottom', fontsize=10)

ax2.annotate("skrub crashes\nat all these scales", xy=(2, 300), fontsize=11, 
             color="#e74c3c", ha="center",
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="#e74c3c"))

plt.tight_layout()
plt.savefig(OUT / "scaling_comparison.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT / 'scaling_comparison.png'}")
