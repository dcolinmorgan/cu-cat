"""Plot skrub vs cu-cat comparison showing crossover point.

    python examples/plot_comparison.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent

# Measured data (hashing=True, n_components=10, max_iter=5)
# Format: (rows, unique, skrub_total, cucat_cpu_total, cucat_gpu_total)

DATA = [
    # rows, unique, skrub, cucat_cpu, cucat_gpu
    (1_000, 100, 0.06, 0.06, None),
    (5_000, 500, 0.30, 0.27, None),
    (10_000, 1_000, 0.72, 0.60, None),
    (25_000, 2_500, 2.30, 1.40, None),
    (50_000, 5_000, 5.40, 3.20, 3.1),      # GPU: 2.0+1.1
    (100_000, 10_000, 11.9, 8.70, 4.8),    # GPU: 2.6+2.2
    (200_000, 20_000, 26.9, 26.9, None),
    (500_000, 50_000, None, None, 25.7),   # GPU: 14.7+11.0, skrub crashes
    (1_000_000, 100_000, None, None, 53.7), # GPU: 30.7+23.0
]

rows = [d[0] for d in DATA]
unique = [d[1] for d in DATA]
skrub = [d[2] for d in DATA]
cucat_cpu = [d[3] for d in DATA]
cucat_gpu = [d[4] for d in DATA]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Total time vs unique strings
ax1.set_title("Total Time (fit + transform) vs Cardinality", fontsize=12)

# skrub (only where it works)
x_s = [u for u, s in zip(unique, skrub) if s is not None]
y_s = [s for s in skrub if s is not None]
ax1.plot(x_s, y_s, "o-", color="#e74c3c", label="skrub (CPU)", linewidth=2, markersize=8)

# Mark where skrub crashes
crash_x = 50_000
ax1.axvline(x=crash_x, color="#e74c3c", linestyle="--", alpha=0.5)
ax1.annotate("skrub crashes\nat 50k unique", xy=(crash_x, 30), fontsize=9,
             color="#e74c3c", ha="left")

# cu-cat CPU
x_c = [u for u, c in zip(unique, cucat_cpu) if c is not None]
y_c = [c for c in cucat_cpu if c is not None]
ax1.plot(x_c, y_c, "s-", color="#3498db", label="cu-cat (CPU)", linewidth=2, markersize=8)

# cu-cat GPU
x_g = [u for u, g in zip(unique, cucat_gpu) if g is not None]
y_g = [g for g in cucat_gpu if g is not None]
ax1.plot(x_g, y_g, "^-", color="#27ae60", label="cu-cat (GPU T4)", linewidth=2, markersize=10)

ax1.set_xlabel("Unique strings (cardinality)", fontsize=11)
ax1.set_ylabel("Total time (seconds)", fontsize=11)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.grid(alpha=0.3, which="both")
ax1.legend(fontsize=10)

# Right plot: Speedup vs skrub
ax2.set_title("Speedup vs skrub (higher = better)", fontsize=12)

# Where we have both skrub and cu-cat measurements
common = [(u, s, c, g) for u, s, c, g in zip(unique, skrub, cucat_cpu, cucat_gpu) 
          if s is not None and (c is not None or g is not None)]

x_common = [c[0] for c in common]
speedup_cpu = [c[1]/c[2] if c[2] else None for c in common]
speedup_gpu = [c[1]/c[3] if c[3] else None for c in common]

# CPU speedup
x_cpu = [x for x, s in zip(x_common, speedup_cpu) if s is not None]
y_cpu = [s for s in speedup_cpu if s is not None]
ax2.bar([i-0.2 for i in range(len(x_cpu))], y_cpu, width=0.4, color="#3498db", label="cu-cat CPU")

# GPU speedup  
x_gpu = [x for x, s in zip(x_common, speedup_gpu) if s is not None]
y_gpu = [s for s in speedup_gpu if s is not None]
offset = len(y_cpu) - len(y_gpu)
ax2.bar([i+0.2+offset for i in range(len(x_gpu))], y_gpu, width=0.4, color="#27ae60", label="cu-cat GPU")

ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
ax2.set_xticks(range(len(x_common)))
ax2.set_xticklabels([f"{x//1000}k" for x in x_common], fontsize=9)
ax2.set_xlabel("Unique strings", fontsize=11)
ax2.set_ylabel("Speedup (skrub_time / cu-cat_time)", fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3, axis="y")

# Add note about GPU scaling beyond skrub limits
ax2.annotate("GPU scales to 100k+ unique\nwhere skrub cannot run",
             xy=(0.95, 0.95), xycoords="axes fraction",
             fontsize=9, ha="right", va="top",
             bbox=dict(boxstyle="round", facecolor="#27ae60", alpha=0.2))

plt.tight_layout()
plt.savefig(OUT / "skrub_comparison.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT / 'skrub_comparison.png'}")
