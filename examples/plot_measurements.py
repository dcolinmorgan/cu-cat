"""Plot the GPU measurements collected while fixing the scaling ceiling.

Every number here was measured on a Colab T4 (15 GB, cudf/cuml 26.02) with
synthetic data: 220-character strings, n_components=10, max_iter=2,
hashing=True. Single runs, no repeats, so treat them as orders of magnitude
rather than precise timings.

    python examples/plot_measurements.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
OOM = "OOM"

# --- measured data ----------------------------------------------------------

# Rows, at fixed 200-category cardinality. "before" carried the discarded
# unq_V[lookup] expansion (~2 KB of GPU memory per input row).
# The 10k "after" point was the first call in its process and carries CUDA
# context + cuml warm-up (7.9s), so it is excluded rather than plotted as a
# regression it is not.
ROWS = [100_000, 1_000_000, 5_000_000, 8_000_000]
ROWS_BEFORE = [0.3, 1.3, 6.0, OOM]
ROWS_AFTER = [0.3, 1.2, 6.7, 16.8]

# Distinct strings, at 1M rows. The dense Ht @ W term is n_unique x vocab.
# After-fix measurements use the unique-axis chunking + OOM recovery.
CARD = [1_000, 10_000, 50_000, 200_000, 400_000]
CARD_BEFORE = [2.1, 3.2, 9.5, OOM, OOM]     # whole-matrix updates only
CARD_AFTER = [2.1, 3.2, 8.1, 30.3, 56.0]    # chunked unique-axis, vectorised

# Transform times at 100k rows (after the transform speedup)
TRANSFORM_100K = [None, None, 4.8, 7.7, 9.1]  # s, for 50k/200k/400k unique

# Hash width at 200k distinct, 1M rows. At 4096 the dense term is ~19.5 GB,
# beyond the T4's 15 GB, so chunking kicks in. With unique-axis chunking this
# is now 30.3s instead of 100.2s with row-axis chunking.
WIDTH = [512, 1024, 4096]
WIDTH_FIT = [14.1, 17.2, 30.3]


def _plot_pair(ax, x, before, after, xlabel, title):
    ok_b = [(a, b) for a, b in zip(x, before) if b != OOM]
    ok_a = [(a, b) for a, b in zip(x, after) if b != OOM]
    ax.plot(*zip(*ok_b), "o--", color="#c0392b", label="before")
    ax.plot(*zip(*ok_a), "o-", color="#27ae60", label="after")
    for xi, b, a in zip(x, before, after):
        if b == OOM:
            # mark where the old code died, at the height the new code reached
            ax.plot([xi], [a], "X", color="#c0392b", markersize=15, zorder=5)
            ax.annotate("before: OOM\nafter: %.1fs" % a, xy=(xi, a),
                        xytext=(-10, 14), textcoords="offset points",
                        color="#c0392b", fontweight="bold", ha="right",
                        fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fit time (s, log)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend()


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
_plot_pair(axes[0], ROWS, ROWS_BEFORE, ROWS_AFTER, "rows (log)",
           "Rows are cheap\n200 distinct strings, Colab T4")
_plot_pair(axes[1], CARD, CARD_BEFORE, CARD_AFTER, "distinct strings (log)",
           "Cardinality sets the ceiling\n1M rows, Colab T4")
fig.suptitle("cu-cat GapEncoder scaling: before vs after the memory fixes", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "scaling_before_after.png", dpi=140)

fig, ax = plt.subplots(figsize=(6.4, 4.4))
bars = ax.bar([str(w) for w in WIDTH], WIDTH_FIT,
              color=["#27ae60", "#27ae60", "#e67e22"])
for bar, v in zip(bars, WIDTH_FIT):
    ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}s",
            ha="center", va="bottom", fontsize=10)
ax.set_xlabel("hashing_n_features")
ax.set_ylabel("fit time (s)")
ax.set_title("Narrowing the hash width is the high-cardinality lever\n"
             "1M rows, 200k distinct strings, Colab T4", fontsize=11)
ax.annotate("does not fit in 15 GB;\nfalls back to chunked updates",
            xy=(1.75, WIDTH_FIT[2] * 0.75), xytext=(0.0, 62), fontsize=9,
            color="#e67e22", ha="left",
            arrowprops=dict(arrowstyle="->", color="#e67e22"))
ax.set_ylim(0, 118)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "hash_width.png", dpi=140)

print("wrote", OUT / "scaling_before_after.png")
print("wrote", OUT / "hash_width.png")
