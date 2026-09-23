# Benchmarks: cu-cat vs skrub

All benchmarks run with:
- `hashing=True`
- `n_components=10`
- `max_iter=5`
- `random_state=0`

## Hardware

- **GPU:** Google Colab T4 (15 GB VRAM)
- **CPU:** Google Colab CPU runtime (12 GB RAM)

## Summary

| Unique strings | skrub (CPU) | cu-cat (GPU T4) | Speedup |
|----------------|-------------|-----------------|---------|
| 100 | 0.06s | 6.2s | 0.01x (GPU warmup) |
| 500 | 0.30s | 0.80s | 0.4x |
| 1,000 | 0.72s | 0.87s | 0.8x |
| 2,500 | 2.30s | 1.93s | 1.2x |
| 5,000 | 5.40s | 3.10s | **1.7x** |
| 10,000 | 11.9s | 4.80s | **2.5x** |
| 20,000 | 26.9s | ~10s | **2.7x** |
| 50,000 | CRASH | 25.7s | ∞ |
| 100,000 | CRASH | 53.7s | ∞ |
| 200,000 | CRASH | 125s | ∞ |
| 500,000 | CRASH | 306s | ∞ |
| 1,000,000 | CRASH | 590s | ∞ |

## Detailed Measurements

### Small Scale (GPU overhead visible)

| Rows | Unique | skrub fit | skrub transform | GPU fit | GPU transform |
|------|--------|-----------|-----------------|---------|---------------|
| 1,000 | 100 | 0.03s | 0.03s | 6.08s* | 0.16s |
| 5,000 | 500 | 0.27s | 0.03s | 0.61s | 0.19s |
| 10,000 | 1,000 | 0.56s | 0.16s | 0.58s | 0.29s |
| 25,000 | 2,500 | 1.50s | 0.80s | 1.05s | 0.88s |

*First GPU call includes CUDA context initialization (~6s one-time cost)

### Medium Scale (crossover region)

| Rows | Unique | skrub fit | skrub transform | GPU fit | GPU transform |
|------|--------|-----------|-----------------|---------|---------------|
| 50,000 | 5,000 | 2.20s | 3.20s | 2.0s | 1.1s |
| 100,000 | 10,000 | 5.30s | 6.60s | 2.6s | 2.2s |
| 200,000 | 20,000 | 12.0s | 14.9s | — | — |

### Large Scale (skrub crashes)

skrub fails with `ValueError: NumPy boolean array indexing assignment` at ~50k unique strings.

| Rows | Unique | GPU fit | GPU transform | Total |
|------|--------|---------|---------------|-------|
| 500,000 | 50,000 | 14.7s | 11.0s | 25.7s |
| 1,000,000 | 100,000 | 30.7s | 23.0s | 53.7s |
| 2,000,000 | 200,000 | 60.1s | 64.9s | 125.0s |
| 5,000,000 | 500,000 | 151.8s | 153.8s | 305.6s |
| 10,000,000 | 1,000,000 | 303.4s | 287.1s | 590.5s |

## Key Findings

1. **CUDA warmup:** First GPU call takes ~6s extra for context initialization. This is a one-time cost per process.

2. **Crossover point:** GPU becomes faster than skrub CPU around 2,500-5,000 unique strings.

3. **Scaling advantage:** At 10k unique, GPU is 2.5x faster. The gap widens with scale.

4. **skrub limit:** skrub crashes at ~50k unique strings due to a NumPy indexing bug. cu-cat GPU continues scaling.

5. **Extreme scale:** cu-cat GPU handles 10M rows × 1M unique strings in ~10 minutes on a T4.

## When to Use GPU

- **Use CPU (or skrub):** < 1,000 unique strings, or when GPU is unavailable
- **Use GPU:** > 5,000 unique strings, especially if you need to scale beyond 20k unique

## Reproducing

```python
import time
import numpy as np
import cudf  # or pandas for CPU
from cu_cat import GapEncoder

# Generate test data
rng = np.random.default_rng(42)
n_rows, n_unique = 100_000, 10_000
pool = [f"category_{i}" for i in range(n_unique)]
data = rng.choice(pool, size=n_rows)
df = cudf.DataFrame({"text": data})  # or pd.DataFrame for CPU

# Benchmark
enc = GapEncoder(n_components=10, hashing=True, max_iter=5, random_state=0)
t0 = time.time(); enc.fit(df); print(f"fit: {time.time()-t0:.1f}s")
t0 = time.time(); enc.transform(df); print(f"transform: {time.time()-t0:.1f}s")
```
