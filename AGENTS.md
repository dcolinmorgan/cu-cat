# Agent Instructions

## Guidelines

- Read the project README and any existing docs before making changes
- Run the project's build/test commands before committing (check package.json, Makefile, pyproject.toml, Cargo.toml)
- Keep changes minimal and focused on the task
- Prefer early returns over nested conditionals
- Handle error states explicitly
- Use semantic HTML and ARIA attributes for accessibility in frontend code
- Follow existing code style and conventions in the repo
- Do not introduce new dependencies without justification

## Verification

- Run linting and type checks before committing
- Run tests relevant to changed code
- Verify the build passes

## Git

- Write clear, concise commit messages
- Stage only files related to the current task
- Do not push to main/master without explicit permission

## GPU testing

GitHub-hosted runners have no NVIDIA GPU, so `test-rapids` in CI only runs on a
self-hosted runner labelled `gpu`, gated behind the repository variable
`HAS_GPU_RUNNER=true`. Without that it is skipped, never silently passed.

To verify GPU changes before a runner exists, run the suite on any CUDA box
(a Colab T4 is enough):

```bash
pip install -e '.[test]'      # cudf/cuml are preinstalled on Colab GPU runtimes
python -m pytest cu_cat/tests -q
./bin/test-rapids.sh
```

Facts worth keeping in mind when changing `_gap_encoder.py`:

- The dominant GPU allocation is the dense `Ht @ W` term: `n_unique *
  hashing_n_features * 8` bytes, with about three copies live at once. Row
  count barely matters; cardinality sets the ceiling.
- Per-row Python loops are the usual cause of a slow GPU path. Every
  bottleneck found so far was host-side Python touching per-element data while
  the GPU sat under 10% utilisation, never GPU math.
- Measure before optimising. Three separate hypotheses about which line was
  slow were wrong; a stopwatch around each operation found the real one each
  time.
