#!/bin/bash
set -ex

# Run from project root
# - Args get passed to pytest phase
# Non-zero exit code on fail

# Deeping tests of gapencoder, speed tests

python -m pytest --version

# The whole CPU suite, including dedup and LRUDict. Tests that need a
# network download or a GPU are skipped by the tests themselves; this
# script does not filter them out because the OpenML fixtures are part
# of test_gap_encoder and already ran here.
python -B -m pytest -vv cu_cat/tests "$@"
