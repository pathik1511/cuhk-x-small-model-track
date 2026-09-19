#!/usr/bin/env bash
# CUHK-X Small Model Track, verification entry point.
#   ./inference.sh <raw_archive_dir> <output_csv>
# Thin wrapper. All logic is in inference.py so that the Linux verification host and
# the Windows machine that trained the models run byte-identical code.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 inference.py "$@"
