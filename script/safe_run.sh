#!/bin/sh
# Runs a command in its own systemd user unit with a hard memory cap (no swap), lower CPU and IO priority, and a GPU memory cap, so that an
# oversized job is killed alone instead of exhausting the desktop (the kernel OOM killer took VS Code with it on 2026-11-15).
# GPU: no preallocation (the desktop holds about 3.5 GB of VRAM; preallocation left cuBLAS without memory), JAX pool capped at gpu_frac of VRAM.
# usage: sh script/safe_run.sh [mem=20G] [gpu_frac=0.6] COMMAND [ARGS ...]     (run from the repository root)
#   example: sh script/safe_run.sh mem=16G python script/port_check.py ref=pc
MEM=20G; FRAC=0.6
while :; do
  case "$1" in
    mem=*) MEM=${1#mem=}; shift ;;
    gpu_frac=*) FRAC=${1#gpu_frac=}; shift ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { sed -n 2,5p "$0"; exit 2; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -x "$ROOT/.venv/bin/python" ] && PATH="$ROOT/.venv/bin:$PATH"
exec systemd-run --user --pipe --wait --quiet --collect --same-dir \
  -p MemoryMax="$MEM" -p MemorySwapMax=0 -p CPUWeight=50 -p IOWeight=50 -p Nice=5 \
  --setenv=PATH="$PATH" --setenv=HOME="$HOME" \
  ${JAX_PLATFORMS+--setenv=JAX_PLATFORMS="$JAX_PLATFORMS"} ${XLA_FLAGS+--setenv=XLA_FLAGS="$XLA_FLAGS"} ${PAIRL_ROOT+--setenv=PAIRL_ROOT="$PAIRL_ROOT"} ${PAIRL_WORKER_GB+--setenv=PAIRL_WORKER_GB="$PAIRL_WORKER_GB"} ${PYTHONPATH+--setenv=PYTHONPATH="$PYTHONPATH"} \
  --setenv=XLA_PYTHON_CLIENT_PREALLOCATE=false --setenv=XLA_PYTHON_CLIENT_MEM_FRACTION="$FRAC" \
  "$@"
