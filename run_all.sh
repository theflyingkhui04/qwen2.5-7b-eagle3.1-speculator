#!/usr/bin/env bash
# M3-04 delegates runtime execution to Modal Linux.
set -euo pipefail
step="${1:-plan}"
case "$step" in
  plan|dataset_check|prepare_hidden_states|smoke|train) ;;
  *) echo "Unsupported step: $step" >&2; exit 2 ;;
esac
exec modal run modal_pipeline.py --step "$step"
