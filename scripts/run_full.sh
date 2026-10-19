#!/usr/bin/env bash
# Runs the published `full` tier stage by stage. Each stage is resumable (finished jobs are skipped).
set -u
cd "$(dirname "$0")/.."
A=${ADAPTARENA:-adaptarena}
for stage in "tune" "parity" "headline" "ablations" "memory" "stress" "redteam" "probes"; do
  echo "=== $stage $(date -u +%FT%TZ)"
  $A $stage --tier full --workers 4 2>&1 || $A $stage --tier full 2>&1 || { echo "stage $stage failed"; exit 1; }
done
echo "=== done $(date -u +%FT%TZ)"
