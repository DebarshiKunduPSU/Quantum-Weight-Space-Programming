#!/usr/bin/env bash
# Run every experiment for one model. Resumable: finished trials are skipped.
#   bash scripts/run_all.sh gpt-6-astra
#   bash scripts/run_all.sh fable
set -u
MODEL="$1"
PY="${PYTHON:-python}"
cd "$(dirname "$0")/.."

# QML, free design (architecture + weights)
bash scripts/run_qml_free.sh "$MODEL"

# Weights only: fixed QML ansatz, VQE, QAOA
"$PY" -u -m weightprog.qml_fixed --model "$MODEL" --trials 3
"$PY" -u -m weightprog.vqe       --model "$MODEL" --trials 3
"$PY" -u -m weightprog.qaoa      --model "$MODEL" --trials 3
