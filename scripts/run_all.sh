#!/usr/bin/env bash
# Run every experiment for one model. Resumable: finished trials are skipped.
#   bash scripts/run_all.sh gpt-6-astra
#   bash scripts/run_all.sh fable
set -u
MODEL="$1"
PY="${PYTHON:-python}"
cd "$(dirname "$0")/.."

run() { echo; echo "### $*"; "$PY" -u -m weightprog.run --model "$MODEL" "$@"; }

# QML, free design (architecture + weights)
run --run-name default                                                        # 3 vs 8, 5 shots, images
run --run-name pair49      --set task.classes='[4,9]'
run --run-name all10       --set task.classes='[0,1,2,3,4,5,6,7,8,9]' --set prompt.shots_per_class=3
run --run-name shots1      --set prompt.shots_per_class=1
run --run-name shots10     --set prompt.shots_per_class=10
run --run-name text        --set prompt.modality=text
run --run-name structured  --config configs/structured.yaml
run --run-name structured_all10 --config configs/structured.yaml --set task.classes='[0,1,2,3,4,5,6,7,8,9]' --set prompt.shots_per_class=3

# Weights only: fixed QML ansatz, VQE, QAOA
"$PY" -u -m weightprog.qml_fixed --model "$MODEL" --trials 3
"$PY" -u -m weightprog.vqe       --model "$MODEL" --trials 3
"$PY" -u -m weightprog.qaoa      --model "$MODEL" --trials 3
