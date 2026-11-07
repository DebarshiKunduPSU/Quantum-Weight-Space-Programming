#!/usr/bin/env bash
# Free-design QML conditions for one model (resumable). Usage: bash scripts/run_qml_free.sh <model>
set -u
MODEL="$1"; PY="${PYTHON:-python}"
cd "$(dirname "$0")/.."
run() { echo; echo "### $*"; "$PY" -u -m weightprog.run --model "$MODEL" "$@" < /dev/null; }
run --run-name default
run --run-name pair49 --set task.classes='[4,9]'
run --run-name all10 --set task.classes='[0,1,2,3,4,5,6,7,8,9]' --set prompt.shots_per_class=3
run --run-name shots1 --set prompt.shots_per_class=1
run --run-name shots10 --set prompt.shots_per_class=10
run --run-name text --set prompt.modality=text
run --run-name structured --config configs/structured.yaml
run --run-name structured_all10 --config configs/structured.yaml --set task.classes='[0,1,2,3,4,5,6,7,8,9]' --set prompt.shots_per_class=3
