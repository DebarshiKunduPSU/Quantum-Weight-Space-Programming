# LLM weight programming for PQC classifiers

**Hypothesis.** A vision-language model can look at a few labelled images and write a complete
parameterized quantum circuit classifier — data encoding, gate layout, every numeric angle, and the
readout — that works with no optimizer.

Each trial shows the model a few digits per class (scikit-learn `digits`, 8x8), asks for a circuit
in a small JSON format, simulates that circuit exactly as written, and scores it on a held-out test
split. Baselines on the same architecture show how much of the accuracy comes from the weights.

## Setup

```bash
conda activate weightprog            # already created: Python 3.11 with requirements.txt installed
cp .env.example .env                 # then add OPENAI_API_KEY and/or ANTHROPIC_API_KEY
```

## Run

```bash
python -m weightprog.run --dry-run                    # write the prompt + images for inspection, no API call
python -m weightprog.run                              # gpt-6-astra, 3 vs 8, 5 shots/class, 5 trials
python -m weightprog.run --model fable                # Claude Fable 5.1
python -m weightprog.run --model anthropic:claude-opus-5-5 --trials 10
python -m weightprog.run --model mock:test            # offline pipeline check
python -m weightprog.report                           # summary table over everything in results/
```

Any setting in `configs/default.yaml` can be overridden with `--set`, or with a second YAML via `--config`:

```bash
python -m weightprog.run --run-name tenclass --set task.classes='[0,1,2,3,4,5,6,7,8,9]' --set prompt.shots_per_class=3
python -m weightprog.run --run-name textonly --set prompt.modality=text
python -m weightprog.run --run-name linear   --set circuit.max_features_per_angle=null
```

Use a different `--run-name` for each condition. Runs resume: finished trials are skipped, failed
API calls are retried.

**Models.** `--model` takes an alias from the `models` table in the config (`gpt-6-astra`, `fable`,
`opus`, `sonnet`, `haiku`) or `provider:model_id` with provider `openai`, `anthropic`,
`openai_chat` (any OpenAI-compatible server; set `base_url` and `api_key_env` in the table), or
`mock`. Per-model options such as `effort` or `max_output_tokens` go in the table entry.

## What the model writes

```json
{
  "reasoning": "...",
  "n_qubits": 2,
  "state_prep": {"type": "zero"},
  "gates": [
    {"gate": "ry", "qubits": [0], "params": ["pi * x[27]"]},
    {"gate": "cx", "qubits": [0, 1]},
    {"gate": "rz", "qubits": [1], "params": [0.7]}
  ],
  "readout": {"type": "bitstrings", "qubits": [1], "classes": {"3": ["0"], "8": ["1"]}}
}
```

- `x[i]` is pixel `i = 8*row + col`, scaled to [0, 1]. Every angle is affine in the features,
  `c + w*x[i]`; the constants and coefficients are the weights.
- State prep is `|0…0>` or amplitude encoding of the 64 pixels on six qubits (three row qubits,
  three column qubits).
- Readout maps bitstrings to classes, or uses one qubit per class. Prediction = highest score.
- The full instructions the model receives are in `weightprog/prompts.py`; `--dry-run` writes the
  rendered prompt to `results/<run>/<model>/trial_000/prompt.txt`.

If the reply is not a valid circuit, the validation error is sent back for up to
`llm.max_repair_attempts` more turns. The model never receives accuracy feedback.

## What is measured

Per trial, in `results/<run>/<model>/trial_NNN/result.json` (next to the prompt, images, raw
replies and `circuit.json`):

| Quantity | Meaning |
|---|---|
| `llm.test` | Accuracy, balanced accuracy and confusion matrix of the circuit as written |
| `random_weights` | Same architecture, weights redrawn uniformly from [-π, π] |
| `trained_on_support` | Same architecture, optimizer fits the weights on the examples the model saw |
| `llm_finetuned_on_support` | Optimizer started from the model's weights |
| `trained_on_full` | Optimizer on the train split (off by default; slow) |
| `logreg_on_support`, `logreg_on_full` | Logistic regression on raw pixels |

The hypothesis predicts `llm.test` well above `random_weights` and close to `trained_on_support`.

## Design choices that affect interpretation

- **Same data for every model.** The train/test split depends only on `task.split_seed`; the support
  set for trial *k* depends only on `(seed, k)`. Models are compared on identical prompts.
- **One pixel per angle by default** (`circuit.max_features_per_angle: 1`). With `null`, an angle may
  be any linear combination of pixels, so a single `ry` can act as a classical linear classifier and
  the "quantum" part does little. Run that as a separate condition.
- **0 vs 1 is too easy to be informative.** A single rotation on one central pixel gets about 93%
  on it, so the default task is 3 vs 8 (the same one-pixel circuit gets about 64%).
- **The digits dataset is public**, so models bring prior knowledge of what digits look like. The
  `text` modality and the shot count are the handles for probing how much the shown examples matter.
- **The optimizer baseline is modest**: L-BFGS-B with finite-difference gradients, one random start,
  capped at `train_max_iter` iterations and `train_max_seconds`. It is a reference point, not a
  tuned upper bound.

## Layout

```
configs/default.yaml   all settings, with comments
weightprog/spec.py     circuit JSON format: parsing, validation, weights
weightprog/sim.py      batched statevector simulator (checked against Qiskit in tests)
weightprog/data.py     split, few-shot sampling, image rendering
weightprog/prompts.py  system prompt and task message
weightprog/providers.py OpenAI / Anthropic / OpenAI-compatible / mock
weightprog/evaluate.py metrics and baselines
weightprog/run.py      experiment runner
weightprog/report.py   summary table / CSV
tests/                 python -m pytest tests
```
