# Weight-Space Programming of Variational Quantum Circuits

Code, prompts, raw model replies and analysis for the paper

> **Weight-Space Programming of Variational Quantum Circuits: Large Language Models Write Quantum Circuit Parameters Without an Optimizer.**
> Debarshi Kundu, Swaroop Ghosh, Vasant Honavar. The Pennsylvania State University.

**Question.** [Weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang & Khattab, MIT CSAIL, Nov 2026) has a general-purpose LLM read and write a neural network's weights directly, with no optimizer. This project, inspired by theirs, asks the same question for quantum circuits: can a large language model write the angles of a parameterized quantum circuit directly — from a description of the problem, with no optimizer and no feedback — for quantum machine learning, VQE and QAOA?

**Protocol.** One request per trial: the model gets the problem (a Hamiltonian, a graph, or a few labelled digit images), the exact circuit with named angles (or a small circuit language in which to design one), and the gate conventions. It replies with a JSON list of numbers. The circuit is simulated exactly as written. No loss, gradient or accuracy is ever returned; only a syntactic validation error is sent back if the reply is malformed. Every trial is compared, on the same circuit, with random weights, a classical optimizer from random starts, and the optimizer started from the model's weights.

**Headline results (GPT-6 Astra, 139 requests, all valid on the first attempt).** VQE: within 0.007 of the optimizer on the normalized energy scale on all 11 problem–ansatz pairs, with the exact H₂ ground state derived in closed form. QAOA: optimal depth-1 angles on all six graphs; at depth 3 and 6 at the optimizer's level when the sign convention is respected (6/42 replies negate the mixing angles). Warm start: the optimizer started from the model's angles reaches its best value after a median of 1 evaluation instead of hundreds. QML: 92–95% on digit pairs with self-designed circuits, 65% on all ten digits, 93–96% from 12–18 hand-written weights of a fixed amplitude-encoding circuit — but accuracy falls to chance as the fixed circuit deepens to 30–54 weights while the optimizer improves.

The paper is in `paper/` (`paper/main.pdf`).

## Repository layout

```
weightprog/            the experiment code
  spec.py              circuit JSON language (free design): parsing, validation, weights
  sim.py               batched state-vector simulator (checked against Qiskit in tests/)
  hamiltonians.py      Pauli-sum Hamiltonians (TFIM, Heisenberg, H2, MaxCut), exact diagonalization
  ansatz.py            fixed circuit templates: HEA, HVA, QAOA, amplitude-encoding classifier
  data.py              digits data, few-shot sampling, image rendering
  prompts.py           free-design QML prompt;  vqe.py / qaoa.py / qml_fixed.py hold their own prompts
  providers.py         OpenAI (Responses API), Anthropic, OpenAI-compatible endpoints, offline mock
  harness.py           ask → parse → repair loop, resumable trial runner, optimizer baselines
  run.py               free-design QML runner;  vqe.py, qaoa.py, qml_fixed.py: weights-only runners
  evaluate.py          metrics and QML baselines
  analysis.py          builds results/summary.csv, paper/figures/*, paper/tables/*
configs/               default.yaml (all settings, model aliases), structured.yaml
scripts/               run_all.sh <model>, run_qml_free.sh <model>
results/               every trial: prompt.txt (+ images), response_k.txt, circuit.json, result.json
paper/                 LaTeX source, figures, generated tables
tests/                 pytest: simulator vs Qiskit, circuit-language validation
```

### Where the traces are

`results/<experiment>/<model>/<case>/trial_NNN/` for the weights-only experiments (`vqe`, `qaoa`, `qml_fixed`) and `results/<condition>/<model>/trial_NNN/` for free-design QML (`default` = 3 vs 8, `pair49`, `all10`, `shots1`, `shots10`, `text`, `structured`, `structured_all10`). Each directory holds the exact prompt the model saw (`prompt.txt`, with the images as PNG), every raw reply (`response_0.txt`, plus `response_1.txt`… for repair turns), the parsed circuit (`circuit.json`, free design only) and `result.json` with the model's weights, all scores, all baselines and token/latency accounting. `results/summary.csv` is the flat table used for the figures.

Results for Claude Fable 5.1 (`results/*/fable/`) are a partial run with identical prompts (QAOA complete, VQE and fixed-ansatz QML partial, three free-design conditions); they are released for completeness and are not analysed in the paper.

## Reproducing

```bash
conda create -n weightprog python=3.11 -y && conda activate weightprog
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY (and ANTHROPIC_API_KEY for Claude models)
python -m pytest tests          # simulator cross-check against Qiskit

bash scripts/run_all.sh gpt-6-astra      # every experiment; finished trials are skipped
python -m weightprog.analysis            # figures, tables, results/summary.csv
cd paper && latexmk -pdf main.tex
```

Individual experiments:

```bash
python -m weightprog.run --model gpt-6-astra --run-name default                 # free-design QML, 3 vs 8
python -m weightprog.run --model gpt-6-astra --config configs/structured.yaml   # with structural rules
python -m weightprog.vqe --model gpt-6-astra --trials 3
python -m weightprog.qaoa --model gpt-6-astra --trials 3 --cases 3reg0
python -m weightprog.qml_fixed --model gpt-6-astra --trials 3 --cases L8
python -m weightprog.run --model mock:test --dry-run     # inspect a prompt without calling any model
```

`--model` takes an alias from `configs/default.yaml` (`gpt-6-astra`, `fable`, `opus`, `sonnet`, `haiku`) or `provider:model_id`; any OpenAI-compatible endpoint can be added to the alias table. Model replies are stochastic: a rerun gives new samples, not the numbers in the paper; the released traces are the record of the reported runs.

## Citation

```bibtex
@article{kundu2026weightspace,
  title   = {Weight-Space Programming of Variational Quantum Circuits: Large Language Models Write Quantum Circuit Parameters Without an Optimizer},
  author  = {Kundu, Debarshi and Ghosh, Swaroop and Honavar, Vasant},
  year    = {2026},
  url     = {https://github.com/DebarshiKunduPSU/Quantum-Weight-Space-Programming}
}
```
