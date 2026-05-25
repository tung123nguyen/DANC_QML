# QNN IDS Thesis

Small-sample generalization study of Quantum Neural Networks vs Classical ML
for intrusion detection on CIC-IDS2017.

## Quick Start

### Local (development, smoke tests)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Smoke test (small data, few epochs) - run any config with --smoke-test
python scripts/run_single.py configs/classical/lr_s1_n100_f4_seed0.yaml --smoke-test
```

### Colab (training)

Open `notebooks/colab_runner.ipynb`, run all cells.

## Project Structure

```
src/
  data/         Data loading, cleaning, sampling, feature selection
  models/       Classical and QNN model factories
  training/     Training loops (classical + QNN)
  evaluation/   Metrics computation
  utils/        Seeding, I/O, logging
  experiment.py Main entry point (config -> result)

configs/        YAML config files (one per experiment)
scripts/        CLI scripts for batch runs
notebooks/      Analysis notebooks (NOT for training)
tests/          Unit tests for critical correctness
```

## Workflow

1. Edit code locally in `src/`
2. Generate configs: `python scripts/generate_configs.py`
3. Smoke test locally with one config
4. Push to GitHub
5. On Colab: pull, run `scripts/run_sweep.py`
6. Results save to Google Drive
7. Aggregate in `notebooks/02_aggregate.ipynb`

## Reproducibility

- Every experiment has a fixed `seed`
- Results folder contains `config.yaml` + `meta.json` (git commit, hardware)
- Idempotent: re-running a sweep skips completed experiments

## Sample semantics

In configs, `train_samples_per_class` is the number of samples per class
**in the training set**, and `test_samples_per_class` is the size **of the
test set** per class. Train and test are sampled disjointly from the source
data, so there is no overlap. The test size is held constant across all
experiments so test F1 numbers are directly comparable across train sizes.

Example: `train_samples_per_class: 100, test_samples_per_class: 500` means
the model trains on 200 rows total (100 per class) and is evaluated on 1000
rows total (500 per class).
