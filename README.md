# Octo

Octo implements the [v1 pointer-model design](docs/octo_v1_design.md): independently encoded text candidates, a shared decision representation, and one learned pointer head. It returns decisions and distributions without generating answer tokens.

The implementation supports Choice, Score, and Noul, hard and soft targets, equal record weighting, LoRA training, typed outputs, and prediction checkpoints. Tiny CPU tests validate the architecture; this repository does not yet contain a semantically trained or calibrated pretrained checkpoint.

## Install and check

```sh
uv sync
uv run python -m unittest discover -s tests -v
```

Tests cover every attention edge, actual forbidden attention weights, all six candidate permutations, candidate isolation, packed-question isolation, right-padded batches, output arithmetic, masked soft-target loss, frozen pretrained parameters, adapter/head updates, single-example memorization, and checkpoint reload. Both eager and SDPA paths are checked on tiny fp32 Qwen3 backbones. Production quality and hardware limits still require evaluation with a trained checkpoint.

## Run a tiny training demonstration

```sh
uv run octo-train --tiny --epochs 60 \
  --wandb online --project octo_v1 \
  --checkpoint artifacts/models/demo/checkpoint
```

This trains a pointer head over a frozen random Qwen3 backbone on three synthetic records, one per primitive. Its metrics establish training wiring and memorization, not held-out language understanding. Outputs include the model/tokenizer checkpoint and a local loss trace. Choose a fresh checkpoint destination for each run.

W&B defaults to disabled. Online logging reads `WANDB_API_KEY` from the environment or the ignored project `.env`; credentials are never part of checkpoint metadata or W&B configuration. `.env` should have owner-only permissions. Use `--wandb offline` to collect metrics locally. The default project is `octo_v1`. Training logs loss, gradient norms, elapsed step time, learning rates, per-primitive training evaluation, and development evaluation when supplied. Training publishes saved prediction checkpoints as model artifacts. Dataset records and provenance are published separately through the artifact publisher. Use an exact --dataset-artifact version to connect the training run and checkpoint to the registered dataset. Source code and credentials are excluded.

## Train the pretrained model

Prepare JSONL with one record per line, then run:

```sh
uv run octo-train --train data/train.jsonl --dev data/dev.jsonl \
  --device mps --epochs 5 --wandb online --project octo_v1 \
  --checkpoint artifacts/models/pilot/checkpoint
```

The default pretrained path is the pinned `Qwen/Qwen3-1.7B-Base` revision from the [local hardware report](docs/octo_v1_hardware_validation.md), fp32 SDPA, LoRA rank 16/alpha 32/dropout 0.05 on verified attention and MLP projections, and a 256-dimensional fp32 pointer. Base weights are frozen. Training uses AdamW with initial adapter LR `1e-4` and pointer LR `1e-3`. These settings are starting points, not measured quality optima. Model and tokenizer downloads may be required.

Separate train and development data by `source_group_id`; the CLI rejects overlapping groups. Keep calibration and locked-test records separate. The CLI saves the final epoch, so select training settings on development data before a final run. No calibration or production-quality claims are made by the demonstration.

Default request limits are one question, 2–8 Choice candidates, 2–10 Score criteria, 512 physical tokens, 512 logical positions, and a 64 MiB dense-mask budget. Requests are rejected rather than truncated. Larger limits and shared-state packing are available through explicit `Limits` settings for experiments; they have tiny-model correctness checks, not pretrained quality or capacity validation. Actual backbone capacity is checked too.

## Request format

```json
{
  "record_id": "route-001",
  "source_group_id": "export-family",
  "state": "PDF export is broken. CSV export is available as a workaround.",
  "questions": [{
    "id": "route",
    "type": "choice",
    "instruction": "Which team should handle this issue?",
    "options": [
      {"id": "billing", "description": "Payment and billing problems"},
      {"id": "engineering", "description": "Broken product functionality"}
    ],
    "label_id": "engineering"
  }]
}
```

Score uses `instruction` and `criteria`, an ordered list of complete descriptive strings. Its stable IDs and values are the original indices (`"0"`, `"1"`, …). Noul uses `statement` and the immutable IDs `true`/`false`.

Supply one supervision field: `label_id`, `target` (probabilities keyed by every candidate ID), Noul `truth_target` in `[0,1]`, or Score `score_target` together with `score_interpolation: "adjacent"`. Omit supervision for prediction. Fractional Score interpolation is an explicit labeling assumption. Binary Noul annotations need a task-specific policy for missing evidence.

Candidate and question IDs, labels, and record metadata stay outside model text. Duplicate IDs, invalid targets, reserved-token collisions, and unsupported budgets are errors. Five existing reserved tokenizer tokens are verified and recorded in checkpoints; no new frozen random embeddings are silently added.

## Use the model in Python

```python
from octo import load_checkpoint

model, encoder, metadata = load_checkpoint("artifacts/models/pilot/checkpoint", device="mps")
batch = encoder.batch([request]).to("mps")
answers = model.predict(batch)  # list of answer lists, one per input record
```

Choice returns `selected_id`, probabilities by ID, and entropy concentration confidence. Exact ties select the lexicographically smallest ID. Score returns its expected level, full distribution, and concentration confidence. Noul returns `probability_true` and its distribution; applications choose Boolean thresholds.

`save_checkpoint` saves the backbone (or adapters with pinned base identity), pointer weights, tokenizer, structural-token map, limits, normalization version, precision/backend, and supplied experiment metadata. Adapter reload requires the exact base weights, cached locally by default. This is a prediction artifact; optimizer state and random/data-order state for exact training resumption are not included.

The reusable implementation lives in `src/octo/`; existing exploratory scripts and hardware reports remain available.

## Public training dataset and artifact management

The [BANKING77 Choice pilot](docs/octo_dataset.md) is built under `artifacts/datasets/banking77-choice-v1/` with pinned source snapshots, checksums, labeling rules, and separate diagnostic/train/dev/calibration/locked-test partitions. See the guide for reproducible rebuilding, exact W&B dataset references, model artifact publication, and the Docker MLflow fallback.

The expanded [BFSI and chess corpus](docs/octo_bfsi_chess_dataset.md) contains all BANKING77 intents plus retail banking, insurance, wealth management, and Lichess tactical chess positions. BFSI and chess use separate configuration files and folders. Build each with `uv run python -m octo.mixture --domain bfsi --fetch` or `--domain chess --fetch`. Use `octo.chess_task.make_chess_request` to turn a FEN, side to move, and caller-provided legal moves into a Choice request.

The [post-training benchmarking guide](docs/octo_benchmarks.md) provides separate frozen BFSI/chess suites and adapters for the pinned prompted Qwen3 baseline, Jev, and trained Octo. Benchmark answers are separated from inference inputs.
