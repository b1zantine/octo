# BANKING77 Choice pilot

This is the Choice-first dataset for implementation-plan sessions 3–8. It is a teaching pilot, not a full BANKING77 benchmark. Score and Noul need independently justified annotations before session 9.

## Labeling policy

Preserve BANKING77 utterances and original intent annotations. Select only `activate_my_card`, `change_pin`, and `lost_or_stolen_card`. Require explicit task words in the utterance; exclude unclear utterances rather than invent labels. Candidate descriptions explain the support issue, not a bank's actual policy or a recommended remedy. Do not treat source metadata as model evidence.

Activation requires `card` and an activation word. PIN changes require `pin` and a change/reset/new/update word. Lost/stolen requires `card` and lost/stolen/missing. These are deterministic evidence filters, not a claim that every upstream annotation has been independently relabeled. The diagnostic examples were individually inspected; the other partitions retain upstream supervision.

Each example has one Choice question with two or three candidates, always including its annotated correct intent. Randomly select the distractor and shuffle the candidates using seed 17. IDs remain stable. No paraphrase augmentation or severity/truth labels are generated.

## Splits and leakage controls

| Partition | Records | Per intent | Purpose |
|---|---:|---:|---|
| diagnostic | 12 | 4 | Memorization/wiring only |
| train | 60 | 20 | Fit weights |
| dev | 18 | 6 | Choose settings |
| calibration | 12 | 4 | Reserved; too small for strong calibration claims |
| test_locked | 12 | 4 | Final evaluation after settings are frozen |

Normalize Unicode/case/punctuation for grouping only; preserve original state text. Build connected components across both original source splits using exact normalized matches, token-set Jaccard >= 0.80, or sequence similarity >= 0.92. Exclude components with conflicting labels. Keep one representative per selected component; source_groups.json records every member. Any component containing upstream test data is eligible only for locked test. All other partitions come from upstream train. All future permutations/questions/paraphrases from a source group must inherit its partition.

These lexical components approximate source families: BANKING77 provides no document or scenario IDs. Semantic leakage cannot be ruled out. The small sample and easy candidate subset cannot establish general 77-intent performance. Compare with uniform probabilities (expected accuracy 5/12 with equal two-/three-option counts) and a fixed global majority-intent baseline (accuracy 1/3 on these balanced labels).

## Provenance and storage

The canonical source is [PolyAI/BANKING77](https://huggingface.co/datasets/PolyAI/banking77), by Casanueva et al. (2020), [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Data was obtained from Hugging Face's pinned Parquet conversion in `legacy-datasets/banking77`, revision `54c053c5487708560dc60df4670185a2a4b27cd3`. This mirror is deprecated; keep the local source snapshot for reproducibility. The pinned canonical card revision and SHA-256 source/tokenizer checksums are in configs/banking77_pilot.json.

Every record preserves source split, zero-based row index, original numeric and named labels, original text checksum, source revisions, transformation version, group, and partition. The manifest includes split counts, file checksums, source checksums, tokenizer identity, and measured maximum token length. Raw Parquet, source card, license/attribution, exclusions, configuration, and group membership are included in the W&B dataset artifact. Source files, datasets, models, and tracking stores live in ignored `artifacts/`.

Rebuild into a new directory (existing versions are never overwritten):

```sh
uv run python -m octo.dataset --fetch --output artifacts/datasets/banking77-choice-v1-rebuild
uv run --extra tracking-fallback python -m octo.tracking artifacts/datasets/banking77-choice-v1-rebuild
```

## W&B and local MLflow

W&B artifact versions are immutable; use an exact `:vN` reference for training. Uploads have a conservative per-artifact budget of 256 MiB. This is a local upload guard, not an account-wide quota meter. The currently published free-plan storage allowance is 5 GB/month; successful upload verifies availability for this version, not remaining account storage. See [current pricing](https://coreweave.com/forge-pricing).

Train with lineage and automatic checkpoint publication:

```sh
uv run --extra tracking-fallback octo-train \
  --train artifacts/datasets/banking77-choice-v1/train.jsonl \
  --dev artifacts/datasets/banking77-choice-v1/dev.jsonl \
  --dataset-artifact octo-team/octo_v1/banking77-choice-v1:v1 \
  --wandb online --device mps --epochs 1 \
  --checkpoint artifacts/models/banking77-pilot-001
```

The trainer consumes the dataset artifact and verifies the selected file checksums before updating weights. It records dataset identity and hashes in the checkpoint and publishes adapters, pointer weights, tokenizer, settings, and training metrics as a `model` artifact. The pinned base model is referenced rather than reuploaded when using LoRA. A dataset upload is not a trained model; no task-quality model has been trained as part of dataset creation.

For existing checkpoints under artifacts/:

```sh
uv run --extra tracking-fallback python -m octo.tracking artifacts/models/banking77-pilot-001 \
  --kind model --dataset-artifact octo-team/octo_v1/banking77-choice-v1:v1
```

If W&B publication is unavailable or rejects the upload, the publisher starts the local MLflow service using compose.mlflow.yaml. The entire repo artifacts/ directory is mounted at /artifacts, including the SQLite tracking database and uploaded artifacts. The server binds to localhost:5000. Docker must be running. Install the fallback extra when using auto publication. To explicitly select local storage:

```sh
uv run --extra tracking-fallback python -m octo.tracking artifacts/datasets/banking77-choice-v1 --backend mlflow
```

No MLflow container is started while W&B publication succeeds. Receipts live under artifacts/tracking/receipts/. An interrupted upload or unavailable Docker service leaves the local source/dataset/checkpoint intact. Existing metric-only W&B runs remain supported; use --dataset-artifact for managed training lineage. W&B run initialization errors stop training before updates; standalone artifact publication can fall back to MLflow.
