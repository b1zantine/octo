# BFSI and chess dataset v2

This dataset expands the Choice pilot into a multi-domain, candidate-conditioned training corpus. It includes all BANKING77 intents, retail banking, insurance, wealth-management support requests, and tactical chess decisions. It supplies training data; model quality must be measured after training. Score and Noul are not fabricated from intent or puzzle labels.

## Sources and provenance

| Domain | Hugging Face source | Supervision | License |
|---|---|---|---|
| Banking77 | canonical PolyAI/banking77; pinned legacy-datasets/banking77 Parquet conversion | Human-labeled banking intents, 77 categories | CC BY 4.0 |
| Retail banking | bitext/Bitext-retail-banking-llm-chatbot-training-dataset | Hybrid synthetic intent labels, 26 categories | CDLA Sharing 1.0 |
| Insurance | bitext/Bitext-insurance-llm-chatbot-training-dataset | Hybrid synthetic intent labels, 39 categories | CDLA Sharing 1.0 |
| Wealth management | bitext/Bitext-wealth-management-llm-chatbot-training-dataset | Hybrid synthetic intent labels, 24 categories | CDLA Sharing 1.0 |
| Chess | Lichess/chess-puzzles | Lichess Stockfish-derived tactical solutions | CC0 1.0 |

Exact source revisions, URLs, byte sizes, and SHA-256 checksums are in configs/bfsi_v2.json and configs/chess_v2.json and the dataset manifest. Original complete files live under artifacts/sources/. Compact original-row snapshots, per-source cards and license notices accompany the published dataset. Every example carries source, original file and zero-based row index, source revision, original label or puzzle solution, group ID, and transformation version. Each source retains its own license: the mixture does not relicense source records. Original Bitext response text is not used as model evidence or supervision; it is retained in the complete local source files.

The synthetic BFSI data supplements human-labeled BANKING77. It does not establish that generated financial advice or policies are accurate. This task predicts the support intent from the request and supplied category descriptions.

## BFSI construction

Use original customer utterances and intent labels, with readable descriptions derived from intent names. Sample candidate sets of 2, 3, 4, 6, or 8 entries from that source's taxonomy, always including the labeled answer, then shuffle deterministically. IDs are namespaced by source and never inserted into model text. This measures choice among supplied categories, rather than a full 77-class BANKING77 classification benchmark.

Group normalized exact matches and lexical near-duplicates across all four BFSI sources before splitting. Replace numeric values with a common marker for grouping only. Use 64-permutation MinHash LSH to propose pairs and verify token-set Jaccard >= 0.85 before merging connected components. Components with conflicting intent labels within a source are excluded. Approximate matching can miss some neighbors; lexical components are a proxy for scenario/template families because no original template IDs are supplied. Semantic paraphrase leakage remains possible.

## Chess construction and caller input

Use a pinned first shard of the official Lichess dataset, with a bounded scan, to select 10,000 puzzles with popularity >= 80, at least 1,000 plays, and rating deviation <= 100. These are tactical positions and do not represent all openings, quiet positional play, or general game strategy.

Lichess stores its FEN **before the opponent's setup move**. Validate the complete published continuation, apply its first move, and present the resulting FEN, board grid, and side to move. The target is the second published UCI move: the solver's first move. Sample legal distractors and shuffle the candidates. For mate-in-one puzzles, exclude alternative checkmating moves from distractors rather than labeling valid mates incorrect. Reject invalid boards, illegal continuations, and positions without a distinguishable distractor.

Each candidate description includes UCI, SAN, piece, source square, and destination square. Model evidence contains the board and side to move, never puzzle IDs, solution lines, themes, ratings, or source labels. Related puzzles from the same game and equivalent board positions are merged into the same split family.

Caller-supplied choices can use UCI or SAN. A helper validates the FEN, playing side, candidate legality, duplicates, and the supported 2–8 move limit:

```python
from octo.chess_task import make_chess_request
request = make_chess_request(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "white", ["e2e4", "d2d4", "g1f3"])
# After training and loading a checkpoint:
# answers = model.predict(encoder.batch([request]).to(device))
```

The generic Octo pointer selects a supplied candidate. This dataset always includes the puzzle solution. If a caller omits the objectively best move, these puzzle labels do not supply an engine ranking of the remaining options. Strong general chess play would require broader engine-ranked candidate sets and independent evaluation; no strength claim is made from dataset creation.

## Partitions and verification

Split source families before any candidate sampling variants are used for evaluation. All future questions and permutations inherit the same source group. Keep a small diagnostic set separate: one representative per reserved domain/group; its siblings are quarantined from task-quality partitions. Assign other families deterministically with intended train/dev/calibration/locked-test proportions 75/10/5/10. Original BANKING77 test families remain locked test, including related upstream training rows. Hash grouping produces approximate proportions, not exact counts. The manifest reports actual domain, candidate-count, label, and source-group counts.

Every record is checked by Octo normalization and the pinned Qwen tokenizer under the 512-token physical/logical budgets. Over-budget records are excluded with a recorded reason; nothing is silently truncated. Keep calibration and locked test separate from training/settings selection. Report domain-specific accuracy and NLL, chess results stratified by side/rating, and candidate-count bands. Uniform-choice accuracy is the mean of 1/K for each partition; use the manifest counts for the actual baseline.

## Build, publish, and train

```sh
uv run python -m octo.mixture --domain bfsi --fetch
uv run python -m octo.mixture --domain chess --fetch
uv run --extra tracking-fallback python -m octo.tracking artifacts/datasets/bfsi-v2
uv run --extra tracking-fallback python -m octo.tracking artifacts/datasets/chess-v2
```

Rebuilds must use a new output directory. Source downloads are checksum-verified; the builder records its own implementation hash. W&B receipts include the exact artifact version. Use that version for training:

```sh
uv run --extra tracking-fallback octo-train \
  --train artifacts/datasets/bfsi-v2/train.jsonl \
  --train artifacts/datasets/chess-v2/train.jsonl \
  --dev artifacts/datasets/bfsi-v2/dev.jsonl \
  --dev artifacts/datasets/chess-v2/dev.jsonl \
  --dataset-artifact octo-team/octo_v1/bfsi-v2:v0 \
  --dataset-artifact octo-team/octo_v1/chess-v2:v0 \
  --wandb online --device mps --epochs 1 \
  --checkpoint artifacts/models/bfsi-chess-001
```

This is a much larger training run than the teaching pilot. The trainer validates records, creates attention masks one example at a time, records the exact dataset version, and publishes the complete prediction checkpoint. It does not preload all dense attention masks onto the GPU. Task-quality results require running and evaluating this training separately.

W&B remains the primary artifact store. Uploads have a 256 MiB per-artifact guard to avoid accidental multi-gigabyte base-model uploads. Full local raw source files are retained outside the published training bundle, with immutable Hugging Face references and checksums inside it. The currently published free allowance is 5 GB/month; the upload guard is not a meter for remaining account capacity. If W&B rejects publication or the artifact exceeds the guard, automatic publication can start the Docker MLflow fallback from compose.mlflow.yaml. Everything remains under the repo artifacts/ folder, mounted at /artifacts. No local container is needed when W&B accepts the artifact. See docs/octo_dataset.md for the fallback commands.

BFSI and chess now have separate configurations and corpus folders. Repeating --train/--dev lets one model learn both while their data remains in separate files. The original combined artifact is retained as historical provenance. See [post-training benchmarks](octo_benchmarks.md) for prompted Qwen, Jev, and Octo comparisons.
