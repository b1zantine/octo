# Post-training benchmarks: prompted Qwen3, Jev, and Octo

BFSI and chess have separate source configurations, dataset folders, benchmark configurations, and evaluation bundles. The original combined W&B artifact remains historical provenance; its split assignments are preserved in the separated corpora. These benchmarks are prepared now; run comparisons only after training and freezing checkpoint/prompt settings.

## Fair inputs and frozen answers

Each model receives the same state, instruction, option descriptions, and option order. Candidate identifiers are neutral letters A–H for all three adapters so semantic IDs cannot leak intent labels to prompted Qwen or Jev. Letter mappings are deterministic and are not model evidence about the correct answer. No source metadata, ratings, themes, source labels, or gold answers enter inference requests. Answers and source provenance are in separate `.gold.jsonl` files used only by the scorer.

Each domain has a small development suite for prompt/integration work, a frozen core benchmark, and the complete locked test. The core is a subset of the full suite: do not treat them as independent evaluations or sum their sample sizes. Choose at most one record per family per source domain for core/dev, visiting each BFSI intent before repeating one and rotating candidate-count bands within each intent and chess side/rating/candidate-count strata. Full retains all locked records. No locked source group overlaps train/dev/calibration/diagnostic. All future permutations inherit the original family and partition.

BFSI core selects 200 examples per source (800 total); chess core selects 200 examples. Development has 40 per source (160 BFSI, 40 chess). Full uses all locked test examples. Core lets you compare all three models with 1,000 Jev calls rather than immediately sending every test example. Run all three on the same suite. Full is optional for the later final report.

Public sources may overlap pretrained or hosted-model training corpora. We can enforce local source-family isolation, not prove that Qwen or Jev never saw this public data. Synthetic BFSI template families are approximate. Chess measures selecting Lichess tactical solutions from legal options, not Elo or full-game strength. The source solution is always supplied as a candidate; other subsets would need engine rankings.

## Three adapters

- **Qwen prompt:** the exact unfine-tuned `Qwen/Qwen3-1.7B-Base` revision used by Octo, `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. Plain zero-shot completion prompting, greedy decoding, eight new tokens maximum, one option letter expected. No instruction-tuned checkpoint is silently substituted and no gold examples are in the prompt. This Base checkpoint may follow instructions less reliably than an Instruct checkpoint; malformed generations count as errors.
- **Jev:** official TypeSafe endpoint `https://api.typesafe.ai/v1/systemone`, pinned `jev-1.13.0`. Read `JEV_API_KEY` from the environment or ignored `.env`. Use a Choice request with `instructions` and letter-keyed `criteria`. Record the actual response model and token usage. Redirects are disabled; credentials are never written to outputs or tracking metadata. See [official API reference](https://docs.typesafe.ai/api) and [Choice documentation](https://docs.typesafe.ai/primitives/choice).
- **Octo:** load the trained checkpoint, use evaluation/inference mode, and score the exact same label-free requests. Record checkpoint file hashes and saved dataset lineage. Validate that its saved training-file checksum corresponds to this corpus or the historical parent corpus. No optimizer step occurs during evaluation.

Freeze prompts, decoding, requested Jev version, checkpoint, and benchmark version before core/full evaluation. Use dev for changes. No live Jev benchmark is run during preparation, so no paid evaluation calls are made now.

## Results and comparison

Accuracy includes every input: API failures, timeouts, and malformed/invalid choices count as wrong. Report response validity and error rate separately; do not drop hard examples or silently shrink the denominator. Compare micro accuracy, macro accuracy across BFSI sources, per-domain accuracy, candidate-count bands, and latency median/p95. Chess also reports side and puzzle-rating bands. The random baseline is the mean of 1/K, computed from each suite's actual candidate counts.

Jev and Octo can additionally report NLL and multiclass Brier score when their full distributions are valid. A greedy prompted answer does not supply a probability distribution, so its NLL is unavailable rather than invented from self-reported confidence. Report probability coverage and probability metrics separately from the primary three-model accuracy comparison.

Compare paired predictions on identical record IDs, reporting wins/losses/ties and accuracy differences. Bootstrap source families, not independently augmented rows, for accuracy and paired-difference confidence intervals. Errors remain wrong in these calculations. Latency is per-example wall-clock inference after model loading; local device time and hosted network time are operational measures, not a pure architecture comparison. Record hardware/device and provider token usage. Dollar cost is not fabricated from missing billing data.

Prediction files contain benchmark/suite/input checksums, model identity, checkpoint hashes where applicable, per-record results, and errors. Resume only a matching run identity. Comparisons require complete prediction files from all three adapters with matching input checksums. Upload benchmark bundles and final evaluation results as separate W&B artifact types, with dataset/checkpoint lineage; retain the MLflow fallback and all local files under artifacts/.

## Commands

Build separate training corpora from the existing historical snapshot, without changing partitions:

```sh
uv run python -m octo.partition_datasets
```

For a fresh source build, use one configuration per corpus:

```sh
uv run python -m octo.mixture --domain bfsi --fetch
uv run python -m octo.mixture --domain chess --fetch
```

Existing output directories are never overwritten. Use a new --output directory for rebuilds.

Prepare the independent benchmarks:

```sh
uv run python -m octo.benchmark_data --domain bfsi
uv run python -m octo.benchmark_data --domain chess
```

After training and freezing settings, run each adapter on a benchmark. Repeat for chess-v1:

```sh
uv run python -m octo.benchmark run --benchmark artifacts/benchmarks/bfsi-v1 --suite core \
  --adapter qwen-prompt --device mps --output artifacts/evaluations/bfsi-qwen-core
uv run python -m octo.benchmark run --benchmark artifacts/benchmarks/bfsi-v1 --suite core \
  --adapter jev --output artifacts/evaluations/bfsi-jev-core
uv run python -m octo.benchmark run --benchmark artifacts/benchmarks/bfsi-v1 --suite core \
  --adapter octo --device mps --checkpoint artifacts/models/bfsi-chess-001 \
  --output artifacts/evaluations/bfsi-octo-core
uv run python -m octo.benchmark compare --benchmark artifacts/benchmarks/bfsi-v1 --suite core \
  --predictions artifacts/evaluations/bfsi-qwen-core artifacts/evaluations/bfsi-jev-core artifacts/evaluations/bfsi-octo-core \
  --output artifacts/evaluations/bfsi-comparison-core
```

Use --resume for an interrupted matching run. No prompt repair is performed on test answers. Jev retries only transient API errors with bounded backoff; authentication/configuration failures stop rather than repeatedly spending calls. Configured retries and usage are logged.

Publish prepared bundles with dataset lineage:

```sh
uv run --extra tracking-fallback python -m octo.tracking artifacts/benchmarks/bfsi-v1 --kind benchmark --name bfsi-benchmark-v1 --dataset-artifact octo-team/octo_v1/bfsi-v2:v0
uv run --extra tracking-fallback python -m octo.tracking artifacts/benchmarks/chess-v1 --kind benchmark --name chess-benchmark-v1 --dataset-artifact octo-team/octo_v1/chess-v2:v0
```

After comparison, publish its directory with --kind evaluation, --benchmark-artifact, --dataset-artifact, and optionally --model-artifact using the exact versions. The comparison includes all three prediction files.
