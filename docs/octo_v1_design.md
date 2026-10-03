# Octo v1 Model Design

Status: proposed architecture; implementation and measured results pending.

Reference: [Option-Isolated Pointer Decision Model Architecture](Option-Isolated%20Pointer%20Decision%20Model%20Architecture.md).

Coding sequence and validation gates: [Octo v1 Implementation Plan](octo_v1_implementation_plan.md).

## 1. Purpose and scope

Octo is a text-based decision model. Given a state and typed questions, it returns decisions and probability distributions without generating answer tokens or explanations. Candidate descriptions are supplied at request time, so the output space is dynamic.

This document adopts the pointer architecture in the reference: independently encoded candidate branches, shared decision aggregation, and a learned pointer head.

The first working milestone evaluates one question per state, with small Choice sets. All three primitives are required. Shared-state question packing and the 255-choice limit are later milestones within the design, enabled only after validation.

Retrieval, tool execution, policy enforcement, image inputs, rationale generation, and extended deliberative reasoning are outside scope. Supplied state provides the evidence; application code owns thresholds and actions.

## 2. Goals and acceptance requirements

- Encode every candidate independently of other candidates.
- Preserve candidate representations and mapped probabilities when physical branch order changes, within measured numerical tolerance.
- Support Choice, Score, and Noul through one learned scoring head.
- Produce exactly one logit per valid candidate and normalize only across that question's candidates.
- Keep separate questions isolated even when their state is shared.
- Learn semantic matching rather than fixed output slots or vocabulary labels.
- Reject requests exceeding supported token or memory budgets without silently removing evidence or candidates.
- Measure task quality, probability calibration, latency, and memory separately; no numerical quality or performance claim is established by this design.

## 3. Question and answer contracts

State is a text string in v1. Each question has a unique request-local ID. IDs are metadata and are not embedded as content. Candidate identities remain attached to descriptions through any internal permutation.

| Primitive | Input | Candidates | Answer |
| --- | --- | --- | --- |
| Choice | Instruction and candidate descriptions | 2–255; initially validate 2–8 | Selected candidate ID, probabilities by ID, confidence |
| Score | Rating instruction and ordered descriptive criteria | 2–10 | Expected level value, probabilities by level ID, confidence |
| Noul | Statement about the state | Exactly True and False after normalization | Probability of True, from 0 to 1 |

The limits describe the intended interface. A release advertises only the subset validated with its checkpoint, context budget, and hardware.

Choice candidate IDs are caller supplied, unique strings. Score assigns immutable level IDs and values from the original criteria order: 0 through $K-1$. Arbitrary numeric scales are deferred; application code may map the returned score to another scale. Noul uses immutable IDs `true` and `false`.

Choice ties select the lexicographically smallest tied candidate ID, so tie handling does not depend on physical order. The distribution still exposes the tie. Identical candidate descriptions with different IDs are allowed but cannot be semantically distinguished by this architecture; users should expect tied probabilities within numerical tolerance.

A missing-evidence choice may be supplied when the task and labels support it. It is not inserted automatically. Noul remains binary: its annotation policy must explain how insufficient evidence is handled, or such cases must be excluded from supervised examples. Absence of evidence is not automatically evidence of falsity.

## 4. Shared representation

Normalize all questions into an instruction and independent candidate descriptions:

- Choice uses the original instruction and options.
- Score uses its instruction and each criterion's complete description. Numeric values remain metadata.
- Noul uses a fixed versioned instruction template, `Is the following statement true of the state?`, followed by the statement, with candidates `True` and `False`.

Score criteria must be independently meaningful, measure one dimension, and describe distinct levels. References to neighboring levels violate the intended branch semantics.

The structural sequence is:

```text
<state> state_text
<question> instruction
<option> candidate_1 </option>
...
<option> candidate_K </option>
<decide>
```

The five delimiters are dedicated tokenizer IDs. Reusing existing reserved tokens is preferred when verified for the selected tokenizer. The strings in the reference are examples, not guaranteed token assignments. If new tokens are necessary, their embeddings must be trained and saved explicitly; this is a compatibility branch requiring validation.

User text must never introduce reserved structural IDs. Structural tokens are appended by the encoder, independently of text tokenization. Reserved-token collisions in content cause a validation error.

## 5. Backbone and learned parameters

Use a pretrained base decoder transformer with conventional attention in every layer. Here, “attention transformer” includes its normal feed-forward blocks; it does not mean removing the MLPs. Layers must respect the supplied arbitrary attention topology and explicit repeated position IDs.

The reference proposes Qwen3 base models around 1.7B or 4B parameters. The exact available checkpoint, revision, dimensions, tokenizer, and attention behavior must be verified before selection. Do not hard-code the reference's example dimensions or assume its model names identify available checkpoints. Hidden dimension $d$ comes from the selected configuration.

Only the transformer backbone participates. The vocabulary projection is absent from the forward path. Input embeddings remain necessary. A causal language-model wrapper may be used temporarily to recover its backbone, but vocabulary logits are never the decision scores.

The trained artifact contains frozen pretrained weights, LoRA updates, and a randomly initialized pointer head. Initial LoRA settings are rank 16, alpha 32, dropout 0.05. Candidate adapter targets are attention projections and feed-forward projections, verified against actual module names. A smaller checkpoint or compatible hardware may be selected for prototyping without changing the topology.

Backbone precision is bf16 where supported, with fp32 pointer projections, scores, softmax, and loss for the baseline. A fp32 backbone is suitable for tiny correctness tests. Quantization and alternate runtimes follow a verified baseline.

## 6. Attention topology

Let $s_i$ denote the question segment, with state segment 0. Let $r_i$ be the role: state, instruction, candidate, decision, or padding. Candidate tokens additionally carry a question-local candidate index $c_i$.

For real tokens, a key is visible only when it is physically causal ($j\le i$) and permitted below:

| Query role | Allowed key roles and ownership |
| --- | --- |
| State | Earlier/current state tokens only |
| Instruction | State plus earlier/current instruction tokens of the same question |
| Candidate | State, same-question instruction, and earlier/current tokens of that same candidate |
| Decision | State and all earlier/current tokens in its own question, including itself |

Every candidate's opening and closing delimiters have that candidate's role and ownership. The question delimiter has instruction role; the state delimiter has state role. Decision tokens cannot become visible to another question or candidate.

The additive mask is 0 for permitted edges and negative infinity for forbidden edges. Padding is never a visible key for a real query. Padding query rows may attend their own padding diagonal as a numerical safeguard; their hidden states never contribute to outputs or loss.

This precise role table resolves cases that an option-ID equality rule alone could leave ambiguous. Causality is based on physical indices, not repeated logical position IDs.

## 7. Logical positions

State tokens use positions 0 through $N_S-1$, including the state delimiter. Every question's instruction starts at position $N_S$, independently of previous packed questions. If its final instruction token has position $p$, each candidate branch of length $L_i$ uses:

$$
p+1,p+2,\ldots,p+L_i.
$$

Branch length includes both candidate delimiters. Its decision token uses:

$$
p+\max_i L_i+1.
$$

Thus neither candidate order nor the position of a question in a packed record changes its logical coordinates. Changing the candidate set or its longest length can change the decision representation; this is intentional candidate-set dependence.

Physical token count and maximum logical position are separate limits. Both must be checked, together with memory budget. This design does not assume a backend accepts sequences longer than its declared capacity merely because logical positions repeat.

## 8. Representations and pointer scoring

The backbone produces $H\in\mathbb{R}^{L\times d}$. Read candidate $i$ at its closing delimiter index $e_i$ and the decision at index $a$:

$$
h_i=H[e_i],\qquad h_D=H[a].
$$

Candidate isolation gives $h_i=f(S,Q,O_i)$. The decision representation aggregates the full candidate set through its attention edges. It can read candidate token states throughout the branches, not only their final summaries.

With pointer dimension $d_p=256$, use shared bias-free projections:

$$
q=W_qh_D,\qquad k_i=W_kh_i,
$$

$$
z_i=\frac{q^\top k_i}{\sqrt{d_p}},\qquad
p_i=\frac{\exp(z_i)}{\sum_{j=1}^{K}\exp(z_j)}.
$$

No slot-specific weights, candidate-ID embeddings, or fixed-size output classifier are introduced. Candidate-count padding is excluded from normalization.

## 9. Typed output interpretation

Choice selects the highest probability using the stable-ID tie rule. Score computes:

$$
\text{score}=\sum_i v_i p_i.
$$

The result lies between the minimum and maximum rubric values. For values `[0, 1, 2]` and probabilities `[0, 0.57, 0.43]`, it is 1.43. A fractional mean can reflect uncertainty or competing levels; it does not uniquely describe the distribution.

Noul returns $p_{\text{true}}$. Selecting a Boolean threshold belongs to application code.

For Choice and Score, the v1 confidence field is normalized concentration:

$$
H(p)=-\sum_i p_i\log p_i,\qquad
\text{confidence}=1-\frac{H(p)}{\log K}.
$$

Use $0\log0=0$ and valid candidates only. Confidence is 0 for a uniform distribution and 1 for a point mass. This is an Octo convention, not a claim of matching another product's confidence formula or empirical correctness probability. Calibrated probability claims require held-out evaluation.

## 10. Learning objective

All primitives share categorical cross-entropy. Hard labels identify the correct candidate:

$$
L_q=-\log p_y.
$$

Soft targets describe a distribution $t$:

$$
L_q=-\sum_i t_i\log p_i,\qquad t_i\ge0,\quad\sum_i t_i=1.
$$

Noul truth target $r$ normalizes to `[r, 1-r]`. Score level labels normalize to point masses. If only a fractional Score annotation is available, an explicitly recorded adjacent-level interpolation may preserve its mean; it introduces an uncertainty assumption rather than recovering the original distribution.

For $R$ records with $Q_r$ labeled questions each, the training reduction is:

$$
L_{\text{batch}}=\frac{1}{R}\sum_{r=1}^{R}
\frac{1}{Q_r}\sum_{q=1}^{Q_r}L_{r,q}.
$$

Each record has equal weight, regardless of question count. No token-generation loss, chain-of-thought loss, regression head, or auxiliary ranking loss is required initially.

## 11. Dataset requirements

The dataset must include all primitives, candidate-count variation, descriptive Score rubrics, both Noul truth classes, and grounded questions whose answers follow from the state. Labeling guidelines cover contradictions, incomplete evidence, rubric overlap, and fallback choices.

Keep provenance, source grouping, primitive type, candidate identities, and transformation versions. These fields support auditing and splitting and do not become model evidence. Split by source document or scenario family before augmentation, then retain all related questions and permutations in the same partition. Use separate train, development, calibration, and locked test partitions.

Supervision is either one valid hard label or one finite normalized target distribution. Candidate count, label alignment, physical token budget, logical-position budget, and reserved-token collisions are validated before training.

Start with about 100 simple examples to validate learning, then a representative pilot. Expand toward 255 choices only after small-set quality and correctness pass. Dataset size and mixture ratios remain empirical choices; the reference establishes no production-scale data target.

## 12. Evaluation and release gates

Structural checks precede quality claims:

- Check every edge of small attention fixtures and confirm forbidden weights are zero in the actual attention backend.
- Compare candidate hidden states across permutations and after perturbing another candidate while preserving instruction and relevant positions.
- Compare mapped distributions, Score means, and Noul truth probabilities across physical permutations.
- Compare a question alone against the same question packed beside other questions, including reordered question segments.
- Verify finite logits, normalized distributions, bounded outputs, and exclusion of padding.
- Verify training updates adapters and pointer weights while pretrained parameters stay frozen.
- Verify save/reload preserves predictions.

Measure Choice accuracy and NLL, Noul NLL/Brier score and thresholded accuracy, and Score MAE plus distribution NLL. Report probability calibration and concentration-confidence behavior separately. Stratify by primitive, candidate count, state length, and rubric size. Latency and peak memory measurements include request encoding and masking as well as backbone execution.

A release gate requires recorded numerical tolerances and measured acceptance targets chosen on development data before opening the locked test. No near-exact invariance promise should be made without accounting for precision and backend reduction order.

## 13. Tradeoffs and deferred work

Dense attention masks use $O(L^2)$ storage, and conventional attention remains expensive even when many edges are forbidden. At length 32,768, one fp32 mask alone occupies approximately 4 GiB per batch item. Shared-state packing avoids duplicating state tokens but does not establish constant latency as questions are added.

This topology adapts a pretrained causal model to parallel branches. Pretraining compatibility, semantic quality, calibration, and large-choice behavior require measurement. A model with recurrent or linear-attention layers is unsuitable unless those layers can satisfy the same isolation contract.

Deferred optimizations include sparse/block attention, prefix caching, FlashAttention integration, distributed training, quantization, temperature calibration, and specialized ordinal objectives. They must preserve the validated topology and output mapping.

## 14. Decisions requiring implementation evidence

1. Exact base checkpoint and revision, with verified model configuration.
2. Runtime/device supporting arbitrary masks and repeated positions reliably.
3. Reserved-token assignments or a validated trainable-embedding fallback.
4. Feasible physical context and candidate-count limits on actual hardware.
5. Development-selected optimizer settings, sampling ratios, and quality thresholds.
6. Evidence policy for binary truth supervision.

These are compatibility and experiment decisions, not missing definitions of the pointer architecture.
