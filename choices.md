# Design choices — topic-level watermark detection (`notebooks/08_topic_watermark_detection.ipynb`)

Decisions made while building the detection notebook for the first-layer
(public, topic-based) watermark generated in `topic_wise_gen_pipeline.ipynb`.

## 1. Detection statistic: z-score + exact binomial p-value

- Under H0 (unwatermarked text), each token independently lands in topic `t`'s
  greenlist with probability `gamma_t = |G_t| / vocab_size`. The number of green
  hits in `n` tokens is therefore Binomial(n, gamma), and we test the **upper tail**:
  - `z = (s - n*gamma) / sqrt(n*gamma*(1-gamma))`
  - `p_value = P(Binom(n, gamma) >= s)` via `scipy.stats.binomtest(..., alternative="greater")`
- Chosen because it is exactly the statistical core already used by
  `src/detection/kgw_detection.py` (same library call, same "greater" alternative),
  so both layers of the dual scheme report comparable, interpretable numbers.
- Both z and p are reported: z is intuitive ("std devs above chance", thresholdable,
  matches the KGW paper's tau-style rule) while p-value is what actually controls
  false positives.

## 2. Decision rule: `confirmed = (z >= Z_THRESHOLD) and (p < P_VALUE_THRESHOLD)`

- Mirrors `kgw_detection.py`, which requires `ownership_score > threshold AND
  p_value < p_threshold`. Defaults here: `Z_THRESHOLD = 4.0`,
  `P_VALUE_THRESHOLD = 0.05`.
- z=4 corresponds to a normal-theory FPR of ~3e-5; the notebook includes a sweep
  table showing FPR at looser thresholds so the trade-off is explicit.

## 3. Per-topic gamma instead of a global green fraction

- Greenlist sizes vary enormously across topics (technology ~7100 tokens vs
  medicine ~95 out of vocab 50272 -> gamma from ~0.14 down to ~0.002). Using one
  global fraction would wildly miscalibrate the null distribution.
- Consequence worth knowing: topics with tiny gamma have very low power on short
  texts (few expected hits); the notebook prints gamma next to every result.

## 4. Every token position is scored — no context window

- Unlike the private KGW layer, where the green set depends on the previous
  tokens and positions before `prev_token_size` must be skipped, this greenlist
  is **fixed per topic**, so all `n` tokenized positions are scoreable. Larger n
  directly increases power.

## 5. Topic re-inference reuses the generation-side geometry exactly

- Same mean-pooled token embeddings, same unit-normalization of OPT's input
  embedding table, same `" " + topic` first-subtoken vectors as
  `topic_wise_gen_pipeline.ipynb` / `src/watermark/first_layer.py`. Any mismatch
  would route detection to the wrong greenlist. The notebook verifies routing by
  comparing detected vs generation-time topic on saved examples.

## 6. Sanity-check strategy (no local execution)

- The repo policy is that notebooks run only on Colab, so the notebook validates
  itself when executed there, in increasing strength:
  1. smoke test on hand-written clean text (expect z ≈ 0, not confirmed);
  2. saved pairs from `data/example/topic_wise_watermarked_example.csv`
     (plain vs watermarked separation + topic-routing recovery rate);
  3. live delta sweep `[0, 1, 2, 4, 6]` with fixed seed — z should climb with delta;
  4. cross-check cell asserting the exported module reproduces the notebook verdicts.

## 7. Exported module: `src/detection/topic_detection.py` via `%%writefile`

- Follows the existing pattern (`topic_wise_gen_pipeline.ipynb` writes
  `first_layer.py` the same way) so Colab stays the single source of execution.
- A small `TopicWatermarkDetector` class caches the unit-norm embedding table,
  topic matrix and green sets once; plain per-call functions (the kgw style)
  would renormalize a 50k x d matrix on every detection. `detect()` returns the
  same dict keys as `detect_private_watermark` (`ownership_score`, `match_count`,
  `num_positions`, `p_value`, `confirmed`, plus topic fields) to keep downstream
  code symmetric between layers.

## 8. Colab-first ergonomics

- Setup cells clone the repo, `%cd` into it, install runtime deps only there;
  nothing is executed or installed locally during development.
- CSV lookup tries `data/example/`, `../data/example/`, `data/complete/`,
  `../data/complete/` so it works whether the cwd is repo root or `notebooks/`
  (same pattern as `05_greenlist_construction.ipynb`). Example-sized CSVs are
  preferred for quick runs; switch to `data/complete/` for full greenlists.

## 9. Delta calibration on LONG texts for the dual-layer setting (Appendix A)

The dual scheme applies both watermarks to long outputs, so the topic-layer delta
is finalized empirically in Appendix A of the notebook rather than assumed:

- **Exact binomial power analysis first** (computed with stdlib math, no GPU):
  detection needs `s >= n*gamma + Z*sqrt(n*gamma(1-gamma))` hits. Inverted per
  topic, the boost must lift the green rate ~2x over natural for technology /
  politics / sports / science / history, but **6-21x** for entertainment /
  finance / medicine (gamma ~0.002). Conclusion: no sane delta rescues the three
  micro-list topics — they are excluded from calibration and flagged as a
  greenlist-construction task (more round-robin connectors or lower threshold),
  not a boosting task.
- **Prefix evaluation trick**: one 300-token generation per
  `(topic, seed, delta)` is scored at prefixes of 50/100/200/300 tokens. This
  yields a full length-vs-delta confirmation table from a fraction of the
  generations a naive grid would need.
- **Decision rule**: smallest delta achieving >=95% confirmation at the dual
  pipeline's operating length `N_TARGET = 200`, across the five viable topics;
  expected outcome from the power math is **delta = 3.0-4.0** (vs KGW's classic
  2.0), because fixed-list boosting competes against fluent high-probability
  tokens and quality visibly degrades only above ~4-6.

### Why the second layer can be made weaker

The layers have different jobs: the public topic layer is a cheap,
anyone-runnable screen (no key needed) that works over long text; the private
KGW layer is the ownership proof. If the topic layer runs at delta ≈ 4 on
n >= 200 tokens it is essentially always confirmed, so the private layer no
longer needs to carry detectability on its own and can drop to a gentler
`delta_private` (~0.5-0.7, vs the usual ~1.0+ tuning) — recovering output
quality where both boosts stack. Appendix A includes a combined-generation
cell (`TopicBoostProcessor` + `PrivateWatermarkProcessor`) verifying each
layer still detects through the other's noise before committing to this split.
Caveat: keep the private layer strong enough for SHORT excerpt ownership
claims, since the topic layer legitimately cannot help there.

## 10. Known limitations (documented, accepted for now)

- Public greenlists: anyone with the CSV can scrub/paraphrase the watermark;
  this layer is meant for attribution/demonstration, not adversarial security.
- Topic inference can disagree with generation-time assignment on short or
  off-topic texts; the detector reports the inferred topic so mismatches are visible.
- Sampling randomness means a delta sweep should be repeated with several seeds
  before quoting exact z values in the report.
