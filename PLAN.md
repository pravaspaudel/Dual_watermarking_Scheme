# Plan — Uniform greenlists → detection, dual-layer integration, metrics

Context: greenlists are now **static on Drive** (`green_list_topics_uniform/<topic>.csv`,
~6.3k tokens per topic incl. round-robined connectors + residuals). This plan covers
adapting generation + detection to them, integrating with the KGW layer, and metrics.

**Key decision: no rewrite of detection.** The design in `choices.md` (§1–§5) already
scores against a fixed per-topic list with per-topic γ and context-free positions.
Only the loader and γ recomputation change.

What actually changes:
- Every topic list is now ~6.3k tokens → γ ≈ 0.13–0.25 (was 0.002–0.14).
- Power increases for ALL topics; micro-topics (medicine/entertainment/finance)
  stop being hopeless.
- δ calibration must be redone: boosting a much larger set behaves differently.
- Loader change: old `load_topic_greenlists()` reads ONE csv with a `topic`
  column; new layout is one CSV PER topic with a `type` column
  (`content|connector|residual`).

Team: **A** (generation side), **B** (detection side). ~2 weeks.

---

## Phase 0 — Setup & decisions (both, half day)

- [ ] Freeze config: lists from `/content/drive/MyDrive/minor_project/green_list_topics_uniform/`,
      `N_TARGET=200` tokens, seeds {0,1,2}, same OPT checkpoint used for embeddings.
- [ ] Agree shared interfaces BEFORE splitting:
  - loader signature: `load_uniform_greenlists(uniform_dir)`
  - results CSV schema:
    `prompt, topic, delta, delta_private, seed, text_type{plain|topic|dual}, text`

## Phase 1 — Generation pipeline v2 (A, days 1–3)

- [ ] Write `load_uniform_greenlists(uniform_dir)` →
      `{topic: {"all": ids, "content": ids}}` using the `type` column;
      replace uses of `first_layer.load_topic_greenlists`.
- [ ] Point `TopicWiseWatermarking` at the directory; keep embedding-table/topic-matrix
      caching. `TopicBoostProcessor` needs ZERO changes — just a bigger id tensor.
- [ ] Regenerate data grid: 8 topics × ~20 prompts × δ ∈ {0,1,2,3,4,6} × seeds.
      Save plain + watermarked texts to Drive.
- [ ] Reuse prefix-evaluation trick (choices.md §9): generate 300 tokens once,
      score at prefixes 50/100/200/300.
- [ ] Pick `delta_topic` = smallest δ with ≥95% confirmation at n=200 across topics
      (expect LOWER than the old 3–4 since γ is now healthy everywhere).

## Phase 2 — Detection v2 (B, days 1–3)

- [ ] Update `notebooks/08_topic_watermark_detection.ipynb` + exported
      `src/detection/topic_detection.py`: load uniform dir,
      compute `gamma_t = |G_t ∩ vocab| / vocab_size` from the files themselves.
      Z-score / exact-binomial logic untouched.
- [ ] Sanity ladder (choices.md §6):
  1. hand-written clean text → z ≈ 0, not confirmed;
  2. saved example pairs (`data/example/topic_wise_watermarked_example.csv`) → separation;
  3. live δ sweep → z climbs monotonically with δ;
  4. cross-check: exported module reproduces notebook verdicts.
- [ ] Verify topic-routing recovery rate did not regress with new lists
      (detected topic vs generation-time topic).
- [ ] Optional experiment: content-only scoring (drop connector/residual rows via
      `type`) vs all-token scoring — report both.

## Phase 3 — Dual-layer integration (pair up, days 4–6)

- [ ] Combined generation: pass BOTH processors in one call —
      `logits_processor=[TopicBoostProcessor(...), PrivateWatermarkProcessor(...)]`
      (they compose additively on logits).
- [ ] Detection cascade: public topic screen first (keyless) → private KGW ownership
      check on the same text when needed. Wrap into one `detect_dual(text)` returning
      both verdicts.
- [ ] Interference check: each layer's score with the other layer ON vs OFF
      (choices.md §9 combined cell). If KGW drops below threshold at
      `delta_private ≈ 0.5–0.7`, nudge upward until stable.
- [ ] Edge cases: short excerpts <100 tokens (KGW-only claim), off-topic routing
      mismatch between generate-time and detect-time topic inference,
      empty/degenerate generations.

## Phase 4 — Metrics (days 7–9)

### A — quality
- [ ] Perplexity under an INDEPENDENT judge model (GPT-2-large / Pythia — NOT OPT;
      self-scoring biases toward the watermark).
- [ ] Table: plain vs topic-only vs dual → mean PPL ± std over topics/seeds.
      Plot PPL vs δ.
- [ ] Diversity: distinct-1/2/3 minimum; repetition check at high δ.

### B — watermark performance
- [ ] ROC curves per layer + dual; report TPR @ FPR = 1%.
- [ ] z vs text-length curves (from prefix scoring).
- [ ] FPR calibration on ≥500 clean C4 continuations per topic.
- [ ] Stretch: paraphrase / truncation robustness mini-test.

## Phase 5 — Write-up (both, days 10+)

- [ ] Results tables + figures committed to repo.
- [ ] Update `choices.md`: uniform-list findings (how round-robin changed γ and δ),
      final operating config.
- [ ] Report sections mapped to phases above.

---

## Sequencing note

A's Phase 1 output blocks B's Phase 2 real-data runs → B starts Phase 2 with
hand-written/sanity checks while A generates. Pair up only in Phase 3 — that is
where cross-layer bugs will surface.
