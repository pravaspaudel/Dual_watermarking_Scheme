# Eval Plan — Dual Watermarking Scheme (Phases E0–E5)

Everything below runs on **Google Colab** (local machines have no GPU). Each phase maps to its
own notebook so a dead session never loses more than one phase. All notebooks follow the repo
convention: clone → `%cd` → install → run → persist artifacts to Google Drive.

Companion docs: `PLAN.md` (build phases), `choices.md` (design decisions §1–§10).

---

## 0. Frozen eval config

| Item | Value |
|---|---|
| Watermarked model | `facebook/opt-2.7b` via `src.utils.model.load_model()`, vocab = `len(tokenizer)` |
| Greenlists | static uniform CSVs, `data/greenlist/<topic>.csv`, 8 topics, split=`all` |
| Layers | L1 public topic boost (`DualLayerWatermarkProcessor` w1=1,w2=0), L2 private KGW (w1=0,w2=1), dual (w1=w2=1) |
| Code defaults | `green_fraction=0.5`, `prev_token_size=5`, `delta_private=0.7`, temp=1.0, top_p=0.9, `max_new_tokens=200` |
| Key | `WATERMARK_SECRET_KEY` from `.env` (`src/utils/key_manager.py`) — SAME key for generation and detection |
| Eval corpus | `data/extracted/c4_samples.csv` (~1000 C4 realnewslike docs; cols `prompt_text, prompt_tokens, reference_text, reference_tokens, full_text, domain`) |
| Main grid | **N=200 prompts**, seed=0, 4 variants × 200 tokens → 800 generations |
| Delta sweep | 50-prompt subset × `delta_public ∈ {0,1,2,3,4}` (dual only, `delta_private=0.7`) → 250 generations |
| PPL judge | **`Qwen/Qwen1.5-7B` fp16** (needs A100/L4 runtime; T4 → 4-bit `bitsandbytes`; emergency fallback `gpt2-large`) |
| Human/FPR corpus | `reference_text` column of the same C4 csv (human-written continuations, ≥500 scored) |
| Decision rule | thresholds picked for **empirical FPR ≤ 1%**; also report numbers at legacy defaults (z≥4, ownership>0.60) |

**Operating deltas**: E1 generates the sweep, E2+E3 pick the smallest `delta_public` whose
TPR@FPR=1% ≥ 95% *without* wrecking PPL/diversity → frozen there for all headline report numbers.

### Drive layout

```
/content/drive/MyDrive/minor_project/
├── green_list_topics_uniform/          # already there (generation-time source)
└── eval_results/
    ├── generation/gen_main.csv         # E1
    ├── generation/gen_sweep.csv        # E1
    ├── detection/detection_results.csv # E2
    ├── detection/thresholds.json       # E2
    ├── quality/quality_results.csv     # E3
    ├── robustness/robustness_results.csv # E4
    └── figures/                        # E5 (all pngs, dpi=300)
```

### Shared results schema (`gen_main.csv`)

```
id, prompt_text, topic, variant{plain|l1_only|l2_only|dual},
delta_public, delta_private, seed, gen_tokens, text
```

`sweep` adds column `delta_public_sweep`. One row per (prompt, variant).

---

## Phase E0 — Shared setup (½ day)

**Goal:** one battle-tested header block every eval notebook pastes verbatim.
**Notebook:** none (snippet lives in this doc; copy into cells 0–3 of every 11–15 notebook).

Steps:
1. Clone repo + install deps + mount Drive:

```python
# ---- paste as the first two cells of EVERY eval notebook ----
!git clone https://github.com/pravaspaudel/Dual_watermarking_Scheme.git
%cd Dual_watermarking_Scheme
!pip install -q transformers scipy pandas matplotlib accelerate bitsandbytes sentencepiece

import os
from google.colab import drive
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/minor_project/eval_results', exist_ok=True)

import torch, pandas as pd, numpy as np
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

from src.utils.model import load_model                      # opt-2.7b + tokenizer + vocab_size
from src.watermark.dual_layer import DualWaterMarking       # generator wrapper (w1/w2 switches)
from src.detection.topic_detection import prepare, detect_topic_watermark, detect_dataframe
from src.detection.kgw_detection   import detect_private_watermark, detect_dataframe as kgw_detect_dataframe

model, tokenizer, VOCAB_SIZE = load_model("facebook/opt-2.7b")

C4_CSV    = "data/extracted/c4_samples.csv"
GREEN_DIR = "data/greenlist"
DRIVE     = "/content/drive/MyDrive/minor_project"
RESULTS   = f"{DRIVE}/eval_results"

def save_csv(df, rel):                     # crash-safe write
    path = f"{RESULTS}/{rel}"
    df.to_csv(path, index=False)
    print("saved", path, len(df), "rows")
# -------------------------------------------------------------
```

2. Load generator once per notebook that needs it:
   `wm = DualWaterMarking(model, tokenizer, greenlist_dir=GREEN_DIR, split="all",
   max_new_tokens=200)` — read deltas/weights per-experiment rather than baking them in.
3. Load detection state once: `state = prepare(model, tokenizer, greenlist_dir=GREEN_DIR)`
   and print per-topic γ table (`len(state['green_sets'][t]) / VOCAB_SIZE`) — sanity vs choices.md §3.
4. Verify `.env` carries the same `WATERMARK_SECRET_KEY` used at generation time; assert non-empty.

**Outputs:** the snippet above; γ table screenshot for the report appendix.
**Acceptance checks:** model loads on GPU; 8 topics listed alphabetically; γ values match
`05_greenlist_construction` outputs (±0); key present.

---

## Phase E1 — Evaluation corpus generation (~1 day)

**Notebook:** `notebooks/11_eval_generation.ipynb`
**Inputs:** C4 csv, `DualWaterMarking`, config above. **Runtime:** ~45–75 min total on T4
(main grid ~40–60 min; sweep ~15–25 min on top). Use A100/L4 if available → 3–4× faster.

Steps:
1. Sample prompts: `prompts = df.sample(n=200, random_state=0)["prompt_text"].tolist()` and
   remember the indices; the same rows' `reference_text` become the human/FPR corpus later.
2. Main grid — for each prompt, one `DualWaterMarking` instance per delta config is wasteful;
   instead instantiate once with `delta_public=DP_OP, delta_private=0.7` and call
   `wm.watermark(prompts_chunk, include_single_layers=True)` in chunks of 25 with
   intermediate `save_csv` after every chunk (**checkpoint-resume**: reload csv, skip done ids).
   Map columns → `variant`: `plain_output→plain`, `layer1_only_output→l1_only`,
   `layer2_only_output→l2_only`, `dual_watermarked_output→dual`.
3. Per-row seed discipline: set `torch.manual_seed(SEED + i)` immediately before each prompt's
   four generations so plain/watermarked pairs share the sampling stream start (fair pairing).
4. Sweep subset: first 50 sampled prompts; loop `delta_public ∈ {0,1,2,3,4}` with
   `w1=1, w2=1` (dual), regenerate fresh processor per delta; store `gen_sweep.csv`.
5. QA cell: no NaN/empty texts; token-length histogram of all variants (expect ≈200);
   spot-print one prompt + its 4 variants.

**Outputs:** `eval_results/generation/gen_main.csv` (800 rows), `gen_sweep.csv` (250 rows).
**Acceptance checks:** exact row counts; zero empty generations; all four variants present per
id; lengths within 180–220 tokens for ≥95% of rows; resumed run reproduces identical row count.

```text
PROMPT (paste to your AI agent in Colab / opencode):

Work in the cloned repo Dual_watermarking_Scheme on Colab. Create notebooks/11_eval_generation.ipynb
cells that do EXACTLY this, using the existing modules (do NOT modify src/):

1. Read data/extracted/c4_samples.csv, sample 200 rows with pandas .sample(random_state=0),
   keep column prompt_text and the integer index (needed later for the human-reference corpus).
2. Build DualWaterMarking(model, tokenizer, greenlist_dir="data/greenlist", split="all",
   max_new_tokens=200, delta_public=<OP>, delta_private=0.7) where <OP> comes from a CONFIG
   dict at the top (delta_public=2.0 initially).
3. For i, prompt in enumerate(prompts): torch.manual_seed(1000+i); call
   wm.watermark([prompt], include_single_layers=True); reshape to long format
   (id=i, prompt_text, topic, variant in {plain,l1_only,l2_only,dual}, delta_public,
   delta_private, seed=1000+i, gen_tokens=len(tokenizer.encode(text)), text).
   Process in chunks of 25 prompts; after each chunk append to
   /content/drive/MyDrive/minor_project/eval_results/generation/gen_main.csv and flush.
   On startup, if that csv exists, load it and skip ids already present (resume support).
4. Sweep: first 50 of those prompts; for delta_public in [0,1,2,3,4] rebuild the processor via
   wm.delta_public = d and wm._make_processor(topic, 1.0, 1.0) semantics (or reconstruct
   DualWaterMarking with that delta); generate ONLY the dual variant; save gen_sweep.csv
   with an extra column delta_public_sweep, same chunked-resume pattern.
5. Final QA cell: assert row counts 800 and 250; assert no empty text; plot gen-token-length
   histograms per variant (matplotlib, dpi=300, saved to the Drive figures dir); print 1 example.
Print progress with tqdm. Keep every magic number in a single CONFIG dict at the top.
```

---

## Phase E2 — Detection performance & statistical power (~1 day)

**Notebook:** `notebooks/12_eval_detection.ipynb`
**Inputs:** E1 csvs + C4 `reference_text` (human). No new generation needed.
**Runtime:** topic detection is cheap (<5 min); KGW detection ≈0.1–0.3 s/text on CPU →
~1050 wm texts + 500 human refs ≈ 20–35 min. GPU barely used.

Steps:
1. Score EVERYTHING with BOTH detectors:
   - L1: `detect_topic_watermark(text, tokenizer, state, vocab_size=VOCAB_SIZE)` per row
     (vectorize with `detect_dataframe` where convenient).
   - L2: `detect_private_watermark(text, tokenizer, key=KEY, vocab_size=VOCAB_SIZE,
     green_fraction=0.5, prev_token_size=5, threshold=0.60, p_value_threshold=0.05)`.
   - Store raw statistics (z_score, p_value, ownership_score, match_count, num_positions,
     detected topic) — NOT just booleans, so thresholds can be re-picked later without rerunning.
2. FPR calibration: score ≥500 human `reference_text` rows; empirical FPR of each statistic;
   pick thresholds targeting FPR ≤ 1% (e.g., z* = 99th percentile of human z's for L1;
   ownership* analogously for L2). Save `thresholds.json`. Also report FPR at legacy defaults.
3. ROC curves: for each variant (plain/l1_only/l2_only/dual) × each detector statistic, sweep
   the threshold → TPR/FPR points; compute AUC; report **TPR @ FPR=1%** table.
   Dual rule reported two ways: `both_confirmed` (AND — ownership proof) and `either_confirmed`
   (OR — loose screen); mark AND as primary.
4. Power vs length: reuse the prefix trick on `gen_main` dual rows — rescore prefixes of
   50/100/150/200 tokens → median-z-vs-length curves per variant.
5. Topic-routing recovery: detected topic vs generation-time `topic` on wm texts → confusion
   matrix heatmap + accuracy.
6. Freeze operating point: smallest `delta_public` from the sweep achieving dual
   TPR@FPR=1% ≥ 0.95 at n=200 → write into `thresholds.json` as `frozen_delta_public`.

**Outputs:** `detection_results.csv` (one row per text × detector), `thresholds.json`,
ROC/z-histogram/z-vs-length/routing figures.
**Acceptance checks:** plain rows' z-median within ±1 of 0 for L1; empirical FPR at chosen
thresholds ≤ 1%; dual ROC dominates l1-only and l2-only ROCs; routing recovery ≥90% on-topic;
`frozen_delta_public` recorded and propagated to E3/E4 configs.

```text
PROMPT (paste to your AI agent):

Create notebooks/12_eval_generation-detection logic in notebooks/12_eval_detection.ipynb using
ONLY existing src/ modules. Inputs: gen_main.csv, gen_sweep.csv from Drive, plus
data/extracted/c4_samples.csv reference_text as HUMAN text (score >=500 rows).

1. state = prepare(model, tokenizer, greenlist_dir="data/greenlist") ; KEY from .env.
2. For every row of both csvs AND the human refs compute and store RAW stats:
   L1 -> detect_topic_watermark(text, tokenizer, state, vocab_size=VOCAB_SIZE):
         z_score, p_value, ownership_score, num_positions, topic.
   L2 -> detect_private_watermark(text, tokenizer, key=KEY, vocab_size=VOCAB_SIZE,
         green_fraction=0.5, prev_token_size=5, threshold=0.60, p_value_threshold=0.05):
         ownership_score, p_value, num_positions.
   Long-format dataframe: source{text_type}, variant, detector{l1,l2}, stats..., split='test'|'human'.
   Wrap in tqdm; chunk-save detection_results.csv to Drive every 100 rows (resumable like E1).
3. FPR calibration: 99th-percentile thresholds on human rows for l1.z_score and l2.ownership_score
   -> thresholds.json {z_star, own_star, empirical_fpr_at_defaults:{...}, frozen_delta_public:null}.
4. ROC: sklearn-free manual threshold sweep over sorted scores; plot ROC per variant for EACH
   detector (log-x), annotate AUC and TPR@FPR=1%; ALSO build the combined dual rule
   (l1 z>=z_star AND l2 own>=own_star => positive) as its own ROC line. dpi=300 to Drive figures.
5. Prefix power: for 100 random dual rows rescore text[:k tokens] for k in {50,100,150,200}
   (re-tokenize prefix by ids[:k]); median z per k per variant -> line plot.
6. Routing confusion matrix (generation topic vs detected topic) heatmap + accuracy printout.
7. Frozen operating point: from gen_sweep.csv find smallest delta_public_sweep whose dual
   TPR@FPR=1% >= 0.95 at n=200; update thresholds.json.
Print all acceptance checks explicitly at the end.
```

---

## Phase E3 — Output quality (~1 day)

**Notebook:** `notebooks/13_eval_quality.ipynb`
**Inputs:** E1 csvs (+sweep for the δ-curve). **Runtime:** Qwen1.5-7B forward passes over
~1300 texts ≈ 10–20 min on L4/A100; diversity metrics CPU-only seconds.
⚠️ Needs **A100/L4** for fp16-7B; on T4 use `load_in_4bit=True`.

Steps:
1. Judge = `AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B", torch_dtype=torch.float16,
   device_map="auto")` (+ its tokenizer). Score PPL of each generated text (continuation-only
   scoring preferred: feed prompt+text, mask/ignore prompt-token losses — implement via labels
   mask; fall back to full-text PPL consistently applied if masking proves fiddly, but SAY so).
   Batch by padding-left; fp16; no gradients.
2. Table 1 (headline): mean±std PPL per variant (plain / l1_only / l2_only / dual) over the
   200 prompts. Violin or box plot. Paired per-prompt deltas (dual − plain) for a significance
   note (Wilcoxon signed-rank, scipy).
3. Quality-vs-strength: from `gen_sweep.csv`, mean PPL vs `delta_public_sweep` curve — this is
   THE trade-off plot against E2's TPR-vs-δ curve.
4. Diversity per variant: distinct-1/2/3 (unique n-grams / total n-grams) + max repeated-n-gram
   rate; grouped bar chart. Flag degenerate repetition at high δ.
5. Optional stretch: BERTScore (or embedding cosine via sentence-transformers) of each variant
   against its plain twin → semantic drift estimate.

**Outputs:** `quality_results.csv` (row per text: variant, ppl, distinct1/2/3, rep_rate),
PPL plots, headline markdown table ready for the report.
**Acceptance checks:** plain PPL in a sane band (single digits–low tens under a 7B judge);
l2_only PPL ≈ plain (δ=0.7 should be gentle); dual degradation quantified; distinct-3 not
collapsed (>0.7·plain's value) at the frozen operating point; judge dtype/device logged.

```text
PROMPT (paste to your AI agent):

Create notebooks/13_eval_quality.ipynb. Do NOT touch src/. Judge model Qwen/Qwen1.5-7B fp16
device_map="auto" (if OOM on T4 retry with bitsandbytes 4-bit and LOG that fact into the
results metadata json).

1. Perplexity: tokenize PROMPT_TEXT + GENERATED_TEXT jointly with the QWEN tokenizer; compute
   NLL only over the continuation tokens (labels=-100 on the prompt part); ppl=exp(mean NLL).
   Batch size 8 with left padding; torch.no_grad; fp16.
2. Merge scores onto gen_main.csv rows -> quality_results.csv with columns
   id, variant, delta_public, ppl, distinct1, distinct2, distinct3, rep_rate
   (distinct-k = unique k-grams / total k-grams computed on lowercase word tokens after basic
   punctuation stripping; rep_rate = fraction of tokens inside their most frequent 5-gram).
3. Headline table: groupby(variant).ppl.agg(['mean','std']) rendered as markdown AND saved.
   Wilcoxon signed-rank scipy.stats.wilcoxon(dual_ppl, plain_ppl) p-value printed.
4. Sweep curve: gen_sweep.csv -> groupby(delta_public_sweep).ppl.mean() line plot titled
   "Quality vs watermark strength"; overlay plain baseline as horizontal dashed line. dpi=300.
5. Diversity grouped-bar chart per variant for distinct-1/2/3. dpi=300.
6. Everything saved to Drive eval_results/quality/ and figures/.
Print acceptance checks: plain ppl band, l2_only vs plain gap %, dual vs plain gap %,
distinct3 ratio dual/plain.
```

---

## Phase E4 — Robustness mini-suite (stretch, ~½ day)

**Notebook:** `notebooks/14_eval_robustness.ipynb`
**Inputs:** `gen_main.csv` dual + l1_only rows (say 100 each). Runtime: CPU mostly; re-detection
reuses E2 costs (~15 min).

Steps:
1. Truncation: already covered by E2 prefix curves — cite, don't redo.
2. Paraphrase-lite attacks (deterministic, no extra model needed):
   - word-level synonym swap via a small mapping or NLTK WordNet at rates {10%, 20%, 30%};
   - random sentence order shuffle (for multi-sentence texts);
   - whitespace/format noise (casual copy-paste simulation).
3. Re-run BOTH detectors on attacked texts → survival = fraction still confirmed at frozen
   thresholds; bar chart TPR vs attack strength per layer and dual.
4. Document honestly: these are weak adversaries (public lists are scrubable by design —
   choices.md §10); frame as "robustness to incidental editing", not adversarial attack.

**Outputs:** `robustness_results.csv`, survival bars figure.
**Acceptance checks:** monotonic survival decay with attack strength; dual survives ≥80% at
20% synonym swap (expected, list is huge); KGV/L2 survival reported separately.

```text
PROMPT (paste to your AI agent):

Create notebooks/14_eval_robustness.ipynb. Take 100 dual + 100 l1_only rows from gen_main.csv.
Implement three deterministic perturbations WITHOUT any neural paraphraser:
(a) NLTK WordNet synonym swap at {0.1,0.2,0.3} rates (nltk.download('wordnet') + omw-1.4;
    only replace when POS tag is verb/noun/adjective and synset exists);
(b) sentence shuffle: split on '. ' keep capitalization simple, shuffle with fixed rng;
(c) formatting noise: collapse double spaces, strip punctuation at 30% of positions.
For each attacked version recompute L1 (detect_topic_watermark with prepared state) and L2
(detect_private_watermark with .env key) stats; confirmed = frozen thresholds from
thresholds.json (load it; fall back to z>=4 / own>=0.6 if missing).
Output robustness_results.csv: id, original_variant, attack, rate, l1_z, l1_confirmed,
l2_own, l2_confirmed, dual_confirmed(AND-rule).
Bar chart: x=attack rate, y=survival fraction, groups={l1_only, dual}; separate panel per
attack type. dpi=300. Print survival table to console and Drive.
```

---

## Phase E5 — Consolidated report pack (~½ day)

**Notebook:** `notebooks/15_report_figures.ipynb`
**Goal:** one click regenerates EVERY figure/table in consistent style for the written report.

Steps:
1. Common style header: `matplotlib.rcParams` (font size 11, dpi=300, colorblind-safe palette,
   one color per variant everywhere: plain=gray, l1=tab:blue, l2=tab:orange, dual=tab:green).
2. Regenerate the full figure checklist (Appendix A) from the saved CSVs — no model loading
   except nothing: pure plotting from Drive artifacts (fast, CPU).
3. Master summary table: per variant → TPR@FPR=1% (L1-det / L2-det / dual-AND), AUC,
   mean PPL ± std, distinct-3, routing accuracy; plus the frozen config footer
   (δ's, thresholds, key fingerprint hash, seeds, judge model, runtime types used).
4. Zip `eval_results/` → download link; commit figures + summary csv to repo `report_assets/`.

**Acceptance checks:** every figure opens, has axis labels + legend + caption-ready title;
summary table numbers match the per-phase csvs exactly.

```text
PROMPT (paste to your AI agent):

Create notebooks/15_report_figures.ipynb that loads ONLY csv/json artifacts from Drive
eval_results/ (no models) and rebuilds the 10-figure checklist with a shared rcParams header
and fixed variant->color map. Produce: (1) z histograms per variant (l1 det);
(2) ownership histograms per variant (l2 det); (3) ROC grid l1/l2/dual-AND with AUC annotations;
(4) TPR@FPR=1% bar chart; (5) median-z vs length lines; (6) PPL violins; (7) PPL vs delta_public
with TPR-vs-delta_public twin-axis overlay (THE trade-off figure); (8) distinct-1/2/3 bars;
(9) routing confusion heatmap; (10) robustness survival bars.
Then assemble master_summary.csv + a markdown rendering with the frozen-config footer
(deltas, thresholds, sha256(key)[:8], seeds, judge, colabs runtimes used). Save all to
Drive figures/ and copy into repo folder report_assets/.
```

---

## Sequencing & ownership

```
E0 ──> E1 ──> E2 ──┬──> E3 (needs frozen deltas ideally, but can run at defaults in parallel)
                   └──> E4 (needs E2 thresholds)
E3/E4 ──> E5 (pure aggregation)
```

- E1 blocks everything real → start it first; while it runs, build E2's scoring machinery and
  test on `data/example/*` pairs + human refs (those need no E1 output).
- E3 can run early on the sweep subset to get an early PPL-vs-δ read.
- Teammates can split: one owns E1+E2 (detection person), other owns E3 (+judge debugging),
  E4/E5 pair up.

## Appendix A — Report figure checklist

| # | Figure | Source phase |
|---|---|---|
| 1 | z-score distributions per variant (L1 detector, plain overlay) | E2 |
| 2 | ownership-score distributions per variant (L2 detector) | E2 |
| 3 | ROC curves: L1 / L2 / dual-AND (+AUC, log-x) | E2 |
| 4 | TPR@FPR=1% bar chart per variant | E2 |
| 5 | median z vs text length (50/100/150/200) | E2 |
| 6 | PPL violin/box per variant | E3 |
| 7 | PPL vs δ_public with TPR-vs-δ overlay (trade-off) | E2+E3 |
| 8 | distinct-1/2/3 grouped bars per variant | E3 |
| 9 | topic-routing confusion matrix heatmap | E2 |
| 10 | robustness survival bars per attack | E4 |

## Appendix B — Known gotchas

- **KGW detection cost**: `derive_set` permutes the whole 50k vocab per token position.
  Never nest it in another loop; ~0.1–0.3 s/text is normal; chunk-save results.
- **Two tokenizers**: OPT ids for everything watermark-related; the Qwen judge only ever sees
  decoded strings. Never mix id spaces.
- **Retokenization drift**: detectors re-encode `skip_special_tokens=True` strings; this is
  fine because it applies identically to all variants — just don't compare against
  generation-time id counts.
- **Same key everywhere**: generation and detection must load the identical
  `WATERMARK_SECRET_KEY`; a regenerated `.env` silently breaks L2 detection. Assert the
  key fingerprint (sha256[:8]) matches the one used in `gen_*.csv` metadata.
- **Colab session death**: every long loop must chunk-append to Drive and skip-done-ids on
  restart. Treat any notebook that can't resume as unfinished.
- **Qwen memory**: fp16 7B ≈ 15 GB → A100/L4 required; T4 needs 4-bit; log which was used.
- **Seeds**: per-row `seed = SEED_BASE + id` set right before that row's generations keeps
  plain↔watermarked pairs comparable and runs reproducible.
