# Event-Aware Token Allocation for Long Video Reasoning

Research codebase for adaptive frame allocation in long-form video understanding.

## Status

**Mock-data phase.** All 4 modules and 3 method pipelines are implemented and pass 36 unit tests. The full pipeline runs end-to-end on synthetic data without any GPU.

Current mock-data results (50 examples, T=32):

| Method | Accuracy | Notes |
|--------|---------:|-------|
| Uniform baseline | 78.0% | Primary baseline |
| **Event-Aware (ours)** | **92.0%** | Proposed |
| Oracle (upper bound) | 100% | Cheat with GT event |

→ Our method captures **63.6%** of the oracle-uniform gap.

Budget sweep shows the expected pattern — largest gains at mid-range budgets (T=16–32), converging at the extremes. This is the accuracy-budget tradeoff curve that will be the paper's main figure.

## Project Structure

```
event_aware_long_video/
├── src/
│   ├── data/
│   │   └── mock_generator.py      # Synthetic video+QA generator
│   ├── models/
│   │   ├── mock_encoder.py        # CLIP/SigLIP stand-in (TODO: REAL)
│   │   └── mock_mllm.py           # Qwen2-VL stand-in    (TODO: REAL)
│   ├── modules/                   # Core algorithmic modules (pure numpy)
│   │   ├── segmentation.py        # Event boundary detection
│   │   ├── importance.py          # Complexity x Relevance
│   │   ├── allocation.py          # Adaptive token allocation
│   │   └── selection.py           # Frame selection within event
│   ├── methods/
│   │   ├── base.py                # Abstract QA method interface
│   │   ├── uniform.py             # Uniform baseline
│   │   ├── oracle.py              # Oracle (pilot study)
│   │   └── event_aware.py         # Our proposed method
│   ├── eval/
│   │   └── runner.py              # Evaluation harness
│   └── utils/
│       └── types.py               # Core dataclasses
├── scripts/
│   ├── pilot_study.py             # Oracle vs Uniform vs Ours
│   ├── budget_sweep.py            # Accuracy-budget curve
│   └── run_ablation.py            # Which components matter
├── tests/                         # 36 unit tests, all passing
└── requirements.txt
```

## Running

```bash
# Install
pip install -r requirements.txt

# Run unit tests (should show 36 passed)
python -m pytest tests/ -v

# Run the key experiments
python scripts/pilot_study.py
python scripts/budget_sweep.py
python scripts/run_ablation.py
```

All of the above run on CPU in under 10 seconds. No GPU needed.

## Architecture Highlights

### Two-stage decoupling

The proposed pipeline separates scoring (Stage 1) from reasoning (Stage 2) to break the circular dependency between token allocation and video encoding:

```
Long video
  -> [Stage 1] Light encoder (CLIP/SigLIP, frozen)
       -> Event segmentation
       -> Importance estimation (Complexity x Relevance)
       -> Adaptive allocation (budget-constrained)
       -> Frame selection within events
  -> [Stage 2] MLLM on selected frames only
```

Stage 1 is cheap and sees the whole video sparsely; Stage 2 is expensive and sees only the frames Stage 1 selected.

### Edge cases handled

Both edge cases identified during design are implemented and tested:

1. **K * n_min > T (too many events):** low-importance adjacent events are merged (preserving information), then pruned as a last resort.
2. **n_k > L_k (overflow):** allocations are capped at event capacity, and overflow is redistributed via water-filling by importance priority.
3. **Exact integer sums:** Largest Remainder Method guarantees `sum(n_k) = T` exactly despite rounding.

## When GPU Becomes Available — Transition Guide

All mock components have a `# TODO: REAL` marker where real implementations plug in. Since interfaces are preserved, no downstream code changes.

### Step 1: Replace mock encoder

File: `src/models/mock_encoder.py`

Replace `MockCLIPEncoder` with an `open_clip`-based SigLIP wrapper. Keep the
two public methods: `encode_frames(frames) -> (N, D)` and
`encode_text(text) -> (D,)`.

### Step 2: Replace mock MLLM

File: `src/models/mock_mllm.py`

Replace `MockMLLM` with a Qwen2-VL wrapper using `transformers`. On 2080 Ti
use 4-bit quantization via `bitsandbytes`. Keep the `answer(...)` signature;
the `spec` and `gt_answer` arguments were mock-only cheats and can be
dropped once the real MLLM does the answering.

### Step 3: Replace mock data loader

Add `src/data/videomme.py` that loads VideoMME from Hugging Face and
converts each row into a `VideoQAExample` (defined in `src/utils/types.py`).

### Step 4: Add real auxiliary signals

In `src/methods/event_aware.py`, replace `_mock_auxiliary_signals` with real
RAFT (for motion) and GroundingDINO (for object density). These are expensive,
so cache aggressively (cache key = video_id).

### Step 5: Sanity check before full experiments

Before running the full benchmark, verify that `UniformBaseline` on
VideoMME long subset with Qwen2-VL-7B matches the published number
(~55-60%). If it does not, the evaluation pipeline has a bug that must be
fixed before proceeding — this is the most common source of wasted weeks.

## Design Notes

- All hyperparameters live in `EventAwareConfig` for easy sweeping.
- `BaseMethod` abstracts over uniform / oracle / event-aware, so they share the same evaluation harness.
- `minmax_normalize` is applied per-video before combining complexity components, because scales differ by orders of magnitude in real data (motion 0-100, density 0-50, variance 0-1).
- Largest Remainder Method is the de facto standard for proportional integer allocation (used in parliamentary seat allocation, for instance). It is the cleanest solution to the "softmax x budget -> round -> sum != T" problem.

## Next Steps (when GPU is ready)

1. Follow transition guide above.
2. Run `scripts/pilot_study.py` on real VideoMME — if oracle-uniform gap is >5% accuracy, proceed.
3. Run full benchmark across VideoMME, LongVideoBench, MLVU.
4. Run ablation matrix (complexity components, selection strategies, hyperparameters).
5. Add LongVU and Video-XL as SOTA baselines for head-to-head comparison.
