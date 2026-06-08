# Head-Recall Improvement Plan — Tiny / Occluded FAR Heads (Fix 5)

**Goal:** Raise the head detector's *recall on small-scale and occluded "far"
passengers* (distant heads on a platform CCTV view) **without retraining in this
repo** and **without overwriting** `models/fine_tuned/head_detector/weights/best.pt`.

This document is a runnable plan: dataset sampling/augmentation, a small-object
training config, and a held-out far-region evaluation. Training is assumed to run
in **Google Colab (GPU)**; nothing here trains weights locally or commits new
weights into this repo.

> Ownership note: this plan touches only `scripts/prepare_head_dataset.py` (read),
> `scripts/train_head_detector.py` (read), `scripts/evaluate_head_detector.py`
> (read) and adds one new read-only helper `scripts/eval_far_heads.py`. It does
> NOT modify `hybrid_detector.py`, `run_video_demo.py`, `config.py`, or
> `configs/*.json` (owned by other agents).

---

## 1. Why recall on far heads is currently weak (measured)

Measured on the prepared RPEE-Heads YOLO data and the **held-out official test
split** (`data/head_datasets/raw/rpee_heads/testing/test`, 294 images, 15,285
head boxes):

| Observation | Value |
|---|---|
| Train head boxes scanned | 34,172 |
| Median head box @ imgsz 640 | ~11.5 × 19.7 px (area ≈ 229 px²) |
| Train heads with area < 32×32 (COCO "small") | **95.3%** |
| **Test** heads with area < 16×16 (very tiny / far) | **58.1%** |
| **Test** heads with area < 24×24 | **87.1%** |
| Test p10 head side length | ~6.9 px |

The previous Colab run (recorded in
`models/fine_tuned/head_detector/args.yaml` and `results.csv`) used settings that
are *not* tuned for objects this small:

| Setting (prior run) | Value | Problem for tiny heads |
|---|---|---|
| `model` | `yolo11n.pt` | Smallest backbone; weak feature capacity for 7–20 px heads |
| `imgsz` | 640 | A 7 px head at 640 is sub-pixel after downsampling — effectively invisible to the P3/P4/P5 strides |
| `epochs` | 25 | Under-trained; val recall still climbing at epoch 25 |
| `scale` | 0.5 | Zoom-out augmentation shrinks already-tiny heads further |
| `mosaic` | 1.0 (closed last 10) | Helps, but combined with downscale starves small heads |
| `copy_paste` | 0.0 | No extra small/occluded head instances synthesized |
| `multi_scale` | 0.0 | Single-scale training; no robustness to scale shift |
| `max_det` (val) | 300 | Crowded platform frames exceed 300 heads → recall capped |

Final prior metrics (epoch 25): precision ≈ 0.866, **recall ≈ 0.723**, mAP50 ≈
0.794, mAP50-95 ≈ 0.431. Recall is the weakest axis and is dominated by missed
tiny heads.

**Conclusion:** the single biggest lever is **inference + training resolution**,
followed by **small-object-preserving augmentation**, a **slightly larger
backbone**, **more epochs**, and **raising `max_det`** for crowds.

---

## 2. (a) Sampling / augmentation — add more tiny + occluded far-head crops

All steps below are run in Colab against the RPEE-Heads data already prepared by
`scripts/prepare_head_dataset.py`. No labels are fabricated.

### 2.1 Keep RPEE-Heads as the primary, on-domain dataset
RPEE-Heads is railway-platform/entrance overhead footage — exactly the far-head
regime. It is already prepared:
```
data/head_datasets/yolo/rpee_heads/images/{train,val}
data/head_datasets/yolo/rpee_heads/labels/{train,val}   (1346 train / 246 val)
```

### 2.2 Add CrowdHuman heads as a *supplemental occlusion* source
CrowdHuman's `hbox` (head box) annotations are dense and heavily occluded —
ideal for the "occluded" half of Fix 5. The converter already exists:
```bash
python3 scripts/prepare_head_dataset.py --dataset crowdhuman_heads \
    --raw-dir data/head_datasets/raw/crowdhuman \
    --out-dir data/head_datasets/yolo/crowdhuman_heads
```
Then train with the mixed config `configs/head_dataset_mixed.example.yaml`
(already in repo). This is optional but recommended for occlusion robustness.
Verify license terms before use (academic/research).

### 2.3 Augmentation that *grows* the tiny/occluded population
Set these in the Ultralytics training call (Section 3). The key idea: stop
shrinking heads, and synthesize more small + occluded instances.

| Aug | Prior | New | Rationale |
|---|---|---|---|
| `mosaic` | 1.0 | **1.0** | 4-image tiling packs many small heads per view (keep on) |
| `close_mosaic` | 10 | **15** | Disable mosaic only for the final 15 epochs to clean up localization |
| `copy_paste` | 0.0 | **0.3** | Paste real head instances into more frames → more tiny/occluded positives |
| `copy_paste_mode` | flip | **flip** | Keep |
| `scale` | 0.5 | **0.2** | Reduce aggressive zoom-out so small heads aren't downscaled into oblivion |
| `multi_scale` | 0.0 | **True** | Train across ±50% scales for scale robustness on far heads |
| `mixup` | 0.0 | **0.1** | Mild; adds occlusion-like blending |
| `fliplr` | 0.5 | **0.5** | Keep |
| `hsv_h/s/v` | 0.015/0.7/0.4 | same | Platform lighting varies; keep |
| `degrees`/`shear`/`perspective` | 0 | **0** | Overhead heads are near upright; geometric warps hurt tiny boxes |
| `erasing` | 0.4 | **0.4** | Random erasing simulates occlusion (keep) |

> Note on `copy_paste`: it duplicates *labeled* instances already in the data —
> it does not invent new annotations, so the "no synthetic labels" rule holds.

---

## 3. (b) Training config tuned for small objects

Run in **Colab (GPU)**. The single most important change is **`imgsz 1280`**:
a 10 px head at 1280 has ~4× the linear resolution it had at 640.

| Param | Prior | New | Why |
|---|---|---|---|
| `model` | `yolo11n.pt` | **`yolo11s.pt`** | More capacity for tiny features; still real-time on GPU |
| `imgsz` | 640 | **1280** | Tiny heads need resolution; biggest recall lever |
| `epochs` | 25 | **100** | Recall was still rising at 25; small objects need longer |
| `batch` | 32 | **16** (or `-1` auto) | imgsz 1280 uses ~4× memory; reduce to fit GPU |
| `patience` | 100 | **30** | Early-stop on val mAP plateau |
| `cos_lr` | false | **true** | Smoother late-stage convergence |
| `close_mosaic` | 10 | **15** | See §2.3 |
| `multi_scale` | 0.0 | **true** | See §2.3 |
| `copy_paste` | 0.0 | **0.3** | See §2.3 |
| `scale` | 0.5 | **0.2** | See §2.3 |
| `mixup` | 0.0 | **0.1** | See §2.3 |
| `lr0` | 0.01 | 0.01 | Keep (auto optimizer) |

`scripts/train_head_detector.py` already exposes `--model --epochs --imgsz
--batch --device`. The augmentation params (`copy_paste`, `multi_scale`,
`scale`, `mixup`, `close_mosaic`, `cos_lr`, `patience`) are **not** CLI flags
there. Two options that respect file ownership:

- **Preferred (no code change):** call Ultralytics directly in the Colab cell
  (snippet in §5). This passes every small-object param explicitly.
- **Alternative:** if the owning agent later wants these as CLI flags, they can
  be added to `scripts/train_head_detector.py`. This plan does not edit it.

**Critical:** train into a **Colab/run directory, not** `models/fine_tuned/...`.
Use `project=/content/runs name=head_detector_far` so the committed `best.pt` is
never overwritten. Bring the candidate `best.pt` back only after it beats the
baseline on the far-region eval (§4), and let the owning agent decide promotion.

---

## 4. (c) Held-out FAR-region evaluation

The official RPEE-Heads **test** split is held out by
`scripts/prepare_head_dataset.py` (left under
`data/head_datasets/raw/rpee_heads/testing/test`, 294 images). It was never used
for training/val, so it is the correct far-region test set.

`scripts/evaluate_head_detector.py` only does a 1-image sanity check + an
annotated video pass — it does not compute size-bucketed recall. This plan adds
a read-only helper: **`scripts/eval_far_heads.py`**, which:

- buckets every GT head by pixel area at the eval `imgsz`
  (`tiny <16px`, `small 16–32`, `medium 32–96`, `large ≥96`),
- greedy IoU-matches predictions to GT,
- reports **recall per bucket and overall**, surfacing the
  **FAR/tiny-head recall** as the key metric,
- never trains and never writes into `models/fine_tuned/...`.

### Baseline (committed model, current settings)
```bash
python3 scripts/eval_far_heads.py \
  --model models/fine_tuned/head_detector/weights/best.pt \
  --imgsz 640 --conf 0.10 --iou 0.5 --max-det 1000 \
  --json-out data/outputs/far_eval_baseline.json
```

### Baseline at high-res inference (free recall, no retrain)
```bash
python3 scripts/eval_far_heads.py \
  --model models/fine_tuned/head_detector/weights/best.pt \
  --imgsz 1280 --conf 0.10 --iou 0.5 --max-det 1000 \
  --json-out data/outputs/far_eval_baseline_1280.json
```
> This often improves far-head recall on its own and is a zero-cost win to try
> first. Note: the *deployed* detector resolution is owned by `config.py` /
> `hybrid_detector.py` — coordinate any inference-imgsz change with that agent.

### Candidate (newly Colab-trained model, brought back outside the repo)
```bash
python3 scripts/eval_far_heads.py \
  --model /path/to/colab/head_detector_far/weights/best.pt \
  --imgsz 1280 --conf 0.10 --iou 0.5 --max-det 1000 \
  --json-out data/outputs/far_eval_candidate.json
```

**Acceptance criteria:** candidate must improve **tiny-bucket recall** and
**overall recall** vs. baseline on this held-out split, without regressing
`large`-bucket recall by more than 2 points. Only then propose promotion (by the
owning agent) of the new `best.pt`.

---

## 5. Runnable Colab / CLI command sequence

```bash
# ---- 0. Environment (Colab GPU) ----
pip install -r requirements.txt        # ultralytics, opencv, etc.

# ---- 1. Prepare data (primary; no download performed by the script) ----
python3 scripts/prepare_head_dataset.py --dataset rpee_heads
python3 scripts/prepare_head_dataset.py --dataset rpee_heads --validate-only
# Optional occlusion supplement (requires local CrowdHuman raw files + license check):
# python3 scripts/prepare_head_dataset.py --dataset crowdhuman_heads \
#     --raw-dir data/head_datasets/raw/crowdhuman \
#     --out-dir data/head_datasets/yolo/crowdhuman_heads

# ---- 2. Baseline far-region eval BEFORE any change (zero-cost) ----
python3 scripts/eval_far_heads.py \
  --model models/fine_tuned/head_detector/weights/best.pt \
  --imgsz 640  --conf 0.10 --json-out data/outputs/far_eval_baseline.json
python3 scripts/eval_far_heads.py \
  --model models/fine_tuned/head_detector/weights/best.pt \
  --imgsz 1280 --conf 0.10 --json-out data/outputs/far_eval_baseline_1280.json
```

```python
# ---- 3. Small-object training (Colab GPU cell) ----
# Trains into /content/runs, NOT models/fine_tuned -> committed best.pt is safe.
from ultralytics import YOLO

model = YOLO("yolo11s.pt")          # larger backbone than prior yolo11n
model.train(
    data="configs/head_dataset.yaml",   # or head_dataset_mixed.example.yaml if CrowdHuman prepared
    imgsz=1280,                         # KEY lever for tiny heads
    epochs=100,
    batch=16,                           # or -1 for auto; lower if OOM at 1280
    patience=30,
    cos_lr=True,
    multi_scale=True,
    mosaic=1.0,
    close_mosaic=15,
    copy_paste=0.3,
    copy_paste_mode="flip",
    scale=0.2,                          # less zoom-out so tiny heads survive
    mixup=0.1,
    degrees=0.0, shear=0.0, perspective=0.0,
    project="/content/runs",            # NOT models/fine_tuned
    name="head_detector_far",
    exist_ok=True,
    device=0,
    workers=8,
)
# Candidate weights: /content/runs/head_detector_far/weights/best.pt
```

```bash
# ---- 4. Candidate far-region eval and A/B compare ----
python3 scripts/eval_far_heads.py \
  --model /content/runs/head_detector_far/weights/best.pt \
  --imgsz 1280 --conf 0.10 --json-out data/outputs/far_eval_candidate.json

# Compare the three JSONs (baseline 640, baseline 1280, candidate 1280):
python3 - <<'PY'
import json, pathlib
for tag in ("far_eval_baseline","far_eval_baseline_1280","far_eval_candidate"):
    p = pathlib.Path(f"data/outputs/{tag}.json")
    if p.exists():
        d = json.loads(p.read_text())
        t = d["buckets"]["tiny"]
        print(f"{tag:28s} tiny_recall={t['recall']:.3f}  overall={d['overall_recall']:.3f}")
PY
```

```bash
# ---- 5. (Optional) sanity demo with the existing evaluator (unchanged) ----
python3 scripts/evaluate_head_detector.py \
  --model /content/runs/head_detector_far/weights/best.pt \
  --imgsz 1280 --conf 0.10
```

Do **not** copy the candidate over
`models/fine_tuned/head_detector/weights/best.pt` from this workstream. Promotion
is the owning agent's call once the acceptance criteria in §4 are met.

---

## 6. Expected impact & risks

- **Highest-impact, lowest-cost:** evaluate the *current* model at `imgsz 1280`
  (§4) — frequently recovers far-head recall with no retraining.
- **Retraining at 1280 + small-object aug:** expected to lift tiny-bucket recall
  materially (the prior run never "saw" heads at adequate resolution).
- **Risks:** (1) 1280 inference is slower — the deployed resolution is owned by
  `config.py`/`hybrid_detector.py`, so coordinate before changing runtime;
  (2) lower `conf` raises recall but can add false positives — tune on the
  held-out split, not on the demo video; (3) `batch 16 @ 1280` may still OOM on
  small GPUs — drop to `batch=-1` (auto) or `8`.

---

## 7. Files in this workstream

| File | Status | Role |
|---|---|---|
| `docs/HEAD_RECALL_IMPROVEMENT_PLAN.md` | new (this doc) | The plan |
| `scripts/eval_far_heads.py` | new | Read-only size-bucketed far-region recall eval |
| `scripts/prepare_head_dataset.py` | unchanged (read) | Dataset prep (RPEE primary) |
| `scripts/train_head_detector.py` | unchanged (read) | Training entry (Colab call preferred for new aug params) |
| `scripts/evaluate_head_detector.py` | unchanged (read) | Existing sanity/video evaluator |
| `models/fine_tuned/.../best.pt` | untouched | Committed weights — never overwritten here |
