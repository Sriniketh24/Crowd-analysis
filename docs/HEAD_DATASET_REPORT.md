# Head Dataset Report

> Prepared: 2026-06-02 · Project: Indian railway platform passenger counting
> (head-detection track). Companion to
> [`HEAD_DETECTION_RESEARCH_PLAN.md`](HEAD_DETECTION_RESEARCH_PLAN.md),
> [`REAL_DATA_SOURCE_REPORT.md`](REAL_DATA_SOURCE_REPORT.md), and
> [`../data/DATASET_NOTES.md`](../data/DATASET_NOTES.md).
>
> **Hard rules honored:** real labeled data only · no synthetic data · no fake
> labels · nothing is claimed to be trained. Every count below was produced by
> `scripts/prepare_head_dataset.py` from the actual downloaded files.

---

## Primary Dataset Target

| Field | Value |
|---|---|
| **Name** | RPEE-Heads |
| **Full name** | Railway Platforms and Event Entrances – Heads |
| **Dataset page** | https://ped.fz-juelich.de/da/doku.php?id=rpee_heads |
| **Download (direct)** | https://ped.fz-juelich.de/data/machine_learning/2024_11_Recognition_In_Field_Studies/data/2024rpee_heads_dataset.zip (~1.1 GB) |
| **DOI** | https://doi.org/10.34735/ped.2024.2 |
| **Paper (arXiv)** | https://arxiv.org/abs/2411.18164 |
| **Paper (IEEE)** | https://ieeexplore.ieee.org/document/10973050 |
| **License** | **CC BY-SA 4.0** (attribution + share-alike) |
| **Known size** | 1,886 images · 109,913 annotated heads · 66 video recordings (~56 heads/image) |
| **Annotation type** | Visible **human-head bounding boxes**, one `.txt` per image in **YOLO format** (`class x_center y_center w h`, normalized 0–1) |

**Why it was selected.** RPEE-Heads is the only public benchmark that is
simultaneously (a) **railway-platform-focused**, (b) ships **real head bounding
boxes** (not dot/point annotations), and (c) is **already in YOLO format**. That
makes it the strongest available proxy for Indian railway-platform CCTV
head-counting, with the least conversion risk.

**Railway-platform relevance.** The railway-platform subset is real footage from
**Merkur Spiel-Arena / Messe Nord train station, Düsseldorf, Germany** — overhead,
CCTV-style crowd scenes with the same "heads visible above a dense, body-occluded
crowd" structure as an Indian platform at rush hour. The event-entrance subset
(music-concert entrances + controlled entrance experiments) adds even denser
bottleneck crowds for the worst-case "fully packed platform" condition.

---

## Dataset Actually Prepared

| Question | Answer |
|---|---|
| **RPEE-Heads prepared?** | **YES** |
| If no, why not | n/a — downloaded and converted successfully |
| **Backup dataset used?** | **NO** (not needed; RPEE-Heads downloaded automatically) |
| Backup dataset name | n/a (SCUT-HEAD converter is implemented but not run) |
| **Local dataset path (YOLO)** | `data/head_datasets/yolo/rpee_heads/` |
| **Raw dataset path** | `data/head_datasets/raw/rpee_heads/` (`training/`, `validation/`, `testing/`) |
| **YOLO config path** | `configs/head_dataset.yaml` |
| **Prep script** | `scripts/prepare_head_dataset.py` |

Prepared YOLO layout:

```
data/head_datasets/yolo/rpee_heads/
├── images/train/   (1346 images, symlinks → raw)
├── images/val/     (246 images, symlinks → raw)
├── labels/train/   (1346 .txt, single class 0=head)
└── labels/val/     (246 .txt, single class 0=head)
```

> Images in the YOLO tree are **relative symlinks** into the raw extracted folders,
> so the dataset is not duplicated on disk (YOLO tree ≈ 6.7 MB of labels; raw ≈ 1.1 GB
> of images). Ultralytics follows the symlinks transparently. Pass `--copy` to the
> prep script if real copies are required (e.g. moving the folder to another machine).

---

## Download and Access Notes

**Download source.** Pedestrian Dynamics Data Archive (Forschungszentrum Jülich):
- Dataset page: https://ped.fz-juelich.de/da/doku.php?id=rpee_heads
- Direct archive: `…/2024_11_Recognition_In_Field_Studies/data/2024rpee_heads_dataset.zip`
  (1,163,119,976 bytes, verified `unzip -t` OK)
- DOI: https://doi.org/10.34735/ped.2024.2

**How it was obtained here (automatic, no browser needed).**

```bash
curl -L --fail --retry 3 -o data/head_datasets/raw/rpee_heads/rpee_heads_dataset.zip \
  "https://ped.fz-juelich.de/data/machine_learning/2024_11_Recognition_In_Field_Studies/data/2024rpee_heads_dataset.zip"
unzip -q data/head_datasets/raw/rpee_heads/rpee_heads_dataset.zip \
  -d data/head_datasets/raw/rpee_heads/
python3 scripts/prepare_head_dataset.py --dataset rpee_heads
```

The `.zip` was deleted after extraction to save disk; the extracted
`training/`, `validation/`, `testing/` folders are the retained raw data.
`scripts/prepare_head_dataset.py` also prints these exact manual steps if the raw
files are ever missing — it never fabricates a dataset.

**License / usage notes.** CC BY-SA 4.0: free to use and adapt **with attribution**,
and any redistributed derivative must be shared under the **same license**
(share-alike). Cite the RPEE-Heads paper (arXiv:2411.18164) and dataset DOI in any
publication or redistribution. This is compatible with internal fine-tuning and
evaluation for this project.

---

## Conversion Details

**Original annotation format.** Already YOLO: each image has a sibling `.txt` under
a parallel `labels/` directory, one row per head: `class x_center y_center w h`,
all normalized 0–1, class always `0`. So conversion is **file organization +
validation**, not re-encoding.

**Raw archive structure (as released):**

```
training/train/{images,labels}/     1346 images / 1346 labels
validation/val/{images,labels}/      246 images /  246 labels
testing/test/{images,labels}/        294 images /  294 labels   (held out)
```

**YOLO conversion method** (`scripts/prepare_head_dataset.py --dataset rpee_heads`):
1. Recursively discover every image; pair each with its `.txt` label (sibling or
   parallel `labels/` dir).
2. Infer the split (`train` / `val` / `test`) from the path; the official
   train→train and val→val mapping is used. The official **test** split is **held
   out** (left only in `raw/…/testing/`) for final detector evaluation — pass
   `--test-into-val` to fold it into val instead.
3. **Validate every box**: must have 5 fields, be numeric, lie within [0, 1], have
   positive width/height, and stay inside the frame. Invalid boxes are **skipped
   and logged**.
4. Force **class id `0` = head** (single class).
5. Place images as relative symlinks under `images/<split>/`; write cleaned labels
   under `labels/<split>/`.

**Class mapping.** `0 = head` (single class, enforced on every row).

**Skipped files / invalid labels.** **14 invalid boxes skipped**, all degenerate
**zero-width** boxes (e.g. `0 0.5685 0.1051 0.000000 0.000379`) that would be
unusable for training. 12 were in train, 2 in val. **0 images** had a missing label;
**0 images** were unreadable. No whole files were dropped — only the 14 bad rows.
This exactly reconciles with the official head counts (see below).

---

## Dataset Counts

Produced by `python3 scripts/prepare_head_dataset.py --dataset rpee_heads`
(and re-verified with `--validate-only`):

| Split | Images | Label files | Head boxes (kept) |
|---|---|---|---|
| **train** | **1,346** | 1,346 | **78,594** |
| **val** | **246** | 246 | **16,020** |
| **Prepared total** | **1,592** | 1,592 | **94,614** |
| test (held out in raw) | 294 | 294 | 15,285 |

**Reconciliation with official figures:** official train+val heads = 78,606 + 16,022
= 94,628; kept = 94,614; difference = **14** = the skipped zero-width boxes. ✔

**Skipped / problem files:** invalid boxes = **14** · missing labels = **0** ·
unreadable images = **0** · broken symlinks = **0** (all 1,592 symlinks resolve to
real JPEGs; verified one opens via PIL at 981×552).

---

## Fine-Tuning Readiness

| Question | Answer |
|---|---|
| **Ready for YOLO training?** | **YES** — real images + real YOLO labels exist on disk |
| **Dataset YAML path** | `configs/head_dataset.yaml` |
| **Base weights present** | `yolo11n.pt`, `yolo11s.pt` (repo root) |
| **Ultralytics present** | yes (8.4.58, `yolo` CLI available) |

**Training command (head detector, single class `head`):**

```bash
# laptop / demo model (YOLO11n)
yolo detect train data=configs/head_dataset.yaml model=yolo11n.pt \
    epochs=80 imgsz=960 project=models/fine_tuned name=head_rpee_n

# accuracy variant (GPU recommended; larger imgsz helps tiny/distant heads)
yolo detect train data=configs/head_dataset.yaml model=yolo11s.pt \
    epochs=100 imgsz=1280 project=models/fine_tuned name=head_rpee_s
```

> `imgsz` is intentionally larger (960–1280) than the body pipeline's 640 because
> heads are tiny and benefit from higher input resolution.
> **Note:** `scripts/train_yolo_indian_platform.py` is a *separate* guardrailed
> entrypoint hard-wired to `data/training_dataset/` (the Indian-platform body track);
> it is not used for the head model. Head training uses the Ultralytics CLI above
> against `configs/head_dataset.yaml`. This agent did **not** run any training.

**What is missing for the model itself:** nothing blocks training — but the held-out
**test** split (`data/head_datasets/raw/rpee_heads/testing/test/`) should be used for
final detector metrics (mAP@0.5, mAP@0.5:0.95, precision/recall, AP-by-head-size).

---

## Limitations

- **RPEE-Heads is railway-platform-relevant but NOT Indian-railway-specific.** The
  railway subset is a German station (Düsseldorf); the rest is concert/event
  entrances. Clothing, luggage (large bags/trolleys), headwear, station geometry,
  camera optics, and lighting differ from Indian platforms → a real **domain gap**.
  Strong RPEE-Heads metrics do **not** prove Indian-platform accuracy.
- **Backup datasets** (CrowdHuman, SCUT-HEAD, Brainwash) match railway platforms even
  less well — CrowdHuman is web images, SCUT-HEAD is classrooms, Brainwash is a single
  café webcam (and its original host was removed over consent concerns). They are
  occlusion/variety supplements only, not platform validation data.
- **Indian Railway CCTV footage is still needed for final validation.** No public
  labeled Indian railway-platform head dataset exists. The real Indian clips in this
  repo (Mumbai local, Delhi Metro) are **unlabeled** and cannot be used for supervised
  training until annotated. Until labeled Indian frames exist, all head-detection
  accuracy numbers are **proxy numbers** — **not production-ready**, and must not be
  presented as such.
- **Head false positives / tiny-head ID switches** remain open risks (round objects
  mistaken for heads; distant few-pixel heads causing tracker ID swaps), to be measured
  during training/evaluation, not assumed away.

---

## Sources

- RPEE-Heads dataset page: https://ped.fz-juelich.de/da/doku.php?id=rpee_heads
- RPEE-Heads DOI: https://doi.org/10.34735/ped.2024.2
- RPEE-Heads paper: https://arxiv.org/abs/2411.18164 · https://ieeexplore.ieee.org/document/10973050
- CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/
- Backups: CrowdHuman https://www.crowdhuman.org/ · SCUT-HEAD https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release
