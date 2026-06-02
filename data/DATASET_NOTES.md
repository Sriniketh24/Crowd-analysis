# Dataset Notes

Working notes for data used in the Indian railway platform passenger-counting project.
See the full report in [`docs/REAL_DATA_SOURCE_REPORT.md`](../docs/REAL_DATA_SOURCE_REPORT.md).

---

## ⛔ Hard rules (do not break)

- **NO synthetic / generated / fabricated video.** Use only real footage or real public datasets.
- **NO fake datasets.** Do not invent annotations.
- **An unlabeled video is NOT a labeled training set.** Unlabeled clips are for demo/testing or
  as raw material to label — never claim they can be used directly for supervised fine-tuning.
- **Avoid synthetic datasets** (e.g. GCC / GTA5 crowd counting, MOTSynth) — they are CGI/game-rendered.
- Respect each source's **license / terms of service**. Do not redistribute stock clips on other
  stock platforms or sell them unaltered.

---

## Head-detection dataset (for YOLO fine-tuning)

> Full details in [`../docs/HEAD_DATASET_REPORT.md`](../docs/HEAD_DATASET_REPORT.md) and
> the research plan [`../docs/HEAD_DETECTION_RESEARCH_PLAN.md`](../docs/HEAD_DETECTION_RESEARCH_PLAN.md).

**Primary head dataset: RPEE-Heads** (Railway Platforms and Event Entrances – Heads).
This is the chosen dataset for head-detection fine-tuning because it is the only public
benchmark that is **railway-platform-focused**, ships **real head bounding boxes**, and is
**already in YOLO annotation format**.

| Field | Value |
|------|-------|
| Name / full name | RPEE-Heads / Railway Platforms and Event Entrances – Heads |
| Size | 1,886 images · 109,913 head boxes · 66 videos |
| Official split | train 1,346 / val 246 / test 294 images |
| Annotation | YOLO `.txt` per image: `class x_center y_center w h` (normalized 0–1) |
| Download | https://ped.fz-juelich.de/da/doku.php?id=rpee_heads → `2024rpee_heads_dataset.zip` (~1.1 GB) |
| DOI | https://doi.org/10.34735/ped.2024.2 |
| Paper | https://arxiv.org/abs/2411.18164 · https://ieeexplore.ieee.org/document/10973050 |
| License | **CC BY-SA 4.0** (attribution + share-alike) |
| Raw path | `data/head_datasets/raw/rpee_heads/` |
| YOLO path | `data/head_datasets/yolo/rpee_heads/` (`images/{train,val}`, `labels/{train,val}`) |
| Prep script | `scripts/prepare_head_dataset.py --dataset rpee_heads` |
| Config YAML | `configs/head_dataset.yaml` (single class `0 = head`) |

**Backup head datasets considered** (only used if RPEE-Heads is unavailable; RPEE-Heads
remains preferred):

| Dataset | Head boxes? | Format | Note |
|---------|-------------|--------|------|
| CrowdHuman | Yes (head + body) | ODGT/JSON | best for occlusion hardening; web images, not CCTV |
| SCUT-HEAD | Yes | Pascal-VOC XML | overhead classroom/surveillance heads; converter implemented in prep script |
| Brainwash | Yes | idl/text | true fixed-camera surveillance, **original host removed over consent concerns** — restricted |

> Only RPEE-Heads is prepared by default. The backup converters run **only** if their raw
> files are physically present — the script never fabricates labels.

---

## What is in this repo (all REAL footage)

| Path | What it is | Real? | Indian railway? | Labeled? |
|------|-----------|-------|-----------------|----------|
| `data/input_videos/sample.mp4` | 20 s demo trim of the Mumbai local platform clip (1080p, 30fps) | ✅ | ✅ Mumbai suburban | ❌ |
| `data/indian_railway_videos/pexels_crowded_train_station_6023186.mp4` | Full 29.4 s Mumbai local platform master | ✅ | ✅ Mumbai suburban | ❌ |
| `data/indian_railway_videos/pexels_people_in_train_station_12899783.mp4` | Delhi Metro concourse (portrait, 27.8 s) — held-out test | ✅ | ✅ Delhi Metro | ❌ |
| `data/indian_railway_videos/station_concourse_overhead_grayscale_PRIOR-sample.mp4` | Prior repo `sample.mp4`, overhead pedestrian concourse (non-Indian; source not documented) | ✅ | ❌ | ❌ |
| `data/input_videos/vtest_backup.avi` | OpenCV `vtest.avi` UK campus CCTV (~6 people, 768×576) | ✅ | ❌ | ❌ |
| `data/sample_frames/indian_railway_platform_*.jpg` | 8 frames extracted from `sample.mp4` | ✅ | ✅ | ❌ |
| `data/sample_frames/_previous_concourse_frames/` | Earlier frames from the prior concourse clip (preserved) | ✅ | ❌ | ❌ |

> All videos here are **unlabeled**. None can be used for supervised fine-tuning until annotated.

---

## Source links

**Real Indian / station videos (unlabeled — demo/testing & labeling material):**
- Mumbai local platform: https://www.pexels.com/video/crowded-train-station-6023186/
- Delhi Metro concourse: https://www.pexels.com/video/people-in-train-station-12899783/
- Pexels "indian railway" search: https://www.pexels.com/search/videos/indian%20railway/
- Pexels "train station" search: https://www.pexels.com/search/videos/train%20station/
- Pixabay "indian train": https://pixabay.com/videos/search/indian%20train/
- Pixabay "crowded train station": https://pixabay.com/videos/search/crowded%20train%20station/

**Real labeled datasets (for fine-tuning — NOT Indian, but real):**
- RPEE-Heads (railway platform head bboxes, CC BY-SA 4.0): https://arxiv.org/abs/2411.18164 · https://doi.org/10.34735/ped.2024.2
- CrowdHuman (full-body/visible/head bboxes): https://www.crowdhuman.org/ · https://huggingface.co/datasets/sshao0516/CrowdHuman
- MOT20 (crowded pedestrian bbox + IDs): https://motchallenge.net/data/MOT20/
- WiderPerson (dense pedestrian bbox): http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/
- JHU-CROWD++ (head dots + approx bbox): http://www.crowd-counting.com/
- ShanghaiTech (head points): https://github.com/desenzhou/ShanghaiTechDataset
- Roboflow railway crowd detection (small, YOLO-ready): https://universe.roboflow.com/muk-1bbp8/railway-crowd-detection

---

## Usage rules per source

- **Pexels** (clips 6023186, 12899783): free to use & modify, commercial OK, no attribution
  required. Don't resell unaltered or re-upload to other stock sites. License: https://www.pexels.com/license/
- **Pixabay**: Pixabay Content License — free, no attribution required; don't redistribute as stock.
- **RPEE-Heads**: CC BY-SA 4.0 — attribute and share-alike if redistributed.
- **CrowdHuman / MOT20 / WiderPerson / JHU-CROWD++ / ShanghaiTech**: academic/research use;
  several are **non-commercial**. **Verify the exact license on each official page before any
  commercial use or redistribution.**
- **Roboflow Universe**: usually CC BY 4.0 — confirm on the dataset page.

---

## Labeled vs unlabeled (at a glance)

- **Labeled (real):** CrowdHuman, MOT20, WiderPerson, JHU-CROWD++, ShanghaiTech, RPEE-Heads,
  metro/boarding CCTV datasets, Roboflow railway-crowd. → usable for fine-tuning after format
  conversion to YOLO.
- **Unlabeled (real):** every video file currently in this repo (Mumbai, Delhi, prior concourse,
  vtest). → demo/testing only, or annotate them first.

---

## Annotation workflow (to make Indian footage trainable)

1. Extract frames (~0.5–1 s spacing) from the real Indian clips.
2. Pre-label `person` boxes with the current YOLO model, then human-correct.
3. Export **YOLO format** (`images/`, `labels/*.txt`, single class `person`).
4. Tools: **Roboflow** (easiest, cloud, YOLO export), **CVAT** (best for video interpolation),
   **Label Studio** (ML-assisted). Keep a held-out clip for evaluation.
