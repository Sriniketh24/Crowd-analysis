# Real Data Source Report

> **Hard rule honored:** Everything in this report is **real footage** or a **real public dataset**.
> No synthetic, generated, or fabricated video was created or used. No unlabeled video is
> claimed to be a labeled training dataset.
>
> Prepared: 2026-06-02 · Project: Indian railway platform passenger counting

---

## Best Demo Video Selected

| Field | Value |
|-------|-------|
| **File path (demo clip)** | `data/input_videos/sample.mp4` |
| **File path (full master)** | `data/indian_railway_videos/pexels_crowded_train_station_6023186.mp4` |
| **Source page URL** | https://www.pexels.com/video/crowded-train-station-6023186/ |
| **Direct file URL** | https://videos.pexels.com/video-files/6023186/6023186-hd_1920_1080_30fps.mp4 |
| **Creator** | Rahul Vhatkar (Pexels) |
| **Why selected** | It is a **genuine Indian railway platform** — a Mumbai suburban ("local") train station. The purple/grey Mumbai EMU livery, train number `5257`, the `022` (Mumbai) phone codes on the platform hoardings, and the platform/awning architecture all confirm it is Indian. It shows **real passengers walking on the platform** with a train at the platform — exactly the scene this project targets. It is 1080p, clear, downloadable under a permissive license, and the baseline detector already finds people in it (see below). This is the closest match to "live Indian railway platform CCTV" among freely downloadable footage. |
| **Indian railway-specific?** | **YES** — Mumbai suburban railway platform (Indian Railways / Mumbai local). |
| **License / usage** | **Pexels License** — free to use, modification allowed, commercial use allowed, attribution not required. Prohibited: reselling unaltered copies, redistributing on other stock platforms, trademark use, implying endorsement. Our use (model testing + creating labeled training frames) is permitted. |
| **Duration** | Master clip: **29.43 s** · `sample.mp4` demo trim: **20.00 s** (first 20 s, re-encoded H.264 / yuv420p / faststart for pipeline compatibility) |
| **Resolution** | **1920 × 1080** |
| **FPS** | **30 fps** (detected; `r_frame_rate = 30/1`) |
| **Baseline detector check** | Legacy baseline check: `yolov8n.pt`, conf 0.25, class=person → **11–13 persons/frame** on the first three extracted frames. Current demo config uses `yolo11n.pt` by default, with `yolo11s.pt` available for higher-accuracy evaluation. Pretrained models catch foreground passengers but can **miss small/distant passengers near the train** → motivates fine-tuning (see Fine-Tuning Feasibility). |

> ⚠️ **Known limitation of this clip:** consecutive frames show larger-than-1-frame crowd
> displacement, so the footage is **mildly sped-up / timelapse-style**. It is excellent for
> **per-frame detection and density/occupancy** demos, but **sub-optimal for tracking-based
> line-crossing counts** (trackers may switch IDs). For accurate in/out line counting, prefer
> a normal-speed clip (e.g. the Delhi Metro held-out clip, or new normal-speed footage).

---

## Downloaded / Available Videos

| # | Name | URL | Local path | Real/Synthetic | Indian railway-specific | Labels available | License / usage note | Usefulness (1–5) |
|---|------|-----|-----------|----------------|-------------------------|------------------|----------------------|------------------|
| 1 | Crowded Train Station (Mumbai local platform) | https://www.pexels.com/video/crowded-train-station-6023186/ | `data/indian_railway_videos/pexels_crowded_train_station_6023186.mp4` | **Real** | **YES** (Mumbai suburban) | No | Pexels License (free, commercial OK) | **5** |
| 2 | People in Train Station (Delhi Metro concourse) | https://www.pexels.com/video/people-in-train-station-12899783/ | `data/indian_railway_videos/pexels_people_in_train_station_12899783.mp4` | **Real** | **YES** (Delhi Metro, "Towards Noida Electronic City") | No | Pexels License (free, commercial OK) | **4** (portrait 1080×1920; good held-out test) |
| 3 | Station concourse, overhead grayscale (prior `sample.mp4`) | (was already in repo; provenance not documented) | `data/indian_railway_videos/station_concourse_overhead_grayscale_PRIOR-sample.mp4` | **Real** | No (looks like a Western transit hall / mall atrium) | No | **Unknown** — verify source before any redistribution | **3** (good overhead pedestrian demo, but not Indian) |
| 4 | OpenCV `vtest.avi` (UK campus CCTV) | https://github.com/opencv/opencv/blob/master/samples/data/vtest.avi | `data/input_videos/vtest_backup.avi` | **Real** | No | No | OpenCV sample (Apache-2.0 / BSD) | **2** (sparse ~6 people, 768×576; pipeline smoke test only) |
| 5 | Indian Railways (freight train, scenic) | https://www.pexels.com/video/indian-railways-video-18626169/ | not downloaded | **Real** | Partly (Indian freight train, **no platform/passengers**) | No | Pexels License | **1** |
| 6 | Indian railway (countryside POV from moving train) | https://www.pexels.com/video/indian-railway-28212383/ | not downloaded | **Real** | Partly (no platform/passengers) | No | Pexels License | **1** |

**More downloadable sources (not pulled, browse/pick as needed):**
- Pexels — "indian railway" videos: https://www.pexels.com/search/videos/indian%20railway/
- Pexels — "train station" videos: https://www.pexels.com/search/videos/train%20station/
- Pixabay — "indian train" (Pixabay Content License, no attribution): https://pixabay.com/videos/search/indian%20train/
- Pixabay — "crowded train station": https://pixabay.com/videos/search/crowded%20train%20station/
- Pixabay — "railway station": https://pixabay.com/videos/search/railway%20station/

---

## Labeled Datasets Found

> None of these are **Indian** railway platforms, but several are **railway/metro-platform-specific**
> and all are **real**. Use them to fine-tune crowded / small-person / head detection, then
> domain-adapt to Indian platforms with locally annotated frames.

| Dataset | URL | Labels available | Classes / annotation | Relevance to Indian railway platforms | Difficulty to use | License / usage note |
|---------|-----|------------------|----------------------|----------------------------------------|-------------------|----------------------|
| **RPEE-Heads** (Railway Platforms & Event Entrances – Heads) | Paper: https://arxiv.org/abs/2411.18164 · Data DOI: https://doi.org/10.34735/ped.2024.2 | **Yes** — 109,913 head bounding boxes, 1,886 images, 66 videos (~56 heads/img) | Head bounding boxes | **High** — actual **railway platforms** + dense crowds; head-based counting transfers well to crowded Indian platforms. Not Indian; heads (not full body). | Medium (convert head-bbox → YOLO; head detector ≠ body detector) | **CC BY-SA 4.0** (per paper page) |
| **CrowdHuman** | https://www.crowdhuman.org/ · HF mirror: https://huggingface.co/datasets/sshao0516/CrowdHuman | **Yes** — ~470k instances; 15,000 train + 4,370 val images, ~23 persons/img | Full-body bbox, visible-region bbox, head bbox | **High** for dense/occluded **person** detection (the baseline's main weakness). Not railway/Indian. | Medium (~3–4 GB; convert ODGT → YOLO; ignore "head"/"visible" if doing body-only) | Academic/research use — verify terms on site |
| **MOT20** (MOTChallenge) | https://motchallenge.net/data/MOT20/ · Paper: https://arxiv.org/abs/2003.09003 | **Yes** — person bboxes + track IDs; 8 very crowded sequences (up to 246 ppl/frame) | Pedestrian bbox + identity | **High** — very dense, **elevated CCTV-style** viewpoints incl. indoor crowded scenes resembling stations. Not Indian. | Medium–Hard (MOT format → YOLO; per-frame split) | CC BY-NC-SA (non-commercial research) — verify |
| **WiderPerson** | http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/ · Paper: https://arxiv.org/abs/1909.12118 | **Yes** — 13,382 images, ~9,000 annotation files | Bbox, 5 classes: pedestrian / rider / partially-visible / crowd / ignore | Medium–High — dense pedestrians "in the wild". Not railway/Indian. | Medium (Google/Baidu Drive; convert `[class,x1,y1,x2,y2]` → YOLO) | Academic/research — verify with authors |
| **JHU-CROWD++** | http://www.crowd-counting.com/ · Paper: https://arxiv.org/pdf/2004.03597 | **Yes** — 4,372 images, 1.51M annotations | Head dots + **approximate** head bboxes + image-level scene labels | Medium — unconstrained crowds (some transit/station scenes). Not Indian. | Medium (point→density for counting; approx bbox usable for head det) | Non-commercial research — verify |
| **ShanghaiTech** (Part A / Part B) | Paper / data: https://github.com/desenzhou/ShanghaiTechDataset | **Yes** — head **point** (dot) annotations | Head points (density estimation, **not** bbox detection) | Medium — Part B = street/pedestrian scenes. Not railway/Indian. | Medium (points → density maps; not directly a YOLO bbox dataset) | Research use |
| **Roboflow "railway crowd detection"** (user: muk) | https://universe.roboflow.com/muk-1bbp8/railway-crowd-detection | **Yes** — small (~tens of images) | Person bbox, **YOLO export ready** | Medium — railway-labeled & YOLO-ready, but very small; likely not Indian | **Easy** (one-click YOLO export from Roboflow) | Roboflow Universe (often CC BY 4.0) — **verify on page** |
| Metro platform dataset (surveillance) | via paper https://pmc.ncbi.nlm.nih.gov/articles/PMC7662571/ | Yes — 627 images, 9,243 annotated heads | Head annotations | Medium–High — metro platform passenger flow. Not Indian. | Medium | Per source paper — verify |
| Metropolitan train boarding (CCTV) | https://pmc.ncbi.nlm.nih.gov/articles/PMC7662571/ | Partial — 348 CCTV sequences, subset head-tracked | Head locate + track | Medium — people boarding/alighting a train carriage (CCTV). Not Indian. | Medium–Hard | Per source paper — verify |

> ❌ **Excluded on purpose (synthetic — violates the hard rule):** GCC (GTA5 Crowd Counting),
> MOTSynth, and any game/CGI-generated crowd data. Do **not** use these.

---

## Unlabeled Videos Found

> These are **real** but have **no annotations**. They are usable for **demo / testing /
> qualitative evaluation only**, and as raw material to **manually label** for fine-tuning.
> They cannot be used directly for supervised training.

| Video name | URL | Why useful | Demo/testing only? |
|------------|-----|-----------|--------------------|
| Crowded Train Station — Mumbai local (6023186) | https://www.pexels.com/video/crowded-train-station-6023186/ | Real **Indian platform** with passengers; primary demo & labeling source | **Yes** (no labels) |
| People in Train Station — Delhi Metro (12899783) | https://www.pexels.com/video/people-in-train-station-12899783/ | Real **Indian metro** concourse; held-out test from a different city | **Yes** (no labels) |
| Prior repo concourse clip (grayscale overhead) | `data/indian_railway_videos/station_concourse_overhead_grayscale_PRIOR-sample.mp4` | Real dense overhead pedestrian flow; good non-Indian stress test | **Yes** (no labels) |
| Pexels "indian railway" / "train station" collections | https://www.pexels.com/search/videos/indian%20railway/ | Many more free Indian/station clips to pick from | **Yes** |
| Pixabay "indian train" / "crowded train station" | https://pixabay.com/videos/search/indian%20train/ | Free (no-attribution) clips for more test scenes | **Yes** |
| OpenCV `vtest.avi` | `data/input_videos/vtest_backup.avi` | Tiny, fast pipeline smoke test | **Yes** |

---

## Fine-Tuning Feasibility

**Did we find a labeled *Indian railway platform* dataset?**
**No.** There is no public, ready-made, labeled **Indian** railway-platform passenger dataset.
The real Indian footage we obtained (Mumbai local, Delhi Metro) is **unlabeled**.

**What labeled real data *does* exist (and is usable)?**
- **Railway/metro-platform-specific, real, labeled:** RPEE-Heads (head bboxes, CC BY-SA 4.0),
  the metro-platform and train-boarding datasets — all **head**-annotated, **not Indian**.
- **General crowded-person detection, real, labeled:** CrowdHuman, MOT20, WiderPerson
  (full-body / pedestrian bboxes) — strongest for fixing the baseline's missed small/occluded people.
- **Crowd-counting (density), real, labeled:** ShanghaiTech, JHU-CROWD++, NWPU-Crowd — head **points**.

**Can fine-tuning happen immediately?**
- ✅ **General crowded-person detection — YES, now.** Fine-tune YOLO on **CrowdHuman** (and/or
  WiderPerson / MOT20) after format conversion. This directly improves the dense / occluded /
  small-passenger detection that pretrained baseline models can miss on `sample.mp4`.
- ✅ **Head-based counting on platforms — YES, now.** Fine-tune a head detector on **RPEE-Heads**.
- ❌ **Indian-platform domain adaptation — NOT YET.** No labeled Indian platform data exists, and
  our Indian clips are unlabeled. To adapt specifically to Indian platforms you **must annotate**
  frames first. **An unlabeled video cannot be used for supervised fine-tuning.**

**Annotation work needed (to enable Indian-platform fine-tuning):**
1. Extract frames from the real Indian clips (Mumbai + Delhi; sample every ~0.5–1 s to avoid
   near-duplicate frames).
2. Label every passenger with a `person` bounding box (full body; for very dense regions, a
   head-point/head-box scheme like RPEE-Heads scales better).
3. Target a first usable set of **~300–500 labeled frames** for domain adaptation; grow toward
   several thousand boxes for robustness. Reserve held-out clips/frames for evaluation.
4. Maintain a single class taxonomy (`person`) and YOLO `.txt` label format.

**Recommended labeling workflow:**
- **Roboflow** — upload frames, **pre-label with the current YOLO model**, human-correct, export
  directly to **YOLO format**. Fastest path; cloud-based; handles train/val split and augmentation.
- **CVAT** — best for **video** annotation (box interpolation across frames); exports YOLO.
- **Label Studio** — flexible, ML-assisted pre-labeling; exports YOLO/COCO.
- **Bootstrap / active learning:** run the current pretrained YOLO model (`yolo11n.pt` by
  default, or `yolo11s.pt` when hardware allows) or a CrowdHuman-fine-tuned model to
  pre-annotate, then correct only the mistakes — far faster than labeling from scratch.

---

## Recommended Dataset Plan

1. **Test the current model on real platform video.**
   Run the existing pipeline on `data/input_videos/sample.mp4` (real Mumbai platform). Record
   per-frame person counts and where it fails. The legacy `yolov8n.pt` spot-check saw
   about 11–13 foreground persons/frame and missed some small distant passengers; record
   current `yolo11n.pt`/`yolo11s.pt` results before claiming improvement.

2. **Extract frames.**
   Sample frames from the real Indian clips (`pexels_crowded_train_station_6023186.mp4`,
   `pexels_people_in_train_station_12899783.mp4`) at ~0.5–1 s spacing into a labeling set.
   (Seed frames already in `data/sample_frames/`.)

3. **Label passengers in YOLO format.**
   Annotate `person` boxes via Roboflow/CVAT/Label Studio (pre-label → human-correct → export
   YOLO `.txt`). Keep a held-out split (e.g. the Delhi Metro clip) untouched for evaluation.

4. **Fine-tune YOLO.**
   *Quick win first:* fine-tune on **CrowdHuman** (person/full-body) to fix dense/small detection.
   *Then domain-adapt:* continue training on the locally labeled Indian-platform frames.
   Target **YOLO11** (the repo currently includes `yolo11n.pt`, `yolo11s.pt`, and
   `yolov8n.pt` as a legacy fallback).
   Consider a separate **head-detection** track trained on **RPEE-Heads** for dense crowd counting.

5. **Evaluate on held-out clips.**
   Measure mAP / precision-recall on held-out labeled frames, and counting error (MAE/RMSE of
   per-frame counts) on the held-out Indian clip. Compare against the step-1 baseline. Iterate:
   add more labeled frames where errors cluster (e.g. far end of platform, heavy occlusion).

---

## Sources
- Pexels — Crowded Train Station (Mumbai): https://www.pexels.com/video/crowded-train-station-6023186/
- Pexels — People in Train Station (Delhi Metro): https://www.pexels.com/video/people-in-train-station-12899783/
- Pexels License: https://www.pexels.com/license/
- Pixabay video search (indian train): https://pixabay.com/videos/search/indian%20train/
- RPEE-Heads: https://arxiv.org/abs/2411.18164 · https://doi.org/10.34735/ped.2024.2
- CrowdHuman: https://www.crowdhuman.org/ · https://huggingface.co/datasets/sshao0516/CrowdHuman
- MOT20: https://motchallenge.net/data/MOT20/ · https://arxiv.org/abs/2003.09003
- WiderPerson: http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/ · https://arxiv.org/abs/1909.12118
- JHU-CROWD++: http://www.crowd-counting.com/ · https://arxiv.org/pdf/2004.03597
- Roboflow railway crowd detection: https://universe.roboflow.com/muk-1bbp8/railway-crowd-detection
- Metro/boarding CCTV datasets (paper): https://pmc.ncbi.nlm.nih.gov/articles/PMC7662571/
