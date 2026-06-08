# Head Model Training Report

## Dataset Used
RPEE-Heads was used. No backup dataset and no synthetic data were used.

The local dataset at `data/head_datasets/yolo/rpee_heads/` was verified on
2026-06-02:

- train: 1,346 images, 1,346 label files, 78,594 head boxes
- val: 246 images, 246 label files, 16,020 head boxes
- labels: class `0 = head`, normalized YOLO values in `[0, 1]`

## Training Configuration
- base model: yolo11n.pt
- epochs: 25
- image size: 640
- batch size: 32
- device used: CUDA GPU 0 (NVIDIA A100-SXM4-40GB)
- dataset YAML: `configs/head_dataset.yaml`
- output path: `models/fine_tuned/head_detector/`

## Training Result
- training completed: YES
- training location: Google Colab, then artifacts copied into this repo
- local training run on this machine: NO, because the user supplied the completed Colab model artifacts
- Colab best model path: `/content/crowd-analysis/models/fine_tuned/head_detector/weights/best.pt`
- imported best.pt source path in this project: `artifacts/colab/best.pt`
- local best model path: `models/fine_tuned/head_detector/weights/best.pt`
- local last model path: `models/fine_tuned/head_detector/weights/last.pt`
- Colab zip artifact: `artifacts/colab/head_detector_colab_outputs.zip`
- results CSV: `models/fine_tuned/head_detector/results.csv`
- metrics available: YES
- SHA-256 verification: `artifacts/colab/best.pt` and
  `models/fine_tuned/head_detector/weights/best.pt` match
  (`33b591caaa4a474ca0ca3b48e7a651eb745ef5aab9f6844cb9870cf1ca286523`)

- epoch: 25
- time: 277.508
- train/box_loss: 1.49421
- train/cls_loss: 0.70712
- train/dfl_loss: 0.87144
- metrics/precision(B): 0.86562
- metrics/recall(B): 0.72303
- metrics/mAP50(B): 0.79376
- metrics/mAP50-95(B): 0.43098
- val/box_loss: 1.52613
- val/cls_loss: 0.70317
- val/dfl_loss: 0.86142
- lr/pg0: 9.92e-05
- lr/pg1: 9.92e-05
- lr/pg2: 9.92e-05

## Evaluation on Sample Video
- head_demo.mp4 created: YES
- source video used for that historical evaluation: `data/input_videos/sample.mp4`; this file is now the selected Pexels platform sample, so re-run evaluation before quoting current sample numbers
- output video path: `data/outputs/head_demo.mp4`
- local evaluation command: `python3 scripts/evaluate_head_detector.py --model models/fine_tuned/head_detector/weights/best.pt --source data/input_videos/sample.mp4 --output data/outputs/head_demo.mp4`
- sample image inference: 94 heads detected on one RPEE-Heads validation image
- historical sample video frames processed: 600
- average heads/frame: 7.31
- max heads/frame: 27
- local processing FPS: 14.34
- observations: automated inference completed and produced an annotated MP4; visual review is still required.
- limitations: RPEE-Heads is not Indian Railway CCTV. Domain validation on approved Indian railway footage is still required.

## Next Steps
- run body-vs-head comparison through `scripts/run_comparison_demo.py`
- validate on Indian Railway CCTV
- optionally train longer
- optionally use yolo11s
- tune thresholds
- compare with body detector
