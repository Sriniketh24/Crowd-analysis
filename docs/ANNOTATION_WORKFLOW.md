# Annotation Workflow

Use this workflow because the repo contains real Indian railway videos but no labeled Indian-platform dataset.

1. Extract frames with:
   `python3 scripts/extract_training_frames.py --source data/indian_railway_videos/pexels_crowded_train_station_6023186.mp4 --out data/training_frames/raw --num 24`
2. Upload the frames to CVAT, Roboflow, or Label Studio.
3. Pre-label with the current YOLO model if available, then correct every `person` box manually.
4. Export YOLO format into:
   `data/training_dataset/images/train`
   `data/training_dataset/images/val`
   `data/training_dataset/images/test`
   `data/training_dataset/labels/train`
   `data/training_dataset/labels/val`
   `data/training_dataset/labels/test`
5. Keep a held-out validation/test split from a different clip or timestamp block.
6. Start training only after labels exist:
   `python3 scripts/train_yolo_indian_platform.py --data configs/train_indian_platform.yaml --model yolo11n.pt`

Rules:
- Label only `person`.
- Do not add identities, faces, or private metadata.
- Unlabeled videos are demo/testing material only and are not a supervised dataset.
