"""Train YOLO on labeled Indian platform frames when labels actually exist."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from ultralytics import YOLO
except ModuleNotFoundError:  # pragma: no cover
    YOLO = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on labeled Indian platform data.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/train_indian_platform.yaml"),
        help="Ultralytics dataset YAML.",
    )
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=Path, default=Path("models/fine_tuned"))
    return parser.parse_args()


def ensure_labels_exist(dataset_yaml: Path) -> None:
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Training config not found: {dataset_yaml}")
    text = dataset_yaml.read_text(encoding="utf-8")
    required = [
        "data/training_dataset/images/train",
        "data/training_dataset/labels/train",
    ]
    for path_str in required:
        path = Path(path_str)
        if not path.exists():
            raise RuntimeError(
                "Labeled dataset is not present. Do not fake fine-tuning.\n"
                "Run scripts/extract_training_frames.py, label the frames in CVAT/Roboflow/Label Studio, "
                "export YOLO format into data/training_dataset/, then rerun this script."
            )
    if "placeholder" in text.lower():
        raise RuntimeError(
            "configs/train_indian_platform.yaml is still a template placeholder. Fill in the real dataset paths first."
        )


def main() -> None:
    args = parse_args()
    ensure_labels_exist(args.data)
    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed. Install dependencies first: pip install -r requirements.txt")
    model = YOLO(args.model)
    args.project.mkdir(parents=True, exist_ok=True)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(args.project),
        name="indian_platform",
    )


if __name__ == "__main__":
    main()
