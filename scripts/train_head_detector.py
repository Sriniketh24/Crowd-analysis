"""Fine-tune a YOLO head detector on a real labeled head dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

import yaml

try:
    from ultralytics import YOLO
except ModuleNotFoundError:  # pragma: no cover - environment-specific dependency
    YOLO = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "docs/HEAD_DATASET_REPORT.md"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO for single-class human head detection.",
    )
    parser.add_argument("--data", type=Path, default=Path("configs/head_dataset.yaml"))
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("models/fine_tuned"))
    parser.add_argument("--name", type=str, default="head_detector")
    parser.add_argument("--device", type=str, default=None, help="Training device, e.g. cpu, mps, 0.")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Data loader workers. Keep low for laptop training.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve a CLI path relative to the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_dataset_source() -> str:
    """Extract the prepared dataset source from the dataset report when available."""
    if not REPORT_PATH.exists():
        return "Unknown - docs/HEAD_DATASET_REPORT.md not found"

    text = REPORT_PATH.read_text(encoding="utf-8")
    name_match = re.search(r"\|\s+\*\*Name\*\*\s+\|\s+([^|]+?)\s+\|", text)
    prepared_match = re.search(r"\|\s+\*\*RPEE-Heads prepared\?\*\*\s+\|\s+([^|]+?)\s+\|", text)
    backup_match = re.search(r"\|\s+Backup dataset name\s+\|\s+([^|]+?)\s+\|", text)

    name = name_match.group(1).strip("* ") if name_match else "Unknown"
    prepared = prepared_match.group(1).strip("* ") if prepared_match else "Unknown"
    backup = backup_match.group(1).strip() if backup_match else "Unknown"
    if prepared.upper() == "YES":
        return f"{name} (RPEE-Heads prepared: YES)"
    return f"Backup dataset: {backup}"


def load_dataset_config(dataset_yaml: Path) -> dict[str, Any]:
    """Load and minimally validate an Ultralytics dataset YAML."""
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}. Do not train without configs/head_dataset.yaml."
        )
    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {dataset_yaml}")
    for key in ("path", "train", "val", "names"):
        if key not in config:
            raise ValueError(f"Dataset YAML missing required key '{key}': {dataset_yaml}")
    names = config["names"]
    if not ((isinstance(names, dict) and names.get(0) == "head") or (isinstance(names, list) and names[:1] == ["head"])):
        raise ValueError("Dataset must be single-class head data with class 0 named 'head'.")
    return config


def labels_dir_for(images_rel: str) -> str:
    """Infer YOLO labels path from an images path."""
    parts = Path(images_rel).parts
    if "images" not in parts:
        raise ValueError(f"Expected an images path in dataset YAML, got: {images_rel}")
    return str(Path(*("labels" if part == "images" else part for part in parts)))


def validate_split(base: Path, images_rel: str, split_name: str) -> tuple[int, int, int]:
    """Validate one YOLO split and return image, label, and box counts."""
    images_dir = base / images_rel
    labels_dir = base / labels_dir_for(images_rel)
    if not images_dir.exists():
        raise FileNotFoundError(f"{split_name} images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"{split_name} labels directory not found: {labels_dir}")

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    label_paths = sorted(labels_dir.glob("*.txt"))
    if not image_paths:
        raise RuntimeError(f"No {split_name} images found under {images_dir}")
    if not label_paths:
        raise RuntimeError(f"No {split_name} labels found under {labels_dir}")

    label_names = {p.stem for p in label_paths}
    missing_labels = [p.name for p in image_paths if p.stem not in label_names]
    if missing_labels:
        sample = ", ".join(missing_labels[:5])
        raise RuntimeError(f"{split_name} images missing YOLO labels: {sample}")

    broken_symlinks = [p for p in image_paths if p.is_symlink() and not p.exists()]
    if broken_symlinks:
        sample = ", ".join(str(p) for p in broken_symlinks[:5])
        raise RuntimeError(f"{split_name} contains broken image symlinks: {sample}")

    total_boxes = 0
    for label_path in label_paths:
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise RuntimeError(f"Invalid YOLO row in {label_path}:{line_number}: expected 5 fields")
            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = (float(value) for value in parts[1:])
            except ValueError as exc:
                raise RuntimeError(f"Invalid numeric label in {label_path}:{line_number}") from exc
            if class_id != 0:
                raise RuntimeError(f"Invalid class id in {label_path}:{line_number}: expected 0=head")
            if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
                raise RuntimeError(f"Non-normalized label values in {label_path}:{line_number}")
            if width <= 0.0 or height <= 0.0:
                raise RuntimeError(f"Non-positive head box size in {label_path}:{line_number}")
            total_boxes += 1

    if total_boxes == 0:
        raise RuntimeError(f"No labeled head boxes found in {split_name} split.")
    return len(image_paths), len(label_paths), total_boxes


def validate_dataset(dataset_yaml: Path) -> dict[str, tuple[int, int, int]]:
    """Validate the dataset is present and labeled for single-class head training."""
    config = load_dataset_config(dataset_yaml)
    dataset_path = Path(config["path"])
    base = dataset_path if dataset_path.is_absolute() else PROJECT_ROOT / dataset_path
    if not base.exists():
        raise FileNotFoundError(f"Dataset path not found: {base}")
    return {
        "train": validate_split(base, str(config["train"]), "train"),
        "val": validate_split(base, str(config["val"]), "val"),
    }


def main() -> None:
    """Validate data and run Ultralytics training."""
    args = parse_args()
    dataset_yaml = resolve_path(args.data)
    project = resolve_path(args.project)

    print("Dataset source:", read_dataset_source())
    split_counts = validate_dataset(dataset_yaml)
    print("Dataset validation:")
    for split, (images, labels, boxes) in split_counts.items():
        print(f"  {split}: {images} images, {labels} label files, {boxes} head boxes")

    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed. Install dependencies first: pip install -r requirements.txt")

    print("Training configuration:")
    print(f"  data: {dataset_yaml}")
    print(f"  model: {args.model}")
    print(f"  epochs: {args.epochs}")
    print(f"  imgsz: {args.imgsz}")
    print(f"  batch: {args.batch}")
    print(f"  project: {project}")
    print(f"  name: {args.name}")
    print(f"  device: {args.device or 'auto'}")
    print(f"  workers: {args.workers}")

    project.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project),
        name=args.name,
        device=args.device,
        workers=args.workers,
        exist_ok=True,
    )

    best_model_path = project / args.name / "weights" / "best.pt"
    print(f"Best model path: {best_model_path}")
    if not best_model_path.exists():
        raise RuntimeError(f"Training finished but best model was not found at {best_model_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
