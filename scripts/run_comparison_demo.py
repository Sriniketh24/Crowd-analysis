"""Run full-body and head-based passenger detection demos on the same video."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.detector import DEFAULT_HEAD_MODEL_PATH


BODY_OUTPUT = Path("data/outputs/body_demo.mp4")
BODY_DB = Path("data/outputs/body_analytics.db")
HEAD_OUTPUT = Path("data/outputs/head_demo.mp4")
HEAD_DB = Path("data/outputs/head_analytics.db")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the comparison demo."""
    parser = argparse.ArgumentParser(
        description="Run body and head passenger detection demos on the same input video."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source video path, webcam index, or stream URL.",
    )
    parser.add_argument(
        "--zones-config",
        type=Path,
        default=Path("configs/zones.cctv_platform.example.json"),
        help="Path to the zone/line JSON config used by both modes.",
    )
    parser.add_argument("--body-model", type=str, default=None, help="Optional body YOLO weights.")
    parser.add_argument(
        "--head-model",
        type=str,
        default=str(DEFAULT_HEAD_MODEL_PATH),
        help="Fine-tuned head detector weights.",
    )
    parser.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="YOLO non-max suppression IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference image size for both modes.")
    parser.add_argument("--body-imgsz", type=int, default=None, help="Body-mode YOLO inference image size.")
    parser.add_argument(
        "--head-confidence",
        type=float,
        default=None,
        help="Head-mode confidence threshold. Omit to use the configurable head default.",
    )
    parser.add_argument(
        "--head-imgsz",
        type=int,
        default=None,
        help="Head-mode YOLO inference image size. Omit to use the configurable head default.",
    )
    parser.add_argument("--tracker", type=str, default=None, choices=["bytetrack", "botsort"])
    parser.add_argument("--device", type=str, default=None, help="Inference device: cpu, cuda, mps, etc.")
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Ultralytics test-time augmentation for both modes when supported.",
    )
    parser.add_argument("--max-det", type=int, default=None, help="Maximum detections per frame for both modes.")
    parser.add_argument("--body-max-det", type=int, default=None, help="Body-mode maximum detections per frame.")
    parser.add_argument("--head-max-det", type=int, default=None, help="Head-mode maximum detections per frame.")
    parser.add_argument("--show", action="store_true", help="Show annotated preview windows while processing.")
    return parser.parse_args()


def _run_video_demo(command_args: list[str]) -> int:
    """Run the shared video demo script and stream its output."""
    command = [sys.executable, "scripts/run_video_demo.py", *command_args]
    print("\n$ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def _base_args(args: argparse.Namespace, *, output: Path, db: Path, mode: str) -> list[str]:
    command_args = [
        "--source",
        args.source,
        "--output",
        str(output),
        "--zones-config",
        str(args.zones_config),
        "--db",
        str(db),
        "--detector-mode",
        mode,
    ]
    if args.confidence is not None:
        command_args.extend(["--confidence", str(args.confidence)])
    if args.iou is not None:
        command_args.extend(["--iou", str(args.iou)])
    if args.tracker is not None:
        command_args.extend(["--tracker", args.tracker])
    if args.device is not None:
        command_args.extend(["--device", args.device])
    if args.augment is not None:
        command_args.append("--augment" if args.augment else "--no-augment")
    if args.show:
        command_args.append("--show")
    return command_args


def main() -> None:
    """Run body first, then head when the Colab-trained model is available."""
    args = parse_args()

    body_args = _base_args(args, output=BODY_OUTPUT, db=BODY_DB, mode="body")
    if args.body_model:
        body_args.extend(["--model", args.body_model])
    body_imgsz = args.body_imgsz if args.body_imgsz is not None else args.imgsz
    if body_imgsz is not None:
        body_args.extend(["--imgsz", str(body_imgsz)])
    body_max_det = args.body_max_det if args.body_max_det is not None else args.max_det
    if body_max_det is not None:
        body_args.extend(["--max-det", str(body_max_det)])
    body_code = _run_video_demo(body_args)

    head_model = Path(args.head_model)
    head_code = 0
    if not head_model.exists():
        print(
            "\nHead detector model not found; body mode already ran if the source was valid.\n"
            f"Expected Colab-trained best.pt at: {DEFAULT_HEAD_MODEL_PATH}\n"
            f"Requested head model path: {args.head_model}\n"
            "Place the imported Colab weights there or pass --head-model with an existing file."
        )
    else:
        head_args = _base_args(args, output=HEAD_OUTPUT, db=HEAD_DB, mode="head")
        head_args.extend(["--model", str(head_model)])
        head_imgsz = args.head_imgsz if args.head_imgsz is not None else args.imgsz
        head_max_det = args.head_max_det if args.head_max_det is not None else args.max_det
        if args.head_confidence is not None:
            head_args.extend(["--confidence", str(args.head_confidence)])
        if head_imgsz is not None:
            head_args.extend(["--imgsz", str(head_imgsz)])
        if head_max_det is not None:
            head_args.extend(["--max-det", str(head_max_det)])
        head_code = _run_video_demo(head_args)

    if body_code != 0 or head_code != 0:
        raise SystemExit(body_code or head_code)


if __name__ == "__main__":
    main()
