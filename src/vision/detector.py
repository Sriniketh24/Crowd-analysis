"""Person detector abstraction over Ultralytics YOLO models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time

try:
    from ultralytics import YOLO
except ModuleNotFoundError:  # pragma: no cover - environment-specific dependency
    YOLO = None  # type: ignore[assignment]

DEFAULT_MODEL_CANDIDATES: tuple[str, ...] = ("yolo11s.pt", "yolo11n.pt", "yolov8n.pt")
DEFAULT_PERSON_CLASS_ID = 0
FINE_TUNED_MODEL_PATH = Path("models/fine_tuned/best.pt")


@dataclass(slots=True)
class NormalizedDetection:
    """Normalized frame-level detection or track output."""

    track_id: int | None
    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str
    frame_index: int
    timestamp: float


@dataclass(slots=True)
class DetectionResult:
    """Normalized detection collection for one frame."""

    detections: list[NormalizedDetection]


def resolve_model_candidates(
    weights_path: str | None = None,
    *,
    accuracy_weights: str | None = None,
    legacy_fallback_weights: str | None = None,
    use_fine_tuned_if_available: bool = True,
) -> tuple[str, ...]:
    """Return ordered model candidates from explicit and fallback settings."""
    candidates: list[str] = []
    if weights_path:
        candidates.append(weights_path)
    if use_fine_tuned_if_available and FINE_TUNED_MODEL_PATH.exists():
        candidates.append(str(FINE_TUNED_MODEL_PATH))
    if accuracy_weights:
        candidates.append(accuracy_weights)
    candidates.extend(DEFAULT_MODEL_CANDIDATES)
    if legacy_fallback_weights:
        candidates.append(legacy_fallback_weights)

    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def build_model(
    weights_path: str | None = None,
    *,
    accuracy_weights: str | None = None,
    legacy_fallback_weights: str | None = None,
    use_fine_tuned_if_available: bool = True,
) -> YOLO:
    """Create a YOLO model, falling back to known lightweight defaults."""
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is not installed. Install dependencies first: pip install -r requirements.txt"
        )
    candidates = resolve_model_candidates(
        weights_path,
        accuracy_weights=accuracy_weights,
        legacy_fallback_weights=legacy_fallback_weights,
        use_fine_tuned_if_available=use_fine_tuned_if_available,
    )
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return YOLO(candidate)
        except Exception as exc:  # pragma: no cover - depends on runtime/model availability
            last_error = exc
    tried = ", ".join(candidates)
    raise RuntimeError(
        f"Unable to load an Ultralytics YOLO model. Tried: {tried}. "
        "Pass an existing local weights file via --model (e.g. --model yolo11n.pt), "
        "or allow network access so Ultralytics can download the weights on first run. "
        f"Underlying error: {last_error}"
    ) from last_error


class Detector:
    """YOLO detector that returns normalized person detections."""

    def __init__(
        self,
        weights_path: str | None = None,
        device: str = "cpu",
        confidence: float = 0.35,
        iou: float = 0.5,
        person_class_id: int = DEFAULT_PERSON_CLASS_ID,
        imgsz: int = 640,
        half: bool = False,
        accuracy_weights: str | None = None,
        legacy_fallback_weights: str | None = None,
        use_fine_tuned_if_available: bool = True,
    ) -> None:
        """Initialize the detector with model and inference parameters."""
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.person_class_id = person_class_id
        self.imgsz = imgsz
        self.half = half
        self.weights_path = weights_path
        self.model = build_model(
            weights_path,
            accuracy_weights=accuracy_weights,
            legacy_fallback_weights=legacy_fallback_weights,
            use_fine_tuned_if_available=use_fine_tuned_if_available,
        )

    def detect(
        self,
        frame: object,
        frame_index: int = 0,
        timestamp: float | None = None,
    ) -> DetectionResult:
        """Run person detection on one frame and normalize outputs."""
        frame_ts = time() if timestamp is None else timestamp
        predictions = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.person_class_id],
            device=self.device,
            imgsz=self.imgsz,
            half=self.half,
            verbose=False,
        )
        if not predictions:
            return DetectionResult(detections=[])

        prediction = predictions[0]
        boxes = prediction.boxes
        if boxes is None:
            return DetectionResult(detections=[])

        names = prediction.names or getattr(self.model, "names", {})
        normalized: list[NormalizedDetection] = []
        for box in boxes:
            class_id = int(box.cls.item())
            if class_id != self.person_class_id:
                continue
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            class_name = str(names.get(class_id, "person"))
            normalized.append(
                NormalizedDetection(
                    track_id=None,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf.item()),
                    class_name=class_name,
                    frame_index=frame_index,
                    timestamp=frame_ts,
                )
            )
        return DetectionResult(detections=normalized)
