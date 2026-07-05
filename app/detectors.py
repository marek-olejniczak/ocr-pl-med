"""Line detector seam for the pipeline.

The pipeline only needs `detect(image) -> list[Line]`, so swapping in the
benchmark winner is a one-line change here. UltralyticsDetector covers the
YOLO family (YOLOv8 / YOLO11 / RT-DETR - same loader). A winner with heavy
deps (detectron2, kraken) gets a small HTTP wrapper in its existing benchmark
container and a matching adapter with the same detect() signature.
"""

from dataclasses import dataclass


@dataclass
class Line:
    x: float
    y: float
    w: float
    h: float
    score: float


class UltralyticsDetector:
    def __init__(self, weights, imgsz=1024, conf=0.25, device=None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf  # 0.25 = the benchmark's operating point
        self.device = device

    def detect(self, image):
        res = self.model.predict(
            image, imgsz=self.imgsz, conf=self.conf,
            device=self.device, verbose=False,
        )[0]
        lines = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            lines.append(Line(x1, y1, x2 - x1, y2 - y1, float(box.conf[0])))
        return lines
