"""Wrapper YOLO pose + tracker (BotSort) — lifted de vigia/fall-detection.

Retorna detecções no formato (person_id, kpts_flat(51,), bbox(4,)).
Keypoints com confidence < CONF_THRESHOLD são zerados (mantêm posição no vetor).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

CONF_THRESHOLD = 0.25   # zera keypoints abaixo disso (alinhado com vigia)


@dataclass
class Detection:
    person_id: int
    kpts_flat: np.ndarray   # shape (51,) — 17 joints × (x, y, conf)
    bbox: np.ndarray        # shape (4,) — xyxy
    n_valid_kpts: int       # quantos joints sobreviveram ao threshold


class PoseDetector:
    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        device: str = "cpu",
        conf: float = 0.25,
        imgsz: int = 640,
    ) -> None:
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
            persist=True,
            tracker="botsort.yaml",
        )

        detections: list[Detection] = []

        for result in results:
            kpts = result.keypoints
            if kpts is None or kpts.data is None or len(kpts.data) == 0:
                continue

            boxes = result.boxes
            ids_tensor = getattr(boxes, "id", None)
            n = len(kpts.data)

            if ids_tensor is not None and len(ids_tensor) >= n:
                person_ids = [int(ids_tensor[i].item()) for i in range(n)]
            else:
                person_ids = list(range(n))

            bboxes_xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else None

            for i, (pid, person_kpts) in enumerate(zip(person_ids, kpts.data)):
                kpts_np = person_kpts.cpu().numpy() if hasattr(person_kpts, "cpu") else person_kpts.numpy()

                # Zera joints abaixo do threshold de confiança
                low_conf_mask = kpts_np[:, 2] < CONF_THRESHOLD
                kpts_np[low_conf_mask] = 0.0
                n_valid = int(np.sum(kpts_np[:, 2] > 0))

                bbox = bboxes_xyxy[i] if bboxes_xyxy is not None else np.zeros(4)
                detections.append(
                    Detection(
                        person_id=pid,
                        kpts_flat=kpts_np.flatten(),
                        bbox=bbox,
                        n_valid_kpts=n_valid,
                    )
                )

        return detections
