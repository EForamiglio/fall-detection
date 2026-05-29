"""Acumula métricas em CSV — uma linha por classificação GRU."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

CSV_COLUMNS = [
    "timestamp",
    "frame_idx",
    "person_id",
    "t_capture_ms",
    "t_yolo_ms",
    "t_gru_ms",
    "n_valid_kpts",
    "n_valid_frames_window",
    "label",
    "prob_adl",
    "prob_fall",
    "alert",
    "n_detections_total",
]


class MetricsLogger:
    def __init__(self, csv_path: Path):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fp, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._fp.flush()

    def record(
        self,
        frame_idx: int,
        person_id: int,
        t_capture_ms: float,
        t_yolo_ms: float,
        t_gru_ms: float,
        n_valid_kpts: int,
        n_valid_frames_window: int,
        label: str,
        prob_adl: float,
        prob_fall: float,
        alert: bool,
        n_detections_total: int,
    ) -> None:
        self._writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "frame_idx": frame_idx,
            "person_id": person_id,
            "t_capture_ms": round(t_capture_ms, 2),
            "t_yolo_ms": round(t_yolo_ms, 2),
            "t_gru_ms": round(t_gru_ms, 2),
            "n_valid_kpts": n_valid_kpts,
            "n_valid_frames_window": n_valid_frames_window,
            "label": label,
            "prob_adl": round(prob_adl, 4),
            "prob_fall": round(prob_fall, 4),
            "alert": int(alert),
            "n_detections_total": n_detections_total,
        })
        self._fp.flush()

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()
