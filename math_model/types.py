"""Tipos partilhados entre classificadores de queda."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PoseObservation:
    """Keypoints YOLO crus de uma pessoa num instante."""

    person_id: int
    keypoints: np.ndarray  # shape (17, 3) — x, y, conf
    timestamp: float


@dataclass(frozen=True)
class FallDecision:
    """Decisão uniforme do miolo de classificação."""

    person_id: int
    label: str
    alert: bool
    detail: dict[str, Any] | None = None
