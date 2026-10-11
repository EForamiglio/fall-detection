"""Classificador matematico: Kalman -> janela -> features -> score -> FallDetector.

COPIA de vigia/fall-detection (branch develop, vigia-fall/capture/classifiers/math_classifier.py),
adaptada so nos imports para o pacote math_model/ deste bench. Logica de negocio
intocada -- nao reimplementar nada daqui, copiar de novo se o Pedro atualizar.
"""

from __future__ import annotations

import logging
from math import dist
from typing import Any

import numpy as np

from math_model.types import FallDecision, PoseObservation
from math_model.features_processor import extract_features, normalize_features
from math_model.fall_detector import FallState
from math_model.slider_window import SlidingWindowManager
from math_model.kalman_filter import (
    align_and_store_pca_angle,
    apply_kalman,
    get_smoothed_scale,
)
from math_model.feature_helpers import (
    compute_fall_score,
    get_center_of_mass,
    get_pca_features,
    get_trunk_angle,
)
from math_model.person_runtime import get_person_runtime_store

logger = logging.getLogger(__name__)

# TODO: Calibrar os valores de weights a partir de dados reais e rotulados
_WEIGHTS = {
    "trunk_angle_delta": 0.25,
    "trunk_angle_trend_r2": 0.20,
    "pca_ratio_delta": 0.15,
    "center_mass_max_accel_y": 0.20,
    "center_mass_accel_poly": 0.10,
    "trunk_angle_max_rate": 0.10,
}


def _central_point(point_left: float, point_right: float) -> float:
    return (point_left + point_right) / 2


def _normalize_body_part(
    scale: float, hip: tuple[float, float], body_part: dict[str, float]
) -> tuple[float, float]:
    x = (body_part["x"] - hip[0]) / scale
    y = (body_part["y"] - hip[1]) / scale
    return x, y


def _normalize_data(
    frame_points: dict[int, tuple[dict, dict]],
) -> dict[int, dict[str, Any]]:
    normalized_data: dict[int, dict[str, Any]] = {}

    for person_id, (_, smoothed_points) in frame_points.items():
        shoulder_left = smoothed_points.get(5)
        shoulder_right = smoothed_points.get(6)
        hip_left = smoothed_points.get(11)
        hip_right = smoothed_points.get(12)

        if (
            shoulder_left is None
            or shoulder_right is None
            or hip_left is None
            or hip_right is None
        ):
            continue

        shoulder_center = (
            _central_point(shoulder_left["x"], shoulder_right["x"]),
            _central_point(shoulder_left["y"], shoulder_right["y"]),
        )
        hip_center = (
            _central_point(hip_left["x"], hip_right["x"]),
            _central_point(hip_left["y"], hip_right["y"]),
        )

        raw_scale = dist(shoulder_center, hip_center)
        scale = get_smoothed_scale(person_id, raw_scale)
        if scale is None:
            continue

        trunk_angle = get_trunk_angle(shoulder_center, hip_center)
        raw_com = get_center_of_mass(shoulder_center, hip_center)
        center_of_mass = (
            (raw_com[0] - hip_center[0]) / scale,
            (raw_com[1] - hip_center[1]) / scale,
        )

        normalized_parts = {
            body_part_id: _normalize_body_part(scale, hip_center, body_part)
            for body_part_id, body_part in smoothed_points.items()
        }

        pca_ratio, raw_pca_angle = get_pca_features(normalized_parts)
        pca_angle = align_and_store_pca_angle(person_id, raw_pca_angle)

        normalized_data[person_id] = {
            "coordinates": normalized_parts,
            "trunk_angle": trunk_angle,
            "center_of_mass": center_of_mass,
            "pca_ratio": pca_ratio,
            "pca_angle": pca_angle,
        }

    return normalized_data


class MathFallClassifier:
    """Pipeline matemático (janela deslizante + score ponderado + FallDetector)."""

    def __init__(self, window_size: int) -> None:
        self._slider_window_manager = SlidingWindowManager(window_size=window_size)
        self._last_state: dict[int, FallState] = {}

    @property
    def window_capacity(self) -> int:
        return self._slider_window_manager.window_size

    def get_window_fill(self) -> dict[int, tuple[int, int]]:
        return self._slider_window_manager.get_fill_status()

    def process(self, observations: list[PoseObservation]) -> list[FallDecision]:
        if not observations:
            return []

        frame_points: dict[int, tuple[dict, dict]] = {}
        timestamp = observations[0].timestamp

        for obs in observations:
            kpts = np.asarray(obs.keypoints, dtype=np.float64)
            if kpts.ndim != 2 or kpts.shape[0] < 17:
                continue
            points = {
                idx: {
                    "x": float(kpts[idx, 0]),
                    "y": float(kpts[idx, 1]),
                    "conf": float(kpts[idx, 2]),
                }
                for idx in range(17)
                if float(kpts[idx, 2]) != 0.0
            }
            # apply_kalman espera dict[int, list] no código legado — adaptar
            points_for_kalman = {
                idx: [v["x"], v["y"], v["conf"]] for idx, v in points.items()
            }
            smoothed = apply_kalman(obs.person_id, obs.timestamp, points_for_kalman)
            frame_points[obs.person_id] = (points_for_kalman, smoothed)

        frame_result: dict[int, dict[str, Any]] = {}
        for person_id, body_parts in _normalize_data(frame_points).items():
            frame_result[person_id] = {
                "coordinates": body_parts["coordinates"],
                "trunk_angle": body_parts["trunk_angle"],
                "center_of_mass": body_parts["center_of_mass"],
                "pca_ratio": body_parts["pca_ratio"],
                "pca_angle": body_parts["pca_angle"],
                "timestamp": timestamp,
            }

        if not frame_result:
            return []

        ready_ids = self._slider_window_manager.update(frame_result)
        decisions: list[FallDecision] = []

        for person_id in ready_ids:
            window = self._slider_window_manager.get_window(person_id)
            if not window:
                continue
            try:
                features = extract_features(list(window))
                normalized_features = normalize_features(features)
                score = compute_fall_score(normalized_features, _WEIGHTS)
                person_state = get_person_runtime_store().get_or_create(person_id)
                state = person_state.fall_detector.update(score, timestamp)

                previous = self._last_state.get(person_id)
                self._last_state[person_id] = state
                alert = state is FallState.FALL and previous is not FallState.FALL

                decisions.append(
                    FallDecision(
                        person_id=person_id,
                        label=state.name,
                        alert=alert,
                        detail={"score": score},
                    )
                )
            except Exception as error:
                logger.warning(
                    "Erro ao extrair features person_id=%s: %s",
                    person_id,
                    error,
                )

        return decisions
