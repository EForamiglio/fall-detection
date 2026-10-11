"""
Implementação do filtro de Kalman para suavização dos keypoints
"""

import numpy as np

from .capture_constants import (
    TRACKED_KPTS,
    MIN_KPT_CONF,
    SCALE_EMA_ALPHA,
    MIN_TORSO_SCALE,
)
from .feature_helpers import align_pca_angle
from .person_runtime import get_person_runtime_store


class KalmanPointTracker:
    """
    Kalman linear para um ponto 2D, modelo de velocidade constante
    """

    def __init__(self, x0, y0, capture_date, process_noise=4.0, measurement_noise=3.0):
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=np.float64)
        self.p = np.eye(4) * 10.0
        self.q = process_noise
        self.r = measurement_noise
        self.h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.last_ts = capture_date
        self.missed_frames = 0

    def _f(self, dt):
        return np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )

    def _q(self, dt):
        q = self.q
        return (
            np.array(
                [
                    [dt**4 / 4, 0, dt**3 / 2, 0],
                    [0, dt**4 / 4, 0, dt**3 / 2],
                    [dt**3 / 2, 0, dt**2, 0],
                    [0, dt**3 / 2, 0, dt**2],
                ],
                dtype=np.float64,
            )
            * q
        )

    def step(self, z, capture_date, conf=1.0):
        """
        Passo do filtro de Kalman
        """
        dt = max(capture_date - self.last_ts, 1e-3)
        self.last_ts = capture_date

        f = self._f(dt)
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + self._q(dt)

        if z is not None:
            r = np.eye(2) * (self.r / max(conf, 1e-3))
            y = z - self.h @ self.x
            s = self.h @ self.p @ self.h.T + r
            k = self.p @ self.h.T @ np.linalg.inv(s)
            self.x = self.x + k @ y
            self.p = (np.eye(4) - k @ self.h) @ self.p
            self.missed_frames = 0
        else:
            self.missed_frames += 1

        return self.x


def apply_kalman(
    person_id: int,
    capture_date: float,
    points: dict[int, list],
) -> dict[int, dict]:
    """
    Recebe os keypoints brutos de uma pessoa nesse frame e retorna
    posição suavizada + velocidade para os pontos rastreados.
    """
    state = get_person_runtime_store().get_or_create(person_id)
    smoothed = {}

    for kpt_idx in TRACKED_KPTS:
        raw = points.get(kpt_idx)
        valid = raw is not None and raw[2] >= MIN_KPT_CONF

        if kpt_idx not in state.kalman_trackers:
            if not valid:
                continue
            state.kalman_trackers[kpt_idx] = KalmanPointTracker(
                raw[0], raw[1], capture_date
            )

        z = np.array([raw[0], raw[1]]) if valid else None
        conf = raw[2] if valid else 0.0
        x, y, vx, vy = state.kalman_trackers[kpt_idx].step(z, capture_date, conf)
        smoothed[kpt_idx] = {
            "x": float(x),
            "y": float(y),
            "vx": float(vx),
            "vy": float(vy),
        }

    return smoothed


def get_smoothed_scale(person_id: int, raw_scale: float) -> float | None:
    """
    EMA do comprimento do torso por pessoa rastreada.
    Retorna None se o scale bruto for inválido e ainda não houver histórico.
    """
    state = get_person_runtime_store().get_or_create(person_id)
    if raw_scale < MIN_TORSO_SCALE:
        return state.scale_ema
    if state.scale_ema is None:
        state.scale_ema = raw_scale
    else:
        state.scale_ema = (
            SCALE_EMA_ALPHA * raw_scale + (1.0 - SCALE_EMA_ALPHA) * state.scale_ema
        )
    return state.scale_ema


def align_and_store_pca_angle(person_id: int, raw_angle: float) -> float:
    state = get_person_runtime_store().get_or_create(person_id)
    aligned = align_pca_angle(raw_angle, state.pca_angle)
    state.pca_angle = aligned
    return aligned
