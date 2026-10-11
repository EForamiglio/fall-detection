"""
Processamento das features de uma janela deslizante
"""

from typing import Any
import math

import numpy as np

from .feature_helpers import (
    get_angle_speed,
    get_linear_speed,
    get_angular_speed,
    get_linear_acceleration,
    get_angular_acceleration,
    sigmoid_normalize,
)


def __get_raw_features(window_coordinates: list) -> list[dict[str, Any]]:
    """
    Extrai as features (velocidade e aceleração linear/angular) de cada parte do
    corpo ao longo de uma janela deslizante, mantendo o alinhamento com os
    frames de entrada.

    """

    features: list[dict[str, Any]] = []

    # posição e timestamp do frame anterior, mantidos por parte do corpo
    previous_positions: dict[int, tuple[float, float]] = {}
    previous_timestamps: dict[int, float] = {}

    # velocidades do instante anterior, usadas para derivar a aceleração
    previous_speeds: dict[int, tuple[float, float]] = {}
    previous_angular_speeds: dict[int, float] = {}
    previous_speed_timestamps: dict[int, float] = {}

    # estado temporal do pca_angle
    previous_pca_angle: float | None = None
    previous_pca_ts: float | None = None
    previous_pca_speed: float | None = None

    for frame in window_coordinates:
        current_ts = frame["timestamp"]
        body_parts = frame["coordinates"]
        pca_angle = frame.get("pca_angle")

        pca_angular_speed = 0.0
        pca_angular_acceleration = 0.0

        if (
            pca_angle is not None
            and previous_pca_angle is not None
            and previous_pca_ts is not None
        ):
            pca_angular_speed = get_angle_speed(
                pca_angle, current_ts, previous_pca_angle, previous_pca_ts
            )

            if previous_pca_speed is not None:
                pca_angular_acceleration = get_angular_acceleration(
                    pca_angular_speed, current_ts, previous_pca_speed, previous_pca_ts
                )

            previous_pca_speed = pca_angular_speed

        if pca_angle is not None:
            previous_pca_angle = pca_angle
            previous_pca_ts = current_ts

        frame_features: dict[str, Any] = {
            "timestamp": current_ts,
            "trunk_angle": frame.get("trunk_angle"),
            "center_of_mass": frame.get("center_of_mass"),
            "pca_ratio": frame.get("pca_ratio"),
            "pca_angle": pca_angle,
            "pca_angular_speed": pca_angular_speed,
            "pca_angular_acceleration": pca_angular_acceleration,
            "parts": {},
        }

        for part, (current_x, current_y) in body_parts.items():
            part_features: dict[str, Any] = {
                "coordinates": (current_x, current_y),
                "linear_speed_x": 0.0,
                "linear_speed_y": 0.0,
                "angular_speed": 0.0,
                "linear_acceleration_x": 0.0,
                "linear_acceleration_y": 0.0,
                "angular_acceleration": 0.0,
            }

            # há frame anterior dessa parte: dá para calcular as velocidades
            if part in previous_positions:
                previous_position = previous_positions[part]
                previous_x, previous_y = previous_position
                previous_ts = previous_timestamps[part]

                speed_x = get_linear_speed(
                    current_x, current_ts, previous_x, previous_ts
                )
                speed_y = get_linear_speed(
                    current_y, current_ts, previous_y, previous_ts
                )
                angular_speed = get_angular_speed(
                    (current_x, current_y), current_ts, previous_position, previous_ts
                )

                part_features["linear_speed_x"] = speed_x
                part_features["linear_speed_y"] = speed_y
                part_features["angular_speed"] = angular_speed

                # há velocidade anterior: dá para derivar a aceleração
                if part in previous_speeds:
                    previous_speed_x, previous_speed_y = previous_speeds[part]
                    previous_angular_speed = previous_angular_speeds[part]
                    previous_speed_ts = previous_speed_timestamps[part]

                    part_features["linear_acceleration_x"] = get_linear_acceleration(
                        speed_x, current_ts, previous_speed_x, previous_speed_ts
                    )
                    part_features["linear_acceleration_y"] = get_linear_acceleration(
                        speed_y, current_ts, previous_speed_y, previous_speed_ts
                    )
                    part_features["angular_acceleration"] = get_angular_acceleration(
                        angular_speed,
                        current_ts,
                        previous_angular_speed,
                        previous_speed_ts,
                    )

                previous_speeds[part] = (speed_x, speed_y)
                previous_angular_speeds[part] = angular_speed
                previous_speed_timestamps[part] = current_ts

            previous_positions[part] = (current_x, current_y)
            previous_timestamps[part] = current_ts

            frame_features["parts"][part] = part_features

        features.append(frame_features)

    return features


def __linear_r2(timestamps: np.ndarray, values: np.ndarray) -> float:
    """R² da regressão linear values ~ timestamps (tendência 'limpa')."""
    if values.size < 2:
        return 0.0

    slope, intercept = np.polyfit(timestamps, values, 1)
    predicted = slope * timestamps + intercept
    residuals = values - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))

    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0

    return 1.0 - ss_res / ss_tot


def __aggregate_window_features(raw_features: list[dict[str, Any]]) -> dict[str, float]:
    """
    Agrega a série bruta da janela em um único vetor de features de classificação.
    """
    timestamps: list[float] = []
    trunk_angles: list[float] = []
    pca_ratios: list[float] = []
    com_ys: list[float] = []

    for frame in raw_features:
        trunk_angle = frame.get("trunk_angle")
        pca_ratio = frame.get("pca_ratio")
        center_of_mass = frame.get("center_of_mass")

        if trunk_angle is None or pca_ratio is None or center_of_mass is None:
            continue

        timestamps.append(frame["timestamp"])
        trunk_angles.append(float(trunk_angle))
        pca_ratios.append(float(pca_ratio))
        com_ys.append(float(center_of_mass[1]))

    if len(timestamps) < 2:
        return {
            "trunk_angle_delta": 0.0,
            "trunk_angle_max_rate": 0.0,
            "pca_ratio_delta": 0.0,
            "center_mass_max_accel_y": 0.0,
            "center_mass_accel_poly": 0.0,
            "trunk_angle_trend_r2": 0.0,
        }

    t = np.asarray(timestamps, dtype=float)
    trunk = np.asarray(trunk_angles, dtype=float)
    pca_ratio = np.asarray(pca_ratios, dtype=float)
    com_y = np.asarray(com_ys, dtype=float)
    t_rel = t - t[0]

    # Postura — mudança estrutural (wrap em [-π, π] no delta do ângulo)
    trunk_angle_delta = float(trunk[-1] - trunk[0] + math.pi) % (2 * math.pi) - math.pi
    trunk_rates = [
        abs(get_angle_speed(trunk[i], t[i], trunk[i - 1], t[i - 1]))
        for i in range(1, len(trunk))
    ]
    trunk_angle_max_rate = max(trunk_rates) if trunk_rates else 0.0
    pca_ratio_delta = float(pca_ratio[-1] - pca_ratio[0])

    # Movimento — intensidade do evento (derivadas do CoM em y)
    com_speeds_y: list[float] = []
    for i in range(1, len(com_y)):
        com_speeds_y.append(get_linear_speed(com_y[i], t[i], com_y[i - 1], t[i - 1]))

    com_accels_y: list[float] = []
    for i in range(1, len(com_speeds_y)):
        # timestamps das velocidades correspondem a t[1], t[2], ...
        com_accels_y.append(
            get_linear_acceleration(
                com_speeds_y[i],
                t[i + 1],
                com_speeds_y[i - 1],
                t[i],
            )
        )

    center_mass_max_accel_y = max(abs(a) for a in com_accels_y) if com_accels_y else 0.0

    # y(t) ≈ a·t² + b·t + c  → 'a' captura aceleração média da trajetória
    if com_y.size >= 3:
        poly_a, _, _ = np.polyfit(t_rel, com_y, 2)
        center_mass_accel_poly = float(poly_a)
    else:
        center_mass_accel_poly = 0.0

    # Consistência — R² da tendência linear do trunk_angle
    trunk_angle_trend_r2 = __linear_r2(t_rel, trunk)

    return {
        "trunk_angle_delta": float(trunk_angle_delta),
        "trunk_angle_max_rate": float(trunk_angle_max_rate),
        "pca_ratio_delta": pca_ratio_delta,
        "center_mass_max_accel_y": float(center_mass_max_accel_y),
        "center_mass_accel_poly": center_mass_accel_poly,
        "trunk_angle_trend_r2": float(trunk_angle_trend_r2),
    }


def extract_features(window_coordinates: list) -> dict[str, float]:
    """
    A partir das features brutas, aplica as regras de seleção e agregação das
    propriedades para a geração das features de classificação dos movimentos.
    """
    raw_features = __get_raw_features(window_coordinates)
    window_features = __aggregate_window_features(raw_features)

    return window_features


def normalize_features(features: dict[str, float]) -> dict[str, float]:
    """
    Normaliza as features para o intervalo [0, 1]
    """

    # TODO: Calibrar os valores de midpoint e steepness a partir de dados reais e rotulados

    f_trunk_delta = sigmoid_normalize(abs(features["trunk_angle_delta"]),midpoint= 0.6, steepness= 8)
    f_r2 = sigmoid_normalize(features["trunk_angle_trend_r2"], midpoint= 0.7, steepness= 6)
    f_pca_delta = sigmoid_normalize(-features["pca_ratio_delta"], midpoint= 0.5, steepness= 4)
    f_accel_max = sigmoid_normalize(features["center_mass_max_accel_y"], midpoint= 2.0, steepness= 1.5)
    f_accel_poly = sigmoid_normalize(features["center_mass_accel_poly"], midpoint= 1.0, steepness= 2)
    f_max_rate = sigmoid_normalize(features["trunk_angle_max_rate"], midpoint= 0.8, steepness= 3)

    return {
        'trunk_angle_delta': f_trunk_delta,
        'trunk_angle_trend_r2': f_r2,
        'pca_ratio_delta': f_pca_delta,
        'center_mass_max_accel_y': f_accel_max,
        'center_mass_accel_poly': f_accel_poly,
        'trunk_angle_max_rate': f_max_rate,
    }