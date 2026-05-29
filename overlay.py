"""Overlay visual: skeleton COCO 17, bbox, label e HUD."""

from __future__ import annotations

import numpy as np
import cv2

# Conexões do esqueleto COCO 17 (índices dos joints)
COCO_SKELETON = [
    (5, 7), (7, 9),           # braço esq
    (6, 8), (8, 10),          # braço dir
    (5, 6),                   # ombros
    (5, 11), (6, 12),         # tronco
    (11, 12),                 # quadris
    (11, 13), (13, 15),       # perna esq
    (12, 14), (14, 16),       # perna dir
    (0, 1), (0, 2),           # nariz-olhos
    (1, 3), (2, 4),           # olhos-orelhas
]

LABEL_COLORS = {
    "ADL": (0, 200, 0),                  # verde
    "FALL_PENDING": (0, 200, 220),       # amarelo (detectado, aguarda confirmação)
    "FALL_ALERT": (0, 0, 255),           # vermelho (confirmado)
    "INVALID": (128, 128, 128),          # cinza
}


def draw_skeleton(frame: np.ndarray, kpts_flat: np.ndarray) -> None:
    """Desenha os 17 keypoints + conexões. Joints com conf=0 são pulados."""
    kp = kpts_flat.reshape(17, 3)
    pts = [(int(x), int(y)) if c > 0 else None for x, y, c in kp]

    # Conexões
    for a, b in COCO_SKELETON:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(frame, pts[a], pts[b], (255, 200, 0), 2)

    # Pontos
    for p in pts:
        if p is not None:
            cv2.circle(frame, p, 4, (0, 255, 255), -1)


def draw_person_label(
    frame: np.ndarray,
    bbox: np.ndarray,
    person_id: int,
    pred: dict | None,
) -> None:
    """Bbox + ID + label de classificação acima da pessoa."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

    if pred is None:
        text = f"ID {person_id}: aguardando..."
        color = LABEL_COLORS["INVALID"]
    else:
        label = pred["label"]
        probs = pred["probs"]
        alert = pred["alert"]

        if label == "FALL" and alert:
            text = f"ID {person_id}: FALL! ({probs[1]:.2f})"
            color = LABEL_COLORS["FALL_ALERT"]
        elif label == "FALL":
            text = f"ID {person_id}: FALL? ({probs[1]:.2f})"
            color = LABEL_COLORS["FALL_PENDING"]
        else:
            text = f"ID {person_id}: ADL ({probs[0]:.2f})"
            color = LABEL_COLORS["ADL"]

    # Fundo do texto
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, text, (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    fps: float,
    t_yolo_ms: float,
    t_gru_ms: float,
    n_detections: int,
    model_name: str,
) -> None:
    """HUD canto superior esquerdo com métricas em tempo real."""
    lines = [
        f"Model : {model_name}",
        f"FPS   : {fps:5.1f}",
        f"YOLO  : {t_yolo_ms:5.1f} ms",
        f"GRU   : {t_gru_ms:5.1f} ms",
        f"People: {n_detections}",
    ]

    x, y = 10, 25
    pad = 4
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(
            frame, (x - pad, y - th - pad), (x + tw + pad, y + pad),
            (0, 0, 0), -1,
        )
        cv2.putText(
            frame, line, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        y += th + 8
