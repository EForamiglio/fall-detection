"""Classificador GRU 2-classes (ADL / FALL) — lifted de vigia/fall-detection.

Recebe janela (T, 51) de keypoints brutos do YOLO pose (17 joints × x,y,conf),
normaliza para (T, 34) hip-centered + escala shoulder-hip, roda ONNX e retorna
predição com flag de alerta confirmado por persistência temporal.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import onnxruntime as ort


class GRUFallClassifier:
    """GRU biclasse ADL/FALL com filtro temporal de confirmação."""

    ALERT_PREDS_FALL = 2          # n° de predições FALL consecutivas pra disparar alert
    _LABELS = ["ADL", "FALL"]
    _REQUIRED_JOINTS = [5, 6, 11, 12]   # ombros e quadris (COCO 17)

    def __init__(self, onnx_path: str | Path | None = None) -> None:
        if onnx_path is None:
            onnx_path = Path(__file__).resolve().parent / "model" / "gru_2classes.onnx"
        self._session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._pred_hist: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.ALERT_PREDS_FALL)
        )

    def predict(self, window: np.ndarray, person_id: int = 0) -> dict | None:
        """
        window: (T, 51) — keypoints brutos do YOLO (x, y, conf por joint).
        person_id: usado pra manter histórico de predições por pessoa.

        Retorna {"label", "probs", "alert", "n_valid_frames"}
        ou None se a janela tiver menos de 20% de frames com keypoints obrigatórios.
        """
        normalized, n_valid = self._normalize_window(window)
        if normalized is None:
            return None

        x = normalized.reshape(1, -1, 34).astype(np.float32)
        probs = self._session.run(None, {self._input_name: x})[0][0]   # (2,)
        pred_class = int(np.argmax(probs))

        self._pred_hist[person_id].append(pred_class)
        alert = pred_class != 0 and self._check_alert(person_id, pred_class)

        return {
            "label": self._LABELS[pred_class],
            "probs": probs.tolist(),
            "alert": alert,
            "n_valid_frames": int(n_valid),
        }

    def reset_person(self, person_id: int) -> None:
        """Limpa histórico de predições (chamar quando pessoa sai de cena)."""
        self._pred_hist.pop(person_id, None)

    # -------------------------------------------------------------------

    def _normalize_window(
        self, window: np.ndarray
    ) -> tuple[np.ndarray | None, int]:
        """
        (T, 51) pixels brutos → (T, 34) hip-centered, escala shoulder-hip.
        Frames com joints obrigatórios inválidos (conf == 0) ficam como zeros.
        Retorna (None, 0) se < 20% dos frames forem válidos.
        """
        kp_full = np.asarray(window, dtype=np.float32).reshape(-1, 17, 3)
        T = kp_full.shape[0]
        kp_xy = kp_full[:, :, :2]
        kp_conf = kp_full[:, :, 2]

        frame_valid = np.all(kp_conf[:, self._REQUIRED_JOINTS] > 0, axis=1)
        n_valid = int(frame_valid.sum())
        if frame_valid.mean() < 0.20:
            return None, n_valid

        sho_c = (kp_xy[:, 5, :] + kp_xy[:, 6, :]) * 0.5
        hip_c = (kp_xy[:, 11, :] + kp_xy[:, 12, :]) * 0.5
        sh_dist = np.linalg.norm(sho_c - hip_c, axis=1, keepdims=True)
        sh_dist = np.where(sh_dist < 1e-6, 1.0, sh_dist)

        out = np.zeros((T, 17, 2), dtype=np.float32)
        if frame_valid.any():
            normalized = (
                (kp_xy[frame_valid] - hip_c[frame_valid, np.newaxis, :])
                / sh_dist[frame_valid, np.newaxis, :]
            )
            # Re-zerar joints com conf=0 (estavam em (0,0) e a normalização os deslocou)
            joint_valid = kp_conf[frame_valid] > 0
            normalized *= joint_valid[:, :, np.newaxis]
            out[frame_valid] = normalized

        return out.reshape(T, 34), n_valid

    def _check_alert(self, person_id: int, current: int) -> bool:
        hist = list(self._pred_hist[person_id])
        return (
            len(hist) >= self.ALERT_PREDS_FALL
            and all(p == current for p in hist[-self.ALERT_PREDS_FALL:])
        )
