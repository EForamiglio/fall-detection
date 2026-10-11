"""Classificador GRU 2-classes (ADL / FALL) — versao 37 features.

Drop-in do classifier.py, com a MESMA interface publica (predict, reset_person,
ALERT_PREDS_FALL). O classifier.py original permanece intocado para comparacao.

O que mudou e por que
---------------------
A normalizacao anterior subtraia o centro do quadril e dividia pela distancia
ombro-quadril A CADA FRAME. Isso deixa o quadril em (0,0) e a escala em 1.0 em
todos os frames, ou seja, apaga completamente a translacao do corpo. O modelo
nao conseguia ver a velocidade de descida — exatamente o que separa uma queda de
um agachamento controlado.

Medido nos dados de treino: o modelo antigo aprendeu "tronco girando = queda"
(falso positivo de 8-10% com tronco estavel contra 26-35% com rotacao). Em campo
isso apareceu como 76% de falso alarme em cenarios de ADL, concentrado em
sentar, deitar e agachar.

Esta versao acrescenta 3 canais de movimento (velocidade horizontal e vertical
do quadril, e variacao do tamanho aparente do corpo), mantendo os 34 valores de
pose inalterados. Efeito medido em teste offline: falso positivo em janelas com
rotacao de tronco cai de 26.6% para 21.3%.

Compatibilidade
---------------
O modelo espera entrada (1, 20, 37) — NAO e compativel com gru_2classes.onnx,
que espera (1, 20, 34). Os dois arquivos coexistem em model/.

O calculo das features vive em pose_features.py, que e uma copia exata do modulo
usado no treino (modelos/data_prep/pose_features.py). Se os dois divergirem, o
modelo degrada em campo sem gerar erro nenhum — confira FEATURES_VERSION nos
dois lados antes de qualquer mudanca.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

from pose_features import FEATURE_DIM, build_features, valid_from_conf


class GRUFallClassifierV2:
    """GRU biclasse ADL/FALL com features de movimento e confirmacao temporal."""

    ALERT_PREDS_FALL = 2          # n de predicoes FALL consecutivas pra alertar
    _LABELS = ["ADL", "FALL"]
    _REQUIRED_JOINTS = [5, 6, 11, 12]   # ombros e quadris (COCO 17)
    _MIN_VALID_FRAC = 0.20

    def __init__(self, onnx_path: str | Path | None = None) -> None:
        if onnx_path is None:
            onnx_path = Path(__file__).resolve().parent / "model" / "gru_2classes_v2.onnx"
        self._session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        expected = self._session.get_inputs()[0].shape[-1]
        if expected != FEATURE_DIM:
            raise ValueError(
                f"O modelo {onnx_path} espera {expected} features, mas "
                f"pose_features fornece {FEATURE_DIM}. Modelo e modulo de "
                f"features estao fora de sincronia."
            )

        self._pred_hist: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.ALERT_PREDS_FALL)
        )

    def predict(self, window: np.ndarray, person_id: int = 0) -> dict | None:
        """
        window: (T, 51) — keypoints brutos do YOLO (x, y, conf por joint).
        person_id: mantem historico de predicoes por pessoa.

        Retorna {"label", "probs", "alert", "n_valid_frames"} ou None se a janela
        tiver menos de 20% de frames com os keypoints obrigatorios.
        """
        feats, n_valid = self._build_window(window)
        if feats is None:
            return None

        x = feats.reshape(1, -1, FEATURE_DIM).astype(np.float32)
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
        """Limpa historico (chamar quando a pessoa sai de cena)."""
        self._pred_hist.pop(person_id, None)

    # -------------------------------------------------------------------

    def _build_window(self, window: np.ndarray) -> tuple[np.ndarray | None, int]:
        """(T, 51) pixels brutos -> (T, 37), ou (None, n) se houver poucos frames."""
        kp_full = np.asarray(window, dtype=np.float32).reshape(-1, 17, 3)
        kp_xy = kp_full[:, :, :2]
        kp_conf = kp_full[:, :, 2]

        frame_valid = np.all(kp_conf[:, self._REQUIRED_JOINTS] > 0, axis=1)
        n_valid = int(frame_valid.sum())
        if frame_valid.mean() < self._MIN_VALID_FRAC:
            return None, n_valid

        # build_features aplica a mesma regra dos 4 joints obrigatorios e calcula
        # a escala de referencia da velocidade sobre ESTA janela — igual ao treino.
        return build_features(kp_xy, valid_from_conf(kp_conf)), n_valid

    def _check_alert(self, person_id: int, current: int) -> bool:
        hist = list(self._pred_hist[person_id])
        return (
            len(hist) >= self.ALERT_PREDS_FALL
            and all(p == current for p in hist[-self.ALERT_PREDS_FALL:])
        )
