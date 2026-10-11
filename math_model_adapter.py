"""Adapta os takes capturados (pixels a 30fps) para o MathFallClassifier.

O modelo matemático tem uma interface fundamentalmente diferente da dos GRUs:
  * GRU: função pura `predict(window_fixa) -> decisão`, sem estado entre chamadas
    (fora o histórico de confirmação, que nós mesmos recalculamos).
  * Matemático: `process(observações_de_1_frame) -> decisões`, com estado
    PERSISTENTE por person_id (Kalman, EMA de escala, ângulo PCA, máquina de
    estados do FallDetector) that vive num singleton de processo
    (`get_person_runtime_store`, decorado com `@lru_cache`).

Isso muda o que precisa ser feito para avaliar corretamente:

1. **Reamostragem própria.** O deploy do modelo matemático usa
   `frame_rate=12` e `slider_window_size=30` (settings.py da vigia-fall) —
   diferente dos 10fps/20 frames do GRU. Reamostramos os takes NATIVOS a
   30fps (via `captures_dataset.resolve_raw_takes`, antes de qualquer
   reamostragem para o GRU) para 12fps com a mesma lógica de vizinho mais
   próximo usada no resto do projeto.

2. **Timestamps sintéticos.** O Kalman e as features de velocidade dependem
   de `timestamp` real (segundos). Como a gravação não carimba tempo por
   frame, geramos `t = i / target_fps` — consistente com o que o pipeline de
   captura real faria a frame_rate constante.

3. **Reset de estado obrigatório entre takes.** Como o runtime store é um
   singleton de processo, sem reset ele vazaria Kalman/EMA/FallDetector de
   um take para o outro (o mesmo `person_id=0` reaparece em todo take).
   `reset_state()` limpa o singleton via `cache_clear()` — equivalente a
   reiniciar o processo.

4. **Cada take pode gerar VÁRIAS decisões**, uma por frame após a janela
   encher (ao contrário do GRU, que dá 1 decisão por janela de 20 frames
   avançando de pouco em pouco). A decisão do take é agregada como "alertou
   em algum momento" — mesma semântica de `detected()` em `eval_captures.py`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from math_model.math_classifier import MathFallClassifier   # noqa: E402
from math_model.person_runtime import get_person_runtime_store  # noqa: E402
from math_model.types import PoseObservation                # noqa: E402

# Defaults de deploy da vigia-fall (shared/settings.py, Settings.from_env):
# frame_rate=12, slider_window_size=30.
DEFAULT_TARGET_FPS = 12.0
DEFAULT_WINDOW_SIZE = 30

MIN_KPT_CONF = 0.3   # capture_constants.MIN_KPT_CONF — mesmo limiar do modelo


def resample_native(px: np.ndarray, labels: np.ndarray, src_fps: float,
                    target_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Reamostragem por vizinho mais próximo, genérica em fps de destino.

    Mesma lógica de `build_windows_v2.resample_indices`, mas parametrizada —
    aquela função tem `DEPLOY_FPS=10.0` fixo no módulo, específico do GRU.
    Reamostrar SEMPRE a partir do array nativo (30fps), nunca em cadeia a
    partir de um array já reamostrado para outro modelo.
    """
    n_src = len(px)
    if n_src == 0 or abs(src_fps - target_fps) < 0.5:
        return px, labels
    n_tgt = max(1, int(round(n_src * target_fps / src_fps)))
    idx = np.clip(np.round(np.linspace(0, n_src - 1, n_tgt)).astype(int), 0, n_src - 1)
    return px[idx], labels[idx]


def reset_state() -> None:
    """Limpa o singleton de estado por pessoa — chamar antes de CADA take.

    Sem isto, Kalman/EMA de escala/ângulo PCA/máquina de estados do
    FallDetector vazam de um take para o outro, porque `person_id=0` se
    repete em todo take e o store é `@lru_cache` (um só por processo).
    """
    get_person_runtime_store.cache_clear()


@dataclass
class MathFrameDecision:
    frame_idx: int
    label: str            # NORMAL | SUSPECT | FALL | FALSE_POSITIVE
    alert: bool
    score: float | None


def run_take(
    px: np.ndarray, target_fps: float = DEFAULT_TARGET_FPS,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> list[MathFrameDecision]:
    """Roda o classificador matemático sobre um take já reamostrado.

    `px`: (T, 17, 3) pixels na taxa `target_fps`, pessoa já selecionada/fundida
    (reaproveita a mesma resolução de track do GRU — ver captures_dataset.py).

    IMPORTANTE: chame `reset_state()` antes desta função para cada take novo.
    """
    classifier = MathFallClassifier(window_size=window_size)
    decisions: list[MathFrameDecision] = []

    for i in range(len(px)):
        kpts = px[i].astype(np.float64).copy()
        # O modelo matemático trata "abaixo do limiar" como ausente (dict sem
        # a chave), igual ao capture real — mas nossos arrays já vêm zerados
        # abaixo do limiar de captura (0.25). Reforçar aqui com o limiar
        # PRÓPRIO do modelo (0.3) evita que um joint "fraco" demais para o
        # modelo matemático, mas válido para o GRU, entre como observação.
        kpts[kpts[:, 2] < MIN_KPT_CONF] = 0.0

        obs = [PoseObservation(person_id=0, keypoints=kpts, timestamp=i / target_fps)]
        frame_decisions = classifier.process(obs)
        for decision in frame_decisions:
            if decision.person_id != 0:
                continue
            decisions.append(MathFrameDecision(
                frame_idx=i,
                label=decision.label,
                alert=decision.alert,
                score=(decision.detail or {}).get("score"),
            ))

    return decisions


def take_detected(decisions: list[MathFrameDecision]) -> bool:
    """O take foi detectado como queda se o FallDetector chegou a FALL."""
    return any(d.label == "FALL" for d in decisions)


def take_alerted(decisions: list[MathFrameDecision]) -> bool:
    """Versão mais estrita: exige a transição de alerta (borda de subida),
    não só ter passado por FALL — é o que o deploy realmente notificaria."""
    return any(d.alert for d in decisions)
