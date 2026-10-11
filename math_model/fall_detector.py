"""
Detector de queda com persistência e histerese por pessoa.
O ciclo de vida por person_id fica no PersonRuntimeStore.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class FallState(Enum):
    """
    Estado do detector de queda
    """

    NORMAL = 0
    SUSPECT = 1
    FALL = 2
    FALSE_POSITIVE = 3


@dataclass(frozen=True)
class FallDetectorConfig:
    """
    Configuração do detector de queda
    """

    threshold_high: float = 0.65
    threshold_low: float = 0.35
    persistence_frames: int = 5
    score_history_size: int = 30
    suspect_max_frames: int = 30
    cooldown_frames: int = 30


@dataclass
class FallDetectorState:
    """
    Estado persistente do detector (uma instância por pessoa no runtime store)
    """

    consecutive_high: int = 0
    state: FallState = FallState.NORMAL
    score_history: deque = field(default_factory=lambda: deque(maxlen=30))
    frames_in_current_state: int = 0
    suspect_started_at: float | None = None


class FallDetector:
    """
    Detector de queda
    """

    def __init__(self, config: FallDetectorConfig | None = None):
        self.config = config or FallDetectorConfig()
        self._state = FallDetectorState(
            score_history=deque(maxlen=self.config.score_history_size)
        )

    @property
    def state(self) -> FallState:
        return self._state.state

    def update(self, score: float, timestamp: float) -> FallState:
        """
        Atualiza o contador de persistência e a máquina de estados.
        """
        s = self._state
        s.score_history.append((timestamp, score))
        s.frames_in_current_state += 1

        if s.state == FallState.NORMAL:
            self._update_persistence_counter(score)

            if s.consecutive_high >= self.config.persistence_frames:
                s.state = FallState.SUSPECT
                s.suspect_started_at = timestamp
                s.frames_in_current_state = 0
                s.consecutive_high = 0

        elif s.state == FallState.SUSPECT:
            # Pós-impacto o score costuma cair da zona alta para a morna; isso
            # ainda é "não recuperou". Só score < low aborta para NORMAL.
            if score < self.config.threshold_low:
                s.state = FallState.NORMAL
                s.consecutive_high = 0
                s.frames_in_current_state = 0
                s.suspect_started_at = None
            else:
                s.consecutive_high += 1
                if s.consecutive_high >= self.config.persistence_frames:
                    s.state = FallState.FALL
                    s.frames_in_current_state = 0
                    s.consecutive_high = 0
                    s.suspect_started_at = None
                elif s.frames_in_current_state >= self.config.suspect_max_frames:
                    s.state = FallState.FALSE_POSITIVE
                    s.frames_in_current_state = 0
                    s.consecutive_high = 0
                    s.suspect_started_at = None

        elif s.state in (FallState.FALL, FallState.FALSE_POSITIVE):
            if s.frames_in_current_state >= self.config.cooldown_frames:
                s.state = FallState.NORMAL
                s.consecutive_high = 0
                s.frames_in_current_state = 0

        return s.state

    def _update_persistence_counter(self, score: float) -> None:
        """Histerese: só zera o contador se o score cair abaixo de threshold_low."""
        s = self._state
        if score >= self.config.threshold_high:
            s.consecutive_high += 1
        elif score >= self.config.threshold_low:
            pass  # zona morna: mantém o contador
        else:
            s.consecutive_high = 0

    def resolve_suspicion(self, confirmed: bool) -> FallState:
        """Chamado pela Etapa 9 ao terminar a checagem de imobilidade."""
        s = self._state
        s.state = FallState.FALL if confirmed else FallState.FALSE_POSITIVE
        s.frames_in_current_state = 0
        return s.state

    def reset(self) -> None:
        """Reinicia o detector (ex.: tracking perdido)."""
        self._state = FallDetectorState(
            score_history=deque(maxlen=self.config.score_history_size)
        )