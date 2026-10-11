"""
Estado em tempo de execução por pessoa rastreada.
Centraliza Kalman, escala, PCA e detector de queda.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from .capture_constants import MAX_MISSED_FRAMES
from .fall_detector import FallDetector

if TYPE_CHECKING:
    from .kalman_filter import KalmanPointTracker


@dataclass
class PersonRuntimeState:
    """Características mantidas em runtime para um person_id."""

    fall_detector: FallDetector = field(default_factory=FallDetector)
    scale_ema: float | None = None
    pca_angle: float | None = None
    kalman_trackers: dict[int, KalmanPointTracker] = field(default_factory=dict)


class PersonRuntimeStore:
    """
    Store por person_id do estado volátil da captura.
    """

    def __init__(self) -> None:
        self._people: dict[int, PersonRuntimeState] = {}

    def get_or_create(self, person_id: int) -> PersonRuntimeState:
        if person_id not in self._people:
            self._people[person_id] = PersonRuntimeState()
        return self._people[person_id]

    def get(self, person_id: int) -> PersonRuntimeState | None:
        return self._people.get(person_id)

    def cleanup(self, active_person_ids: set[int]) -> None:
        """Remove estado de IDs fora de cena e trackers Kalman ociosos demais."""
        stale_people = [pid for pid in self._people if pid not in active_person_ids]
        for pid in stale_people:
            del self._people[pid]

        for state in self._people.values():
            stale_kpts = [
                kpt_idx
                for kpt_idx, tracker in state.kalman_trackers.items()
                if tracker.missed_frames > MAX_MISSED_FRAMES
            ]
            for kpt_idx in stale_kpts:
                del state.kalman_trackers[kpt_idx]


@lru_cache
def get_person_runtime_store() -> PersonRuntimeStore:
    """Retorna o store singleton de estado por pessoa."""
    return PersonRuntimeStore()
