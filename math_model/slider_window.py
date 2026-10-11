"""
Customização de gerenciamento de janelas deslizantes conforme o Id reconhecidos
"""

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Optional


@dataclass
class PersonWindow:
    """
    Janela deslizante destinada a um ID (person)
    """
    window: deque = field(default_factory=deque())
    last_seen: float = 0.0


class SlidingWindowManager:
    """
    Matém uma janela deslizante por ID, juntamente com o release de janelas ociosas
    """

    def __init__(self, window_size: int, stale_timeout: float = 10.0) -> None:
        self.window_size = window_size
        self.stale_timeout = stale_timeout
        self._windows: dict[int, PersonWindow] = {}

    def update(self, frame_result: dict[int, dict[str, Any]]) -> list[int]:
        """
        Insere os dados do frame nas janelas dos IDs presentes.

        Retorna a lista de IDs cuja janela está cheia (prontos p/ extração).
        """
        ready: list[int] = []
    
        for person_id, data in frame_result.items():
            pw = self._windows.get(person_id)

            if pw is None:
                pw = PersonWindow(window=deque(maxlen=self.window_size))
                self._windows[person_id] = pw

            pw.window.append(data)

            timestamp = data["timestamp"]

            pw.last_seen = timestamp

            if len(pw.window) == self.window_size:
                ready.append(person_id)

        # print("Ready IDs antes da limpeza de stale:", ready)
        # print("Windows antes da limpeza de stale:", self._windows)
        # self._cleanup_stale()
        # print("Ready IDs depois da limpeza de stale:", ready)
        # print("Windows depois da limpeza de stale:", self._windows)
        return ready

    def get_window(self, person_id: int) -> Optional[deque]:
        """
        Obtém a janela de um ID
        """
        pw = self._windows.get(person_id)
        return pw.window if pw else None

    def get_fill_status(self) -> dict[int, tuple[int, int]]:
        """Retorna person_id → (preenchimento atual, capacidade)."""
        return {
            person_id: (len(pw.window), self.window_size)
            for person_id, pw in self._windows.items()
        }

    def _cleanup_stale(self) -> None:
        """
        Remove as janelas ociosas
        """
        stale = [
            pid
            for pid, pw in self._windows.items()
            if time.time() - pw.last_seen > self.stale_timeout
        ]
        for pid in stale:
            del self._windows[pid]
