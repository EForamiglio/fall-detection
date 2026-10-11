"""Carrega e prepara os takes capturados pelo app `keypoint_extraction`.

Responsabilidade única: descobrir takes, resolver qual track de pessoa usar,
reamostrar 30 -> 10 fps, aplicar padding e janelar. Nenhuma dependência de
modelo — quem consome isto é `eval_captures.py`.

IMPORTANTE — execute com o venv de `modelos`:
    ../modelos/venv/Scripts/python.exe eval_captures.py

`resample_indices` e `pad_to_window` são importados de `modelos/data_prep` de
propósito, em vez de reimplementados, para garantir comportamento idêntico ao
pipeline de treino. Esses módulos importam pandas/sklearn, que não estão no
requirements.txt deste bench.

Duas armadilhas dos dados desta campanha, tratadas aqui:

* `n_frames` do manifest é inútil (vale 1112 para um take cujo array tem 0
  frames — é comprimento de sessão, não do take). O T real vem sempre de
  `len(np.load(...))`.
* `px_path`/`lbl_path` do manifest são caminhos absolutos de OUTRA máquina.
  Os arquivos são localizados por convenção de pasta; essas colunas nunca são
  lidas.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
MODELOS_DATA_PREP = BENCH_DIR.parent / "modelos" / "data_prep"

# O dir do bench entra ANTES de modelos/data_prep para que `pose_features`
# resolva deterministicamente na cópia do bench (hoje são byte-idênticas).
for path in (str(BENCH_DIR), str(MODELOS_DATA_PREP)):
    if path not in sys.path:
        sys.path.insert(0, path)

from build_heldout_v2 import pad_to_window          # noqa: E402
from build_windows_v2 import WINDOW_SIZE, resample_indices  # noqa: E402

DEFAULT_DATA_DIR = Path(r"D:\Erik\pessoal\material faculdade\PFC\captura pontos\data")

REQUIRED_JOINTS = [5, 6, 11, 12]      # ombros e quadris (COCO 17)
EXPECTED_SRC_FPS = 30.0

# Cenários em que 2 pessoas em quadro são legítimas — nunca fundir tracks aqui.
MULTI_PERSON_SCENARIOS = {"abracando", "duas_pessoas_andando"}

# Abaixo desta sobreposição temporal entre os frames válidos dos dois maiores
# tracks, tratamos como troca de ID do tracker (mesma pessoa partida em dois
# tracks) e não como duas pessoas distintas.
DISJOINT_OVERLAP_MAX = 0.15


@dataclass
class TrackCandidate:
    person_idx: int
    clip_id: str
    px: np.ndarray           # (T, 17, 3) pixels
    labels: np.ndarray       # (T,)
    valid: np.ndarray        # (T,) bool — os 4 joints do tronco presentes

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())


@dataclass
class PreparedTake:
    take_id: str
    scenario_id: str
    is_fall: bool
    decision: str            # single | dominant | dominant_non_p0 | merged_disjoint
    chosen_person_idx: int
    n_tracks: int
    n_valid_chosen: int
    runner_up_person_idx: int | None
    runner_up_n_valid: int | None
    overlap_frac: float | None
    t30: int                 # frames antes da reamostragem
    t10: int                 # frames após reamostragem, antes do padding
    n_pad: int
    px: np.ndarray           # (T, 17, 3) já reamostrado e com padding
    labels: np.ndarray       # (T,)

    def windows(self, stride: int = 1) -> list[tuple[int, np.ndarray]]:
        """Janelas (start, (WINDOW_SIZE, 51)) prontas para os classificadores.

        Stride 1 por padrão: stride 3 e 5 são subconjuntos exatos dos starts,
        então uma única passada de inferência serve os três regimes.
        """
        out = []
        for start in range(0, len(self.px) - WINDOW_SIZE + 1, stride):
            window = self.px[start:start + WINDOW_SIZE]
            out.append((start, window.reshape(WINDOW_SIZE, 51)))
        return out


@dataclass
class RawTake:
    """Take resolvido (track escolhido/fundido), na taxa NATIVA de captura (30fps).

    Ponto de extensão para qualquer modelo com convenção própria de fps/janela
    — quem precisa de 10fps (GRU) ou 12fps (modelo matemático) reamostra a
    partir daqui. Nunca reamostrar em cadeia (30->10->12): sempre a partir
    desta sequência nativa, para não compor dois arredondamentos.
    """
    take_id: str
    scenario_id: str
    is_fall: bool
    decision: str
    chosen_person_idx: int
    n_tracks: int
    n_valid_chosen: int
    runner_up_person_idx: int | None
    runner_up_n_valid: int | None
    overlap_frac: float | None
    px: np.ndarray            # (T30, 17, 3) pixels, taxa nativa
    labels: np.ndarray        # (T30,)


@dataclass
class Inventory:
    takes: list[PreparedTake] = field(default_factory=list)
    missing_takes: list[str] = field(default_factory=list)
    missing_clips: list[str] = field(default_factory=list)


def _valid_mask(px: np.ndarray) -> np.ndarray:
    return (px[:, REQUIRED_JOINTS, 2] > 0).all(axis=1)


def _load_manifest(data_dir: Path) -> list[dict]:
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.csv não encontrado em {data_dir}")
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        fps = float(row["src_fps"])
        if abs(fps - EXPECTED_SRC_FPS) > 0.01:
            raise SystemExit(
                f"{row['clip_id']}: src_fps={fps}, esperado {EXPECTED_SRC_FPS}. "
                "A reamostragem 30->10fps assume 30fps; revise antes de seguir."
            )
    return rows


def _load_track(data_dir: Path, row: dict) -> TrackCandidate | None:
    """Carrega um track pela convenção de pasta. None se os arquivos não existem."""
    base = data_dir / "captures" / row["scenario_id"]
    px_path = base / f"{row['clip_id']}.npy"
    lbl_path = base / f"{row['clip_id']}_labels.npy"
    if not px_path.exists() or not lbl_path.exists():
        return None

    px = np.load(px_path)
    labels = np.load(lbl_path)
    n = min(len(px), len(labels))
    px, labels = px[:n], labels[:n]
    if n == 0:
        return None

    return TrackCandidate(
        person_idx=int(row["person_idx"]),
        clip_id=row["clip_id"],
        px=px,
        labels=labels,
        valid=_valid_mask(px),
    )


def _overlap_frac(a: TrackCandidate, b: TrackCandidate) -> float:
    """Fração de sobreposição temporal entre os frames válidos de dois tracks."""
    n = min(len(a.valid), len(b.valid))
    denom = min(a.n_valid, b.n_valid)
    if denom == 0:
        return 0.0
    return float((a.valid[:n] & b.valid[:n]).sum()) / denom


def _merge_tracks(dominant: TrackCandidate, other: TrackCandidate) -> np.ndarray:
    """Funde dois tracks frame a frame: onde só o `other` é válido, usa o `other`.

    Usado quando os dois tracks são a mesma pessoa partida por uma troca de ID —
    escolher só o dominante descartaria metade do evento.
    """
    n = min(len(dominant.px), len(other.px))
    merged = dominant.px[:n].copy()
    take_other = (~dominant.valid[:n]) & other.valid[:n]
    merged[take_other] = other.px[:n][take_other]
    return merged


def _resolve_track(
    scenario_id: str, candidates: list[TrackCandidate], merge_disjoint: bool,
) -> tuple[np.ndarray, np.ndarray, str, TrackCandidate, TrackCandidate | None, float | None]:
    """Decide o que usar como sequência do take.

    Ordena por (-n_valid, person_idx): o person_idx como segundo critério torna
    a escolha determinística e resolve empates sem regra ad hoc.
    """
    ordered = sorted(candidates, key=lambda c: (-c.n_valid, c.person_idx))
    chosen = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    overlap = _overlap_frac(chosen, runner_up) if runner_up else None

    if (
        merge_disjoint
        and runner_up is not None
        and scenario_id not in MULTI_PERSON_SCENARIOS
        and overlap is not None
        and overlap < DISJOINT_OVERLAP_MAX
    ):
        px = _merge_tracks(chosen, runner_up)
        labels = chosen.labels[:len(px)]
        return px, labels, "merged_disjoint", chosen, runner_up, overlap

    if len(ordered) == 1:
        decision = "single"
    elif chosen.person_idx == 0:
        decision = "dominant"
    else:
        decision = "dominant_non_p0"

    return chosen.px, chosen.labels, decision, chosen, runner_up, overlap


def resolve_raw_takes(
    data_dir: Path = DEFAULT_DATA_DIR, merge_disjoint: bool = True,
) -> tuple[list[RawTake], list[str], list[str]]:
    """Descobre e resolve todos os takes, na taxa NATIVA de captura (30fps).

    Etapa comum a qualquer modelo — seleção/merge de track acontece uma vez só
    aqui. Retorna (takes, missing_takes, missing_clips).
    """
    rows = _load_manifest(data_dir)

    by_take: dict[str, list[dict]] = {}
    for row in rows:
        by_take.setdefault(row["group_id"], []).append(row)

    raw_takes: list[RawTake] = []
    missing_takes: list[str] = []
    missing_clips: list[str] = []

    for take_id in sorted(by_take):
        take_rows = by_take[take_id]
        candidates = []
        for row in take_rows:
            track = _load_track(data_dir, row)
            if track is None:
                missing_clips.append(row["clip_id"])
            else:
                candidates.append(track)

        if not candidates:
            missing_takes.append(take_id)
            continue

        scenario_id = take_rows[0]["scenario_id"]
        is_fall = take_rows[0]["is_fall"].strip().lower() == "true"

        px, labels, decision, chosen, runner_up, overlap = _resolve_track(
            scenario_id, candidates, merge_disjoint,
        )

        # Rótulo é constante por clipe nesta campanha; conferir coerência com o
        # manifest evita avaliar contra ground truth errado.
        expected_label = 2 if is_fall else 0
        observed = set(np.unique(labels).tolist())
        if observed != {expected_label}:
            raise SystemExit(
                f"{take_id}: labels {sorted(observed)} incoerentes com "
                f"is_fall={is_fall} (esperado {{{expected_label}}})."
            )

        raw_takes.append(RawTake(
            take_id=take_id,
            scenario_id=scenario_id,
            is_fall=is_fall,
            decision=decision,
            chosen_person_idx=chosen.person_idx,
            n_tracks=len(candidates),
            n_valid_chosen=chosen.n_valid,
            runner_up_person_idx=runner_up.person_idx if runner_up else None,
            runner_up_n_valid=runner_up.n_valid if runner_up else None,
            overlap_frac=overlap,
            px=px,
            labels=labels,
        ))

    return raw_takes, missing_takes, missing_clips


def load_inventory(
    data_dir: Path = DEFAULT_DATA_DIR, merge_disjoint: bool = True,
) -> Inventory:
    """Descobre e prepara todos os takes avaliáveis, na convenção do GRU
    (reamostragem para 10fps + padding para 20 frames)."""
    raw_takes, missing_takes, missing_clips = resolve_raw_takes(data_dir, merge_disjoint)

    inventory = Inventory(missing_takes=missing_takes, missing_clips=missing_clips)
    for raw in raw_takes:
        t30 = len(raw.px)
        idx = resample_indices(t30, EXPECTED_SRC_FPS)
        px, labels = raw.px[idx], raw.labels[idx]
        t10 = len(px)
        n_pad = max(0, WINDOW_SIZE - t10)
        if n_pad:
            px, labels = pad_to_window(px, labels)

        inventory.takes.append(PreparedTake(
            take_id=raw.take_id,
            scenario_id=raw.scenario_id,
            is_fall=raw.is_fall,
            decision=raw.decision,
            chosen_person_idx=raw.chosen_person_idx,
            n_tracks=raw.n_tracks,
            n_valid_chosen=raw.n_valid_chosen,
            runner_up_person_idx=raw.runner_up_person_idx,
            runner_up_n_valid=raw.runner_up_n_valid,
            overlap_frac=raw.overlap_frac,
            t30=t30,
            t10=t10,
            n_pad=n_pad,
            px=px,
            labels=labels,
        ))

    return inventory


TRACK_SELECTION_FIELDS = [
    "take_id", "scenario_id", "is_fall", "n_tracks", "chosen_person_idx",
    "chosen_n_valid", "t30", "t10", "n_pad", "runner_up_person_idx",
    "runner_up_n_valid", "overlap_frac", "decision",
]


def write_track_selection(inventory: Inventory, path: Path) -> None:
    """Relatório de auditoria: como cada take foi resolvido."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRACK_SELECTION_FIELDS)
        writer.writeheader()
        for take in inventory.takes:
            writer.writerow({
                "take_id": take.take_id,
                "scenario_id": take.scenario_id,
                "is_fall": take.is_fall,
                "n_tracks": take.n_tracks,
                "chosen_person_idx": take.chosen_person_idx,
                "chosen_n_valid": take.n_valid_chosen,
                "t30": take.t30,
                "t10": take.t10,
                "n_pad": take.n_pad,
                "runner_up_person_idx": take.runner_up_person_idx
                if take.runner_up_person_idx is not None else "",
                "runner_up_n_valid": take.runner_up_n_valid
                if take.runner_up_n_valid is not None else "",
                "overlap_frac": f"{take.overlap_frac:.3f}"
                if take.overlap_frac is not None else "",
                "decision": take.decision,
            })
        for take_id in inventory.missing_takes:
            writer.writerow({"take_id": take_id, "decision": "missing"})
