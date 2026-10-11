"""Checkpoint: a janela do modelo matemático sequer enche nos clipes capturados?

Existe separado de qualquer script de métricas de propósito — mesma disciplina
usada para o GRU (inventory_captures.py): medir o orçamento de dados ANTES de
gerar qualquer número, porque já sabemos que os clipes de queda são curtos.

Uso:
    ../modelos/venv/Scripts/python.exe inventory_math.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from captures_dataset import DEFAULT_DATA_DIR, resolve_raw_takes
from math_model_adapter import (
    DEFAULT_TARGET_FPS,
    DEFAULT_WINDOW_SIZE,
    resample_native,
)

EXPECTED_SRC_FPS = 30.0


def main() -> None:
    raw_takes, missing_takes, missing_clips = resolve_raw_takes(DEFAULT_DATA_DIR)
    print(f"Takes resolvidos: {len(raw_takes)}  (missing_takes={len(missing_takes)}, "
          f"missing_clips={len(missing_clips)})")

    fall = [t for t in raw_takes if t.is_fall]
    adl = [t for t in raw_takes if not t.is_fall]

    print(f"\nReamostrando 30fps -> {DEFAULT_TARGET_FPS}fps, "
          f"janela do modelo matemático = {DEFAULT_WINDOW_SIZE} frames "
          f"({DEFAULT_WINDOW_SIZE / DEFAULT_TARGET_FPS:.2f}s)")

    print(f"\n{'classe':<8} {'T30 min/med/max':>20} {'T@12fps min/med/max':>22} "
          f"{'takes que enchem a janela':>28}")
    for name, group in [("FALL", fall), ("ADL", adl)]:
        t30 = np.array([len(t.px) for t in group])
        t_target = np.array([
            len(resample_native(t.px, t.labels, EXPECTED_SRC_FPS, DEFAULT_TARGET_FPS)[0])
            for t in group
        ])
        n_fill = int((t_target >= DEFAULT_WINDOW_SIZE).sum())
        print(f"{name:<8} "
              f"{f'{t30.min()} / {int(np.median(t30))} / {t30.max()}':>20} "
              f"{f'{t_target.min()} / {int(np.median(t_target))} / {t_target.max()}':>22} "
              f"{f'{n_fill}/{len(group)}':>28}")

    print("\nPor cenário:")
    print(f"  {'cenário':<24} {'GT':>5} {'takes':>6} {'enche a janela':>15} "
          f"{'T@12fps mediana':>16}")
    by_scenario: dict[str, list] = {}
    for take in raw_takes:
        by_scenario.setdefault(take.scenario_id, []).append(take)
    for scenario in sorted(by_scenario):
        group = by_scenario[scenario]
        gt = "FALL" if group[0].is_fall else "ADL"
        t_target = np.array([
            len(resample_native(t.px, t.labels, EXPECTED_SRC_FPS, DEFAULT_TARGET_FPS)[0])
            for t in group
        ])
        n_fill = int((t_target >= DEFAULT_WINDOW_SIZE).sum())
        print(f"  {scenario:<24} {gt:>5} {len(group):>6} "
              f"{f'{n_fill}/{len(group)}':>15} {int(np.median(t_target)):>16}")

    n_fall_fill = sum(
        1 for t in fall
        if len(resample_native(t.px, t.labels, EXPECTED_SRC_FPS, DEFAULT_TARGET_FPS)[0])
        >= DEFAULT_WINDOW_SIZE
    )
    print(f"\n{'='*70}")
    if n_fall_fill < len(fall) * 0.5:
        print(f"CONFIRMADO: só {n_fall_fill}/{len(fall)} takes de queda enchem a janela "
              f"de {DEFAULT_WINDOW_SIZE} frames na configuração de deploy "
              f"({DEFAULT_TARGET_FPS}fps). A maioria dos clipes de queda é curta demais "
              "para o modelo matemático sequer produzir uma decisão nessa configuração.")
    else:
        print(f"{n_fall_fill}/{len(fall)} takes de queda enchem a janela — orçamento "
              "razoável para avaliar na configuração de deploy.")
    print("=" * 70)


if __name__ == "__main__":
    main()
