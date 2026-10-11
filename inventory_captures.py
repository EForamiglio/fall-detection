"""Inventário dos takes capturados — checkpoint antes de gerar qualquer métrica.

Existe separado de `eval_captures.py` de propósito: erros de dados precisam
aparecer antes de virarem métrica. Os números aqui são comparados contra o que
foi medido na exploração; se divergirem, a avaliação não deve ser rodada.

Uso:
    ../modelos/venv/Scripts/python.exe inventory_captures.py
    ../modelos/venv/Scripts/python.exe inventory_captures.py --no-merge-disjoint
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from captures_dataset import (
    DEFAULT_DATA_DIR,
    WINDOW_SIZE,
    Inventory,
    load_inventory,
    write_track_selection,
)

OUT_DIR = Path(__file__).resolve().parent / "output" / "captures_eval"

# Valores medidos na exploração; o inventário precisa reproduzi-los.
EXPECTED = {
    "takes": 160,
    "missing_takes": 4,
    "missing_clips": 18,
    "merged_disjoint": 3,
    "dominant_non_p0": 9,
    "padded": 31,
}


def print_inventory(inv: Inventory) -> dict:
    fall = [t for t in inv.takes if t.is_fall]
    adl = [t for t in inv.takes if not t.is_fall]

    print("=" * 74)
    print("INVENTÁRIO")
    print("=" * 74)
    print(f"takes avaliáveis        : {len(inv.takes)}  (FALL {len(fall)} / ADL {len(adl)})")
    print(f"takes sem nenhum clipe  : {len(inv.missing_takes)}  {inv.missing_takes}")
    print(f"clipes do manifest sem .npy: {len(inv.missing_clips)}")

    decisions = Counter(t.decision for t in inv.takes)
    print("\ndecisão de track:")
    for decision, count in sorted(decisions.items()):
        print(f"  {decision:<18} {count:>4}")

    padded = [t for t in inv.takes if t.n_pad > 0]
    padded_fall = sum(1 for t in padded if t.is_fall)
    print(f"\ntakes com padding       : {len(padded)}  "
          f"(FALL {padded_fall}/{len(fall)}, ADL {len(padded) - padded_fall}/{len(adl)})")

    print("\nduração (frames):")
    print(f"  {'classe':<8} {'T30 min/med/max':>20} {'T10 min/med/max':>20}")
    for name, group in [("FALL", fall), ("ADL", adl)]:
        t30 = np.array([t.t30 for t in group])
        t10 = np.array([t.t10 for t in group])
        print(f"  {name:<8} "
              f"{f'{t30.min()} / {int(np.median(t30))} / {t30.max()}':>20} "
              f"{f'{t10.min()} / {int(np.median(t10))} / {t10.max()}':>20}")

    print("\njanelas por take (após padding):")
    print(f"  {'classe':<8} {'stride':>7} {'min':>5} {'mediana':>8} {'max':>5} {'total':>7}")
    for name, group in [("FALL", fall), ("ADL", adl)]:
        for stride in (1, 5):
            counts = np.array([len(t.windows(stride)) for t in group])
            print(f"  {name:<8} {stride:>7} {counts.min():>5} "
                  f"{int(np.median(counts)):>8} {counts.max():>5} {counts.sum():>7}")

    print("\npor cenário:")
    print(f"  {'cenário':<24} {'GT':>5} {'takes':>6} {'jan.(s5)':>9} {'frac_val':>9}")
    by_scenario: dict[str, list] = {}
    for take in inv.takes:
        by_scenario.setdefault(take.scenario_id, []).append(take)
    for scenario in sorted(by_scenario):
        group = by_scenario[scenario]
        gt = "FALL" if group[0].is_fall else "ADL"
        n_win = sum(len(t.windows(5)) for t in group)
        frac = np.mean([t.n_valid_chosen / max(1, t.t30) for t in group])
        print(f"  {scenario:<24} {gt:>5} {len(group):>6} {n_win:>9} {frac:>8.3f}")

    non_trivial = [t for t in inv.takes if t.decision not in ("single", "dominant")]
    if non_trivial:
        print(f"\ntakes desambiguados ({len(non_trivial)}):")
        print(f"  {'take':<32} {'decisão':<18} {'escolhido':>10} {'runner-up':>10} {'sobrep.':>8}")
        for take in sorted(non_trivial, key=lambda t: (t.decision, t.take_id)):
            print(f"  {take.take_id:<32} {take.decision:<18} "
                  f"p{take.chosen_person_idx}({take.n_valid_chosen}){'':>2} "
                  f"p{take.runner_up_person_idx}({take.runner_up_n_valid}){'':>2} "
                  f"{take.overlap_frac:>8.3f}")

    return {
        "takes": len(inv.takes),
        "missing_takes": len(inv.missing_takes),
        "missing_clips": len(inv.missing_clips),
        "merged_disjoint": decisions.get("merged_disjoint", 0),
        "dominant_non_p0": decisions.get("dominant_non_p0", 0),
        "padded": len(padded),
    }


def check_expected(observed: dict) -> bool:
    print("\n" + "=" * 74)
    print("CHECKPOINT — comparação com os valores medidos na exploração")
    print("=" * 74)
    print(f"  {'métrica':<20} {'esperado':>9} {'observado':>10}   ")
    all_ok = True
    for key, expected in EXPECTED.items():
        got = observed[key]
        ok = got == expected
        all_ok &= ok
        print(f"  {key:<20} {expected:>9} {got:>10}   {'OK' if ok else 'DIVERGE'}")
    print()
    print("Inventário reproduz a exploração — seguro seguir para a avaliação."
          if all_ok else
          "DIVERGÊNCIA: não rode a avaliação antes de entender a causa.")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-merge-disjoint", action="store_true",
                        help="Desliga a fusão de tracks disjuntos (para comparação)")
    args = parser.parse_args()

    inv = load_inventory(args.data_dir, merge_disjoint=not args.no_merge_disjoint)
    observed = print_inventory(inv)

    suffix = "_nomerge" if args.no_merge_disjoint else ""
    out_path = OUT_DIR / f"track_selection{suffix}.csv"
    write_track_selection(inv, out_path)
    print(f"\nRelatório de seleção de track: {out_path}")

    if not args.no_merge_disjoint:
        check_expected(observed)


if __name__ == "__main__":
    main()
