"""Métricas preliminares dos modelos sobre os takes de teste (n muito pequeno).

Dois níveis: por clipe (1 decisão por take) e por janela. O rótulo por janela segue
a regra do treino: início da queda = tronco > 60° por 2+ frames; janelas que
terminam a partir de 1 s antes disso são positivas (PREFALL+FALLEN = FALL).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analyze_new_takes import (DEFAULT_DIR, GRU_FPS, GRU_STRIDE_DEPLOY, MATH_FPS,   # noqa: E402
                               MATH_WINDOW, SCORE_HIGH, gru_pass, resample_by_time,
                               run_of_fall, trunk_series)
from eval_captures import build_models                                              # noqa: E402
from math_model_adapter import reset_state, run_take                                # noqa: E402

PRE_FALL_S = 1.0


def metrics(tp, fn, fp, tn) -> dict:
    r = lambda a, b: a / b if b else float("nan")
    p, rec = r(tp, tp + fp), r(tp, tp + fn)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "acc": r(tp + tn, tp + fn + fp + tn),
            "prec": p, "rec": rec, "spec": r(tn, tn + fp),
            "f1": 2 * p * rec / (p + rec) if p == p and rec == rec and p + rec > 0 else float("nan")}


def onset_fallen(t: np.ndarray, px: np.ndarray) -> float | None:
    angle, _ = trunk_series(px)
    ok = angle > 60
    for i in range(len(ok) - 1):
        if ok[i] and ok[i + 1]:
            return float(t[i])
    return None


def row(name, m) -> str:
    f = lambda v: "  n/a " if v != v else f"{100*v:5.1f}%"
    return (f"  {name:<34} VP={m['tp']:<3} FN={m['fn']:<3} FP={m['fp']:<3} VN={m['tn']:<4}"
            f" acc={f(m['acc'])} prec={f(m['prec'])} sens={f(m['rec'])} esp={f(m['spec'])}"
            f" F1={'  n/a ' if m['f1'] != m['f1'] else f'{m['f1']:.3f}'}")


def main() -> None:
    with (DEFAULT_DIR / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    models = build_models()

    clip_dec: dict[str, dict[str, bool]] = {}
    win: dict[str, list[tuple[bool, bool]]] = {k: [] for k in
          ("GRU 34f", "GRU 37f", "Matemático (estado FALL)", "Matemático (score>=0,65)")}

    for r in rows:
        base = DEFAULT_DIR / "captures" / r["scenario_id"] / r["clip_id"]
        px, t = np.load(f"{base}.npy"), np.load(f"{base}_t.npy")
        is_fall = r["is_fall"].strip().lower() == "true"
        onset = onset_fallen(t, px)
        gt_pos = lambda te: onset is not None and te >= onset - PRE_FALL_S
        print(f"{r['clip_id']}: início da queda (tronco>60°) = "
              f"{'—' if onset is None else f'{onset:.1f}s'}")

        px10, t10 = resample_by_time(px, t, GRU_FPS)
        for key, label in (("34f", "GRU 34f"), ("37f", "GRU 37f")):
            res = gru_pass(models[key], px10, t10)
            for te, lab in zip(res["t_end"], res["label"]):
                if lab != "INVALID":
                    win[label].append((gt_pos(te), lab == "FALL"))
            clip_dec.setdefault(f"{label} confirm_n=1", {})[r["clip_id"]] = \
                run_of_fall(res["label"], GRU_STRIDE_DEPLOY, 1) is not None
            clip_dec.setdefault(f"{label} confirm_n=2 (deploy)", {})[r["clip_id"]] = \
                run_of_fall(res["label"], GRU_STRIDE_DEPLOY, 2) is not None

        px12, _ = resample_by_time(px, t, MATH_FPS)
        reset_state(); dec = run_take(px12, MATH_FPS, MATH_WINDOW); reset_state()
        for d in dec:
            te = d.frame_idx / MATH_FPS
            win["Matemático (estado FALL)"].append((gt_pos(te), d.label == "FALL"))
            win["Matemático (score>=0,65)"].append((gt_pos(te), (d.score or 0) >= SCORE_HIGH))
        clip_dec.setdefault("Matemático (máq. de estados)", {})[r["clip_id"]] = \
            any(d.label == "FALL" for d in dec)

    fall_ids = [r["clip_id"] for r in rows if r["is_fall"].strip().lower() == "true"]
    adl_ids = [r["clip_id"] for r in rows if r["is_fall"].strip().lower() != "true"]

    print("\n" + "=" * 100 + f"\nNÍVEL DE CLIPE  ({len(fall_ids)} queda + {len(adl_ids)} ADL — n=1 por classe)\n" + "=" * 100)
    for name, dec in clip_dec.items():
        tp = sum(dec[c] for c in fall_ids); fn = len(fall_ids) - tp
        fp = sum(dec[c] for c in adl_ids); tn = len(adl_ids) - fp
        print(row(name, metrics(tp, fn, fp, tn)))

    print("\n" + "=" * 100 + "\nNÍVEL DE JANELA  (rótulo pela regra do treino; janelas sobrepostas, não independentes)\n" + "=" * 100)
    for name, items in win.items():
        tp = sum(1 for g, p in items if g and p); fn = sum(1 for g, p in items if g and not p)
        fp = sum(1 for g, p in items if not g and p); tn = sum(1 for g, p in items if not g and not p)
        print(row(name, metrics(tp, fn, fp, tn)))


if __name__ == "__main__":
    main()
