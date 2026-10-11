"""Analisa takes gravados pela versão do `keypoint_extraction` que grava
timestamps por frame (`<clip_id>_t.npy`), rodando os GRUs (34f e 37f) e o modelo
matemático com reamostragem POR TEMPO — não por índice de frame.

Por que por tempo: o app grava um frame por inferência do YOLO, então a taxa real
depende da máquina e da iluminação (medido: 12 a 18 fps, contra os 30 que o
driver reporta). Reamostrar "de 30 para 10 fps" pegando 1 de cada 3 frames
assumiria um intervalo de 0,10 s entre frames escolhidos quando o real seria de
0,17 a 0,25 s — e as features de velocidade do GRU de 37 entradas ficariam
infladas na mesma proporção.

Uso:
    ../modelos/venv/Scripts/python.exe analyze_new_takes.py [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

from captures_dataset import WINDOW_SIZE, resample_indices          # noqa: E402
from eval_captures import build_models                              # noqa: E402
import pose_features                                                # noqa: E402
from math_model_adapter import reset_state, run_take                # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_DIR = BENCH_DIR.parent / "keypoint_extraction" / "data" / "teste_usb2"
OUT_DIR = BENCH_DIR / "output" / "new_takes"

TORSO = [5, 6, 11, 12]
GRU_FPS, GRU_STRIDE_DEPLOY = 10.0, 5           # GRU_INTERVAL=0.5s a 10fps
MATH_FPS, MATH_WINDOW = 12.0, 30                # config de deploy da vigia-fall
SCORE_HIGH, SCORE_LOW = 0.65, 0.35              # FallDetectorConfig


def resample_by_time(px: np.ndarray, t: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Para cada instante k/fps, escolhe o frame gravado mais próximo no tempo."""
    if len(t) < 2:
        return px, t
    targets = np.arange(int(np.floor(t[-1] * fps)) + 1) / fps
    idx = np.clip(np.searchsorted(t, targets), 1, len(t) - 1)
    pick_left = (targets - t[idx - 1]) < (t[idx] - targets)
    idx = np.where(pick_left, idx - 1, idx)
    return px[idx], targets


def trunk_series(px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ângulo do tronco em relação à vertical (graus) e altura do quadril (px)."""
    valid = (px[:, TORSO, 2] > 0).all(axis=1)
    sho = (px[:, 5, :2] + px[:, 6, :2]) / 2
    hip = (px[:, 11, :2] + px[:, 12, :2]) / 2
    d = sho - hip
    angle = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1]) + 1e-9))
    return np.where(valid, angle, np.nan), np.where(valid, hip[:, 1], np.nan)


def quality(px: np.ndarray, t: np.ndarray) -> dict:
    valid = (px[:, TORSO, 2] > 0).all(axis=1)
    dt = np.diff(t)
    # maior sequência de frames sem tronco
    longest, run = 0, 0
    for v in valid:
        run = 0 if v else run + 1
        longest = max(longest, run)
    conf = px[:, :, 2][px[:, :, 2] > 0]
    return {
        "frames": len(px), "duration": float(t[-1]), "fps": (len(t) - 1) / t[-1],
        "dt_mean_ms": 1000 * dt.mean(), "dt_p95_ms": 1000 * np.percentile(dt, 95),
        "dt_max_ms": 1000 * dt.max(), "torso_valid": float(valid.mean()),
        "longest_gap_s": longest * dt.mean(), "kpt_conf": float(conf.mean()),
    }


def gru_pass(model, px10: np.ndarray, times10: np.ndarray) -> dict:
    model.reset_person(0)
    starts, probs, labels = [], [], []
    for s in range(0, len(px10) - WINDOW_SIZE + 1):
        pred = model.predict(px10[s:s + WINDOW_SIZE].reshape(WINDOW_SIZE, 51), person_id=0)
        starts.append(s)
        if pred is None:
            probs.append(np.nan); labels.append("INVALID")
        else:
            probs.append(pred["probs"][1]); labels.append(pred["label"])
    return {"t_end": np.array([times10[s + WINDOW_SIZE - 1] for s in starts]),
            "starts": np.array(starts), "prob": np.array(probs), "label": labels}


def run_of_fall(labels: list[str], stride: int, n: int) -> int | None:
    """Índice (na lista de stride) em que se completam `n` FALL consecutivos."""
    sub = [l for l in labels[::stride]]
    streak = 0
    for i, l in enumerate(sub):
        if l == "INVALID":
            continue
        streak = streak + 1 if l == "FALL" else 0
        if streak >= n:
            return i * stride
    return None


def mean_abs_vy(px: np.ndarray) -> float:
    vals = []
    for s in range(0, len(px) - WINDOW_SIZE + 1, 5):
        w = px[s:s + WINDOW_SIZE]
        f = pose_features.build_features(w[:, :, :2], pose_features.valid_from_conf(w[:, :, 2]))
        vals.append(np.abs(f[:, 35]).mean())
    return float(np.mean(vals)) if vals else float("nan")


def analyze_clip(row: dict, data_dir: Path, models: dict) -> None:
    base = data_dir / "captures" / row["scenario_id"] / row["clip_id"]
    px = np.load(f"{base}.npy")
    t = np.load(f"{base}_t.npy")
    is_fall = row["is_fall"].strip().lower() == "true"
    q = quality(px, t)

    print("\n" + "=" * 78)
    print(f"{row['clip_id']}   ({'QUEDA' if is_fall else 'ADL'})")
    print("=" * 78)
    print(f"  qualidade: {q['frames']} frames em {q['duration']:.1f}s → {q['fps']:.1f} fps medidos "
          f"(manifesto: {row['src_fps']})")
    print(f"             intervalo entre frames: média {q['dt_mean_ms']:.0f} ms, "
          f"p95 {q['dt_p95_ms']:.0f} ms, máx {q['dt_max_ms']:.0f} ms")
    print(f"             tronco válido em {100*q['torso_valid']:.0f}% dos frames, "
          f"maior lacuna {q['longest_gap_s']:.1f}s, conf. média dos keypoints {q['kpt_conf']:.2f}")

    # ── GRU ──
    px10, t10 = resample_by_time(px, t, GRU_FPS)
    print(f"\n  GRU — reamostrado por tempo para {GRU_FPS:.0f} fps: {len(px10)} frames → "
          f"{max(0, len(px10) - WINDOW_SIZE + 1)} janelas (stride 1), "
          f"{len(range(0, max(0, len(px10) - WINDOW_SIZE + 1), GRU_STRIDE_DEPLOY))} no stride do deploy")
    gru = {}
    for key, name in (("34f", "GRU 34 features"), ("37f", "GRU 37 features")):
        res = gru_pass(models[key], px10, t10)
        gru[key] = res
        n_fall = sum(1 for l in res["label"] if l == "FALL")
        first = run_of_fall(res["label"], 1, 1)
        c1 = run_of_fall(res["label"], GRU_STRIDE_DEPLOY, 1)
        c2 = run_of_fall(res["label"], GRU_STRIDE_DEPLOY, 2)
        fmt = lambda i: "—" if i is None else f"{res['t_end'][i]:.1f}s"
        print(f"    {name}: {n_fall}/{len(res['label'])} janelas FALL ({100*n_fall/len(res['label']):.0f}%)  "
              f"| 1ª FALL em {fmt(first)}  | confirm_n=1 (deploy): {fmt(c1)}  | confirm_n=2 (deploy): {fmt(c2)}")

    # sanidade de velocidade: por tempo (correto) vs por índice assumindo 30 fps (método antigo)
    old_idx = resample_indices(len(px), 30.0)
    v_time, v_idx = mean_abs_vy(px10), mean_abs_vy(px[old_idx])
    print(f"    |v_y| médio das janelas: por tempo = {v_time:.4f} | por índice assumindo 30fps = {v_idx:.4f} "
          f"(inflação {v_idx / v_time:.2f}x)")

    # ── Matemático ──
    px12, t12 = resample_by_time(px, t, MATH_FPS)
    reset_state()
    dec = run_take(px12, target_fps=MATH_FPS, window_size=MATH_WINDOW)
    reset_state()
    scores = np.array([d.score for d in dec if d.score is not None])
    states = [d.label for d in dec]
    first_state = {s: next((d.frame_idx / MATH_FPS for d in dec if d.label == s), None)
                   for s in ("SUSPECT", "FALL")}
    above = int((scores >= SCORE_HIGH).sum()) if len(scores) else 0
    print(f"\n  Matemático — {MATH_FPS:.0f} fps, janela {MATH_WINDOW}: {len(px12)} frames → {len(dec)} decisões")
    if len(scores):
        print(f"    score: média {scores.mean():.2f}, máx {scores.max():.2f}, "
              f"{above}/{len(scores)} decisões ≥ {SCORE_HIGH}")
    fmt_t = lambda v: "—" if v is None else f"{v:.1f}s"
    print(f"    estados atingidos: {sorted(set(states))}  | 1º SUSPECT: "
          f"{fmt_t(first_state['SUSPECT'])}  | 1º FALL: {fmt_t(first_state['FALL'])}")

    # ── Gráfico ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    angle, hip_y = trunk_series(px)
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(t, angle, color="tab:blue", label="ângulo do tronco (°)")
    ax[0].set_ylabel("tronco vs vertical (°)"); ax[0].set_ylim(0, 95); ax[0].grid(alpha=.3)
    a2 = ax[0].twinx(); a2.plot(t, hip_y, color="tab:gray", alpha=.6)
    a2.set_ylabel("altura do quadril (px, y↓)"); a2.invert_yaxis()
    ax[0].set_title(f"{row['clip_id']}  —  {q['duration']:.1f}s, {q['fps']:.1f} fps medidos")
    ax[1].plot(gru["34f"]["t_end"], gru["34f"]["prob"], label="GRU 34f", color="tab:green")
    ax[1].plot(gru["37f"]["t_end"], gru["37f"]["prob"], label="GRU 37f", color="tab:red")
    ax[1].axhline(0.5, color="k", ls="--", lw=.7); ax[1].set_ylabel("P(queda)")
    ax[1].set_ylim(-0.02, 1.02); ax[1].legend(loc="upper right"); ax[1].grid(alpha=.3)
    if len(dec):
        ax[2].plot([d.frame_idx / MATH_FPS for d in dec], [d.score for d in dec], color="tab:purple")
        fall_t = [d.frame_idx / MATH_FPS for d in dec if d.label == "FALL"]
        sus_t = [d.frame_idx / MATH_FPS for d in dec if d.label == "SUSPECT"]
        ax[2].scatter(sus_t, [0.02] * len(sus_t), color="orange", s=10, label="SUSPECT")
        ax[2].scatter(fall_t, [0.02] * len(fall_t), color="red", s=10, label="FALL")
    ax[2].axhline(SCORE_HIGH, color="r", ls="--", lw=.7); ax[2].axhline(SCORE_LOW, color="g", ls="--", lw=.7)
    ax[2].set_ylabel("score matemático"); ax[2].set_xlabel("tempo (s)")
    ax[2].set_ylim(-0.02, 1.02); ax[2].grid(alpha=.3)
    if len(dec):
        ax[2].legend(loc="upper right")
    fig.tight_layout()
    out = OUT_DIR / f"{row['clip_id']}.png"
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"\n  gráfico: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    manifest = args.data_dir / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"{len(rows)} clipe(s) em {args.data_dir}")

    models = build_models()
    for row in rows:
        analyze_clip(row, args.data_dir, models)


if __name__ == "__main__":
    main()
