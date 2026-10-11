"""Benchmark de latência de inferência dos dois modelos GRU.

Separado de `eval_captures.py` porque precisa de máquina quieta — feche o resto
antes de rodar, e não some estes números com os de uma execução em paralelo.

Dois níveis, porque medem coisas diferentes:

  L1  `session.run` puro, single-thread controlado. Isola o custo do grafo ONNX.
  L2  `predict()` ponta a ponta, incluindo normalização e, no modelo de 37
      features, o cálculo dos canais de cinemática. É o custo real por janela em
      operação, e usa a configuração de sessão que o deploy usa de fato.

ARMADILHA: o L2 NÃO pode usar `np.zeros` como entrada. Com `conf` toda zero, o
`predict()` retorna None antes de tocar o ONNX (menos de 20% de frames válidos),
e o benchmark mediria o early-return — algo como 0,01 ms, um número bonito e
completamente falso. Usamos janelas reais do dataset.

Uso:
    ../modelos/venv/Scripts/python.exe bench_latency_captures.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

import captures_tables
from captures_dataset import DEFAULT_DATA_DIR, WINDOW_SIZE, load_inventory
from eval_captures import MODEL_34, MODEL_37, OUT_DIR, build_models

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BENCH_DIR = Path(__file__).resolve().parent
MODEL_PATHS = {
    MODEL_34: BENCH_DIR / "model" / "gru_2classes.onnx",
    MODEL_37: BENCH_DIR / "model" / "gru_2classes_v2.onnx",
}
FEATURE_DIMS = {MODEL_34: 34, MODEL_37: 37}

N_WARMUP = 10
N_REPS = 500
N_SAMPLE_WINDOWS = 50


def stats(samples_ms: list[float]) -> dict:
    arr = np.array(samples_ms)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "n": len(arr),
    }


def bench_onnx(model_key: str) -> dict:
    """L1 — session.run puro, com threading fixado para comparação justa."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(MODEL_PATHS[model_key]), options, providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    x = np.zeros((1, WINDOW_SIZE, FEATURE_DIMS[model_key]), dtype=np.float32)

    for _ in range(N_WARMUP):
        session.run(None, {input_name: x})

    samples = []
    for _ in range(N_REPS):
        t0 = perf_counter()
        session.run(None, {input_name: x})
        samples.append((perf_counter() - t0) * 1000)
    return stats(samples)


def sample_real_windows(data_dir: Path, models: dict) -> list[np.ndarray]:
    """Janelas reais, válidas nos dois modelos, de takes distintos."""
    inv = load_inventory(data_dir)
    windows: list[np.ndarray] = []
    for take in inv.takes:
        for _, window in take.windows(stride=WINDOW_SIZE):
            if all(m.predict(window, person_id=0) is not None for m in models.values()):
                windows.append(window)
                break
        if len(windows) >= N_SAMPLE_WINDOWS:
            break
    for model in models.values():
        model.reset_person(0)
    return windows


def bench_predict(model, windows: list[np.ndarray]) -> dict:
    """L2 — predict() ponta a ponta sobre janelas reais, em ordem fixa."""
    for i in range(N_WARMUP):
        model.predict(windows[i % len(windows)], person_id=0)

    samples = []
    for i in range(N_REPS):
        window = windows[i % len(windows)]
        t0 = perf_counter()
        model.predict(window, person_id=0)
        samples.append((perf_counter() - t0) * 1000)
    model.reset_person(0)
    return stats(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    models = build_models()
    windows = sample_real_windows(args.data_dir, models)
    if len(windows) < 10:
        raise SystemExit(
            f"Só {len(windows)} janelas válidas encontradas — insuficiente para "
            "medir o L2 com entrada real."
        )
    print(f"Janelas reais para o L2: {len(windows)} (de takes distintos)")

    results = {
        "n_reps": N_REPS,
        "n_warmup": N_WARMUP,
        "n_sample_windows": len(windows),
        "platform": platform.processor() or platform.machine(),
        "platform_full": platform.platform(),
        "onnxruntime": ort.__version__,
        "providers": ["CPUExecutionProvider"],
    }

    for key in (MODEL_34, MODEL_37):
        results[key] = {
            "onnx": bench_onnx(key),
            "end_to_end": bench_predict(models[key], windows),
        }

    print("\n" + "=" * 72)
    print("LATÊNCIA DE INFERÊNCIA")
    print(f"  {N_REPS} repetições, {N_WARMUP} warm-ups, batch=1")
    print("=" * 72)
    print(f"  {'modelo':<12} {'etapa':<14} {'média':>9} {'p50':>9} {'p95':>9} {'mín':>9}")
    print("  " + "-" * 66)
    for key in (MODEL_34, MODEL_37):
        for stage, label in [("onnx", "ONNX puro"), ("end_to_end", "ponta a ponta")]:
            s = results[key][stage]
            print(f"  {key:<12} {label:<14} {s['mean']:>8.3f}ms {s['p50']:>8.3f}ms "
                  f"{s['p95']:>8.3f}ms {s['min']:>8.3f}ms")

    ratio_onnx = results[MODEL_37]["onnx"]["mean"] / results[MODEL_34]["onnx"]["mean"]
    ratio_e2e = (results[MODEL_37]["end_to_end"]["mean"]
                 / results[MODEL_34]["end_to_end"]["mean"])
    print(f"\n  custo relativo 37f/34f:  ONNX {ratio_onnx:.2f}x   "
          f"ponta a ponta {ratio_e2e:.2f}x")
    print(f"  medido em x86 ({results['platform']}); o alvo é Raspberry Pi 5 — "
          "a razão transfere, o absoluto não.")

    # Sanidade: o L2 tem de custar mais que o L1. Se estiverem iguais, o L2 está
    # medindo o early-return de janela inválida.
    for key in (MODEL_34, MODEL_37):
        if results[key]["end_to_end"]["mean"] <= results[key]["onnx"]["mean"]:
            print(f"\n  AVISO [{key}]: ponta a ponta <= ONNX puro. O L2 provavelmente "
                  "está medindo early-return, não inferência real.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "latency.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    captures_tables.write_latency_tables(results, OUT_DIR)


if __name__ == "__main__":
    main()
