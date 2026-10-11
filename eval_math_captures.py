"""Avalia o modelo matemático (Kalman + features + score ponderado) sobre os
takes capturados, em duas configurações:

  DEPLOY    window=30 frames, 12 fps  (o que está de fato configurado hoje:
            shared/settings.py::Settings.frame_rate=12, slider_window_size=30)
  REDUZIDA  window=20 frames, 10 fps  (mesmo envelope temporal do GRU — 2s —
            para comparação lado a lado, não é uma config real de deploy)

Ressalva que precisa acompanhar qualquer número deste script: os pesos e os
midpoints das sigmoides em `math_model/math_classifier.py` estão marcados
como TODO no código original do Pedro ("calibrar a partir de dados reais e
rotulados"). O que está sendo medido é o desempenho das CONSTANTES ATUAIS,
não do "melhor modelo matemático possível".

Achado que motiva rodar as duas configurações: na config de DEPLOY, a janela
de 2,5s nunca enche em 44 dos 50 takes de queda desta campanha (clipes de
~1,9s de mediana) — ver `inventory_math.py`. Isso não é um bug do modelo, é a
duração dos clipes vs. o parâmetro de janela configurado.

Uso:
    ../modelos/venv/Scripts/python.exe eval_math_captures.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from captures_dataset import DEFAULT_DATA_DIR, RawTake, resolve_raw_takes  # noqa: E402
from eval_captures import wilson_ci                                        # noqa: E402
from math_model_adapter import (                                           # noqa: E402
    MIN_KPT_CONF,
    reset_state,
    resample_native,
    run_take,
    take_detected,
)
from math_model.math_classifier import MathFallClassifier                  # noqa: E402
from math_model.types import PoseObservation                               # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "captures_eval"
EXPECTED_SRC_FPS = 30.0

CONFIGS = {
    "deploy": {"fps": 12.0, "window": 30,
              "label": "Configuração de deploy (12 fps, janela 2,5 s)"},
    "reduzida": {"fps": 10.0, "window": 20,
                "label": "Janela reduzida (10 fps, janela 2 s — envelope do GRU)"},
}


SCORE_THRESHOLD_HIGH = 0.65   # FallDetectorConfig.threshold_high, duplicado aqui só p/ o diagnóstico


@dataclass
class TakeResult:
    take_id: str
    scenario_id: str
    is_fall: bool
    n_frames_target: int
    n_decisions: int
    filled: bool          # a janela chegou a encher ao menos uma vez
    detected: bool         # o FallDetector chegou a FALL (com persistência)
    max_score: float | None
    frame_first_fall: int | None

    @property
    def detected_score_only(self) -> bool:
        """Diagnóstico: score cruzou o limiar em algum frame, IGNORANDO a
        exigência de persistência de 10 frames da máquina de estados.

        Isola a qualidade do score (features + pesos) do timing da política de
        confirmação — mesma lógica de separar 'dados' de 'features' na
        ablação do GRU.
        """
        return self.max_score is not None and self.max_score >= SCORE_THRESHOLD_HIGH


def evaluate_config(raw_takes: list[RawTake], fps: float, window: int) -> list[TakeResult]:
    results = []
    for take in raw_takes:
        reset_state()
        px, labels = resample_native(take.px, take.labels, EXPECTED_SRC_FPS, fps)
        decisions = run_take(px, target_fps=fps, window_size=window)

        scores = [d.score for d in decisions if d.score is not None]
        fall_frames = [d.frame_idx for d in decisions if d.label == "FALL"]

        results.append(TakeResult(
            take_id=take.take_id,
            scenario_id=take.scenario_id,
            is_fall=take.is_fall,
            n_frames_target=len(px),
            n_decisions=len(decisions),
            filled=len(decisions) > 0,
            detected=take_detected(decisions),
            max_score=max(scores) if scores else None,
            frame_first_fall=fall_frames[0] if fall_frames else None,
        ))
    reset_state()
    return results


@dataclass
class Confusion:
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0
    undecided_fall: int = 0
    undecided_adl: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fn + self.fp + self.tn

    def metrics(self) -> dict:
        def ratio(num, den):
            return num / den if den else float("nan")

        recall = ratio(self.tp, self.tp + self.fn)
        specificity = ratio(self.tn, self.tn + self.fp)
        precision = ratio(self.tp, self.tp + self.fp)
        accuracy = ratio(self.tp + self.tn, self.n)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else float("nan"))
        return {
            "tp": self.tp, "fn": self.fn, "fp": self.fp, "tn": self.tn, "n": self.n,
            "undecided_fall": self.undecided_fall, "undecided_adl": self.undecided_adl,
            "accuracy": accuracy, "precision": precision, "recall": recall,
            "specificity": specificity,
            "false_alarm": 1 - specificity if not math.isnan(specificity) else float("nan"),
            "f1": f1,
            "accuracy_ci": wilson_ci(self.tp + self.tn, self.n),
            "recall_ci": wilson_ci(self.tp, self.tp + self.fn),
            "specificity_ci": wilson_ci(self.tn, self.tn + self.fp),
        }


def confusion_for(results: list[TakeResult], score_only: bool = False) -> Confusion:
    conf = Confusion()
    for r in results:
        if not r.filled:
            if r.is_fall:
                conf.undecided_fall += 1
            else:
                conf.undecided_adl += 1
            continue
        hit = r.detected_score_only if score_only else r.detected
        if r.is_fall:
            conf.tp += 1 if hit else 0
            conf.fn += 0 if hit else 1
        else:
            conf.fp += 1 if hit else 0
            conf.tn += 0 if hit else 1
    return conf


def pct(value: float) -> str:
    return "  n/a  " if (value is None or math.isnan(value)) else f"{100 * value:5.1f}%"


def print_config_result(name: str, results: list[TakeResult]) -> dict:
    cfg = CONFIGS[name]
    conf = confusion_for(results)
    m = conf.metrics()
    n_fall = sum(1 for r in results if r.is_fall)
    n_adl = len(results) - n_fall
    filled_fall = sum(1 for r in results if r.is_fall and r.filled)
    filled_adl = sum(1 for r in results if not r.is_fall and r.filled)

    print("\n" + "=" * 74)
    print(f"MODELO MATEMÁTICO — {cfg['label']}")
    print("=" * 74)
    print(f"  janela preenche ao menos 1x: FALL {filled_fall}/{n_fall}   "
          f"ADL {filled_adl}/{n_adl}")
    print(f"  decididos (janela encheu): {conf.n}  "
          f"(indecisos: FALL {conf.undecided_fall}, ADL {conf.undecided_adl})")
    print()
    print(f"  {'métrica':<24} {'valor':>10}   (sobre os decididos)")
    print("  " + "-" * 50)
    for label, key in [("Acurácia", "accuracy"), ("Precisão", "precision"),
                       ("Sensibilidade (recall)", "recall"),
                       ("Especificidade", "specificity"),
                       ("Falso-alarme", "false_alarm")]:
        print(f"  {label:<24} {pct(m[key]):>10}")
    print(f"  {'F1':<24} {m['f1']:>10.3f}")
    print(f"\n  matriz de confusão (só decididos): VP={m['tp']} FN={m['fn']} "
          f"FP={m['fp']} VN={m['tn']}")

    m["n_fall"], m["n_adl"] = n_fall, n_adl
    m["filled_fall"], m["filled_adl"] = filled_fall, filled_adl
    m["config"] = {"fps": cfg["fps"], "window": cfg["window"], "label": cfg["label"]}

    conf_score = confusion_for(results, score_only=True)
    m_score = conf_score.metrics()
    print(f"\n  DIAGNÓSTICO — só o score, sem a máquina de estados (persistência ignorada):")
    print(f"  {'métrica':<24} {'com persistência':>17} {'só score':>12}")
    print("  " + "-" * 56)
    for label, key in [("Sensibilidade (recall)", "recall"),
                       ("Especificidade", "specificity"),
                       ("Falso-alarme", "false_alarm")]:
        print(f"  {label:<24} {pct(m[key]):>17} {pct(m_score[key]):>12}")
    print(f"  {'F1':<24} {m['f1']:>17.3f} {m_score['f1']:>12.3f}" if not math.isnan(m_score['f1'])
          else f"  {'F1':<24} {m['f1']:>17.3f} {'nan':>12}")
    m["score_only"] = m_score

    return m


def per_scenario(name: str, results: list[TakeResult]) -> list[dict]:
    by_scenario: dict[str, list[TakeResult]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario_id, []).append(r)

    records = []
    print(f"\n  por cenário ({CONFIGS[name]['label']}):")
    print(f"  {'cenário':<24} {'GT':>5} {'encheu':>8} {'detectados':>11}")
    for scenario in sorted(by_scenario, key=lambda s: (not by_scenario[s][0].is_fall, s)):
        group = by_scenario[scenario]
        gt = "FALL" if group[0].is_fall else "ADL"
        filled = [r for r in group if r.filled]
        hits = sum(1 for r in filled if r.detected)
        records.append({
            "scenario_id": scenario, "gt": gt, "n_takes": len(group),
            "n_filled": len(filled), "n_detected": hits,
            "rate": hits / len(filled) if filled else float("nan"),
        })
        rate_str = pct(hits / len(filled)) if filled else "  n/a  "
        print(f"  {scenario:<24} {gt:>5} {len(filled):>5}/{len(group):<2} "
              f"{hits:>4}/{len(filled) if filled else 0:<2} ({rate_str})")
    return records


def bench_latency(fps: float, window: int, n_reps: int = 300) -> dict:
    """Custo de `process()` em regime estacionário (janela já cheia) — é a
    chamada mais cara (extração de features + PCA + score), e é o que roda a
    cada frame em produção (diferente do GRU, que só roda a cada
    GRU_INTERVAL=0.5s)."""
    reset_state()
    classifier = MathFallClassifier(window_size=window)

    rng = np.random.default_rng(0)

    def synthetic_frame(i: int) -> np.ndarray:
        kpts = np.zeros((17, 3), dtype=np.float64)
        kpts[:, 2] = 0.9
        jitter = rng.normal(0, 1.5, size=(17, 2))
        base = np.array([
            [100, 80], [95, 75], [105, 75], [90, 78], [110, 78],
            [90, 100], [110, 100], [85, 140], [115, 140], [85, 175], [115, 175],
            [90, 160], [110, 160], [85, 200], [115, 200], [85, 240], [115, 240],
        ], dtype=np.float64)
        kpts[:, :2] = base + jitter
        return kpts

    # preenche a janela (custo não medido)
    for i in range(window):
        obs = [PoseObservation(person_id=0, keypoints=synthetic_frame(i),
                               timestamp=i / fps)]
        classifier.process(obs)

    samples = []
    for i in range(window, window + n_reps):
        obs = [PoseObservation(person_id=0, keypoints=synthetic_frame(i),
                               timestamp=i / fps)]
        t0 = perf_counter()
        classifier.process(obs)
        samples.append((perf_counter() - t0) * 1000)

    reset_state()
    arr = np.array(samples)
    return {"mean": float(arr.mean()), "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)), "min": float(arr.min()), "n": n_reps}


def main() -> None:
    raw_takes, missing_takes, missing_clips = resolve_raw_takes(DEFAULT_DATA_DIR)
    print(f"Takes avaliáveis: {len(raw_takes)}  "
          f"(FALL {sum(1 for t in raw_takes if t.is_fall)} / "
          f"ADL {sum(1 for t in raw_takes if not t.is_fall)})  "
          f"| missing_takes={len(missing_takes)}")
    print(f"\nNOTA: pesos e limiares de {MIN_KPT_CONF=} e das sigmoides em "
          "math_model/math_classifier.py estão marcados como TODO no código "
          "original (não calibrados). Os números abaixo avaliam essas "
          "constantes tais como estão hoje.")

    all_results: dict[str, dict] = {}
    for name, cfg in CONFIGS.items():
        results = evaluate_config(raw_takes, cfg["fps"], cfg["window"])
        metrics = print_config_result(name, results)
        scenarios = per_scenario(name, results)
        latency = bench_latency(cfg["fps"], cfg["window"])
        print(f"\n  latência process() em regime estacionário: "
              f"média={latency['mean']:.3f}ms  p50={latency['p50']:.3f}ms  "
              f"p95={latency['p95']:.3f}ms")

        all_results[name] = {
            "metrics": metrics, "per_scenario": scenarios, "latency": latency,
            "takes": [
                {"take_id": r.take_id, "scenario_id": r.scenario_id,
                 "is_fall": r.is_fall, "n_frames_target": r.n_frames_target,
                 "n_decisions": r.n_decisions, "filled": r.filled,
                 "detected": r.detected, "max_score": r.max_score,
                 "frame_first_fall": r.frame_first_fall}
                for r in results
            ],
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "math (Kalman + features + score ponderado, pesos não calibrados)",
        "configs": all_results,
    }
    (OUT_DIR / "summary_math.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"\nArtefatos em {OUT_DIR / 'summary_math.json'}")

    import captures_tables
    captures_tables.write_math_table(summary, OUT_DIR)


if __name__ == "__main__":
    main()
