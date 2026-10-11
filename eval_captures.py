"""Avalia os dois modelos GRU sobre os takes capturados.

Modelos comparados:
  34f — `model/gru_2classes.onnx`    (o que está no deploy hoje)
  37f — `model/gru_2classes_v2.onnx` (acrescenta 3 canais de cinemática)

Decisões metodológicas que o leitor precisa conhecer para interpretar os números:

* **Ponto de operação `confirm_n=1`.** Nesta campanha 44 dos 50 takes de queda
  produzem UMA única janela de 20 frames (clipes de ~1,9 s a 10 fps). Com o
  `confirm_n=2` do deploy, o recall máximo alcançável seria ~12% — o número
  mediria duração de clipe, não o modelo. O sweep de `confirm_n` é reportado
  com a coluna de teto alcançável justamente para tornar isso visível.

* **Padding assimétrico.** 27 de 50 takes FALL precisam de padding contra 4 de
  110 ADL. Como `pad_to_window` repete o último frame, ele estende a evidência
  de "corpo no chão" nas quedas — infla recall e reduz falso-alarme. Por isso a
  tabela de robustez restrita aos takes sem padding.

* **O campo `alert` dos classificadores nunca é lido.** Ele tem `maxlen=2` fixo
  e não é parametrizável; toda a confirmação temporal é recalculada aqui. Uma
  asserção ao final confere que o recálculo com `confirm_n=2` reproduz os
  eventos embutidos — se não reproduzir, há bug de `reset_person`.

Uso:
    ../modelos/venv/Scripts/python.exe eval_captures.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from captures_dataset import (
    DEFAULT_DATA_DIR,
    WINDOW_SIZE,
    Inventory,
    PreparedTake,
    load_inventory,
    write_track_selection,
)

BENCH_DIR = Path(__file__).resolve().parent
MODELOS_GRU = BENCH_DIR.parent / "modelos" / "training" / "gru"
if str(MODELOS_GRU) not in sys.path:
    sys.path.append(str(MODELOS_GRU))

# O console do Windows é cp1252 e não encoda os símbolos usados nas tabelas
# (Δ, en-dash). Sem isto o script morre no meio do relatório.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pose_features                                          # noqa: E402
from classifier import GRUFallClassifier                      # noqa: E402
from classifier_v2 import GRUFallClassifierV2                 # noqa: E402
from compare_transition_errors import (                       # noqa: E402
    BIN_NAMES,
    DELTA_BINS,
    bin_of,
    delta_trunk,
)

OUT_DIR = BENCH_DIR / "output" / "captures_eval"
HEADLINE_STRIDE = 5          # cadência do deploy (GRU_INTERVAL=0.5s a 10fps)
HEADLINE_CONFIRM_N = 1
MIN_BIN_N = 30               # faixas com menos janelas que isso são suprimidas

MODEL_34 = "34f"
MODEL_37 = "37f"
MODEL_LABELS = {
    MODEL_34: "GRU (34 features)",
    MODEL_37: "GRU + cinemática (37 features)",
}


@dataclass
class WindowRow:
    take_id: str
    scenario_id: str
    is_fall: bool
    window_start: int
    label: str               # ADL | FALL | INVALID
    prob_fall: float
    n_valid_frames: int
    delta_trunk: float
    n_pad: int


# ── Inferência ────────────────────────────────────────────────────────────────

def build_models() -> dict[str, object]:
    return {
        MODEL_34: GRUFallClassifier(BENCH_DIR / "model" / "gru_2classes.onnx"),
        MODEL_37: GRUFallClassifierV2(BENCH_DIR / "model" / "gru_2classes_v2.onnx"),
    }


def window_delta_trunk(windows: list[np.ndarray]) -> np.ndarray:
    """Rotação do tronco por janela, para o diagnóstico por faixa.

    Um único vetor serve os dois modelos: `delta_trunk` usa só as 34 colunas de
    pose, que são idênticas nos dois caminhos de feature.
    """
    if not windows:
        return np.empty(0, dtype=np.float32)
    feats = []
    for window in windows:
        kp = window.reshape(WINDOW_SIZE, 17, 3)
        feats.append(pose_features.build_features(
            kp[:, :, :2], pose_features.valid_from_conf(kp[:, :, 2]),
        ))
    return delta_trunk(np.stack(feats))


def run_inference(
    inv: Inventory, models: dict[str, object],
) -> tuple[dict[str, list[WindowRow]], dict[str, dict[str, list[bool]]]]:
    """Roda os dois modelos em todas as janelas (stride 1).

    Retorna as linhas por janela e, separadamente, os eventos de `alert`
    embutidos por take — usados só para a asserção de consistência.
    """
    rows: dict[str, list[WindowRow]] = {key: [] for key in models}
    builtin_alerts: dict[str, dict[str, list[bool]]] = {key: {} for key in models}

    for take in inv.takes:
        windows = take.windows(stride=1)
        deltas = window_delta_trunk([w for _, w in windows])

        for key, model in models.items():
            model.reset_person(0)
            alerts: list[bool] = []
            for (start, window), delta in zip(windows, deltas):
                pred = model.predict(window, person_id=0)
                if pred is None:
                    label, prob, n_valid = "INVALID", 0.0, 0
                    alerts.append(False)
                else:
                    label = pred["label"]
                    prob = float(pred["probs"][1])
                    n_valid = int(pred["n_valid_frames"])
                    alerts.append(bool(pred["alert"]))
                rows[key].append(WindowRow(
                    take_id=take.take_id,
                    scenario_id=take.scenario_id,
                    is_fall=take.is_fall,
                    window_start=start,
                    label=label,
                    prob_fall=prob,
                    n_valid_frames=n_valid,
                    delta_trunk=float(delta) if np.isfinite(delta) else float("nan"),
                    n_pad=take.n_pad,
                ))
            builtin_alerts[key][take.take_id] = alerts

    return rows, builtin_alerts


# ── Decisão por clipe ─────────────────────────────────────────────────────────

def take_labels(
    rows: list[WindowRow], stride: int,
) -> dict[str, list[str]]:
    """Rótulos por take, na ordem temporal, filtrados pela cadência."""
    out: dict[str, list[str]] = {}
    for row in rows:
        if row.window_start % stride:
            continue
        out.setdefault(row.take_id, []).append(row.label)
    return out


def detected(labels: list[str], confirm_n: int) -> bool:
    """True se houver `confirm_n` predições FALL consecutivas entre as válidas.

    Janelas INVALID são puladas sem zerar o histórico — mesma semântica de
    `eval_scenarios.recompute_alert_events`.
    """
    streak = 0
    for label in labels:
        if label == "INVALID":
            continue
        if label == "FALL":
            streak += 1
            if streak >= confirm_n:
                return True
        else:
            streak = 0
    return False


def has_valid_window(labels: list[str]) -> bool:
    return any(label != "INVALID" for label in labels)


def n_valid_windows(labels: list[str]) -> int:
    return sum(1 for label in labels if label != "INVALID")


# ── Métricas ──────────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC de Wilson 95% para uma proporção. Mais honesto que Wald com n pequeno."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b: int, c: int) -> float:
    """p-valor bilateral exato (binomial) para as discordâncias b e c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


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
        def ratio(num: int, den: int) -> float:
            return num / den if den else float("nan")

        recall = ratio(self.tp, self.tp + self.fn)
        specificity = ratio(self.tn, self.tn + self.fp)
        precision = ratio(self.tp, self.tp + self.fp)
        accuracy = ratio(self.tp + self.tn, self.n)
        f1_fall = (2 * precision * recall / (precision + recall)
                   if precision + recall > 0 else float("nan"))
        # F1 da classe ADL, para o macro
        prec_adl = ratio(self.tn, self.tn + self.fn)
        rec_adl = specificity
        f1_adl = (2 * prec_adl * rec_adl / (prec_adl + rec_adl)
                  if prec_adl + rec_adl > 0 else float("nan"))
        f1_macro = (f1_fall + f1_adl) / 2

        return {
            "tp": self.tp, "fn": self.fn, "fp": self.fp, "tn": self.tn,
            "n": self.n,
            "undecided_fall": self.undecided_fall,
            "undecided_adl": self.undecided_adl,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "false_alarm": 1 - specificity if not math.isnan(specificity) else float("nan"),
            "f1_fall": f1_fall,
            "f1_adl": f1_adl,
            "f1_macro": f1_macro,
            "accuracy_ci": wilson_ci(self.tp + self.tn, self.n),
            "recall_ci": wilson_ci(self.tp, self.tp + self.fn),
            "specificity_ci": wilson_ci(self.tn, self.tn + self.fp),
        }


def confusion_for(
    labels_by_take: dict[str, list[str]], takes: list[PreparedTake], confirm_n: int,
) -> Confusion:
    conf = Confusion()
    for take in takes:
        labels = labels_by_take.get(take.take_id, [])
        if not has_valid_window(labels):
            # Não forçar para ADL: isso infla especificidade de graça.
            if take.is_fall:
                conf.undecided_fall += 1
            else:
                conf.undecided_adl += 1
            continue
        hit = detected(labels, confirm_n)
        if take.is_fall:
            if hit:
                conf.tp += 1
            else:
                conf.fn += 1
        else:
            if hit:
                conf.fp += 1
            else:
                conf.tn += 1
    return conf


# ── Impressão ─────────────────────────────────────────────────────────────────

def pct(value: float) -> str:
    return "  n/a  " if math.isnan(value) else f"{100 * value:5.1f}%"


def print_headline(results: dict[str, dict], takes: list[PreparedTake], title: str) -> None:
    n_fall = sum(1 for t in takes if t.is_fall)
    n_adl = len(takes) - n_fall
    print("\n" + "=" * 78)
    print(title)
    print(f"  {len(takes)} takes ({n_fall} FALL / {n_adl} ADL)  |  "
          f"confirm_n={HEADLINE_CONFIRM_N}  |  stride={HEADLINE_STRIDE}")
    print("=" * 78)
    print(f"  {'métrica':<26} {MODEL_LABELS[MODEL_34]:>24} {MODEL_LABELS[MODEL_37]:>26}")
    print("  " + "-" * 76)
    rows = [
        ("Acurácia", "accuracy"), ("Precisão", "precision"),
        ("Sensibilidade (recall)", "recall"), ("Especificidade", "specificity"),
        ("Falso-alarme", "false_alarm"),
    ]
    for label, key in rows:
        v34, v37 = results[MODEL_34][key], results[MODEL_37][key]
        print(f"  {label:<26} {pct(v34):>24} {pct(v37):>26}")
    for label, key in [("F1 (classe FALL)", "f1_fall"), ("F1 macro", "f1_macro")]:
        v34, v37 = results[MODEL_34][key], results[MODEL_37][key]
        print(f"  {label:<26} {v34:>24.3f} {v37:>26.3f}")

    print("\n  matriz de confusão (FALL = positivo):")
    print(f"  {'':<26} {'VP':>6} {'FN':>6} {'FP':>6} {'VN':>6} {'indecisos':>11}")
    for key in (MODEL_34, MODEL_37):
        m = results[key]
        undecided = m["undecided_fall"] + m["undecided_adl"]
        print(f"  {MODEL_LABELS[key]:<26} {m['tp']:>6} {m['fn']:>6} "
              f"{m['fp']:>6} {m['tn']:>6} {undecided:>11}")


def print_sweep(
    rows: dict[str, list[WindowRow]], takes: list[PreparedTake],
) -> list[dict]:
    print("\n" + "=" * 78)
    print("SWEEP DE confirm_n — com teto alcançável")
    print("=" * 78)
    print("  A coluna 'teto' é a fração de takes que tem pelo menos n janelas válidas.")
    print("  Sem ela, 'recall 0% com confirm_n=4' seria lido como propriedade do")
    print("  modelo, quando é limite de duração de clipe nesta campanha.\n")

    sweep_records = []
    fall_takes = [t for t in takes if t.is_fall]
    for stride in (1, HEADLINE_STRIDE):
        labels_34 = take_labels(rows[MODEL_34], stride)
        labels_37 = take_labels(rows[MODEL_37], stride)
        print(f"  stride {stride}")
        print(f"    {'n':>3} {'teto FALL':>10} {'recall 34f':>11} {'esp. 34f':>10} "
              f"{'recall 37f':>11} {'esp. 37f':>10}")
        for confirm_n in range(1, 11):
            ceiling = sum(
                1 for t in fall_takes
                if n_valid_windows(labels_34.get(t.take_id, [])) >= confirm_n
            ) / max(1, len(fall_takes))
            m34 = confusion_for(labels_34, takes, confirm_n).metrics()
            m37 = confusion_for(labels_37, takes, confirm_n).metrics()
            print(f"    {confirm_n:>3} {pct(ceiling):>10} {pct(m34['recall']):>11} "
                  f"{pct(m34['specificity']):>10} {pct(m37['recall']):>11} "
                  f"{pct(m37['specificity']):>10}")
            sweep_records.append({
                "stride": stride, "confirm_n": confirm_n, "ceiling_fall": ceiling,
                "recall_34f": m34["recall"], "specificity_34f": m34["specificity"],
                "recall_37f": m37["recall"], "specificity_37f": m37["specificity"],
            })
        print()
    return sweep_records


def per_scenario(
    rows: dict[str, list[WindowRow]], takes: list[PreparedTake],
) -> list[dict]:
    labels = {key: take_labels(rows[key], HEADLINE_STRIDE) for key in rows}
    by_scenario: dict[str, list[PreparedTake]] = {}
    for take in takes:
        by_scenario.setdefault(take.scenario_id, []).append(take)

    records = []
    for scenario in sorted(by_scenario, key=lambda s: (not by_scenario[s][0].is_fall, s)):
        group = by_scenario[scenario]
        record = {
            "scenario_id": scenario,
            "gt": "FALL" if group[0].is_fall else "ADL",
            "n_takes": len(group),
        }
        for key in (MODEL_34, MODEL_37):
            hits = sum(
                1 for t in group
                if detected(labels[key].get(t.take_id, []), HEADLINE_CONFIRM_N)
            )
            record[f"detected_{key}"] = hits
            record[f"rate_{key}"] = hits / len(group)
        records.append(record)

    print("\n" + "=" * 78)
    print(f"POR CENÁRIO  (confirm_n={HEADLINE_CONFIRM_N}, stride={HEADLINE_STRIDE})")
    print("=" * 78)
    print(f"  {'cenário':<24} {'GT':>5} {'takes':>6} {'34f':>10} {'37f':>10}")
    print("  " + "-" * 62)
    for rec in records:
        print(f"  {rec['scenario_id']:<24} {rec['gt']:>5} {rec['n_takes']:>6} "
              f"{rec['detected_34f']}/{rec['n_takes']} ({pct(rec['rate_34f']).strip()})".ljust(2)
              + f"{'':>1}{rec['detected_37f']}/{rec['n_takes']} ({pct(rec['rate_37f']).strip()})")
    return records


def print_mcnemar(
    rows: dict[str, list[WindowRow]], takes: list[PreparedTake],
) -> dict:
    labels = {key: take_labels(rows[key], HEADLINE_STRIDE) for key in rows}
    b = c = 0
    for take in takes:
        hit34 = detected(labels[MODEL_34].get(take.take_id, []), HEADLINE_CONFIRM_N)
        hit37 = detected(labels[MODEL_37].get(take.take_id, []), HEADLINE_CONFIRM_N)
        ok34 = hit34 == take.is_fall
        ok37 = hit37 == take.is_fall
        if ok34 and not ok37:
            b += 1
        elif ok37 and not ok34:
            c += 1
    p = mcnemar_exact(b, c)
    print("\n" + "=" * 78)
    print("McNEMAR (34f vs 37f, decisão por take)")
    print("=" * 78)
    print(f"  34f acerta / 37f erra : {b}")
    print(f"  37f acerta / 34f erra : {c}")
    print(f"  p-valor exato bilateral: {p:.4f}")
    print("  " + ("Diferença não distinguível de ruído (p >= 0,05)." if p >= 0.05
                  else "Diferença estatisticamente significativa (p < 0,05)."))
    return {"b_34f_only": b, "c_37f_only": c, "p_value": p}


def window_diagnostics(rows: dict[str, list[WindowRow]]) -> dict:
    """FPR por janela, só em clipes ADL (onde o rótulo vale a duração inteira)."""
    print("\n" + "=" * 78)
    print("DIAGNÓSTICO POR JANELA — só clipes ADL, stride 1")
    print("=" * 78)

    out: dict = {"global": {}, "by_scenario": {}, "by_delta_bin": {}}

    for key in (MODEL_34, MODEL_37):
        adl = [r for r in rows[key] if not r.is_fall and r.label != "INVALID"]
        fp = sum(1 for r in adl if r.label == "FALL")
        lo, hi = wilson_ci(fp, len(adl))
        out["global"][key] = {"n_windows": len(adl), "n_fp": fp,
                              "fpr": fp / len(adl) if adl else float("nan"),
                              "ci": [lo, hi]}
        print(f"  {MODEL_LABELS[key]:<30} FPR = {pct(out['global'][key]['fpr'])} "
              f"({fp}/{len(adl)})  IC95 [{100*lo:.1f}–{100*hi:.1f}]")

    # Por cenário
    scenarios = sorted({r.scenario_id for r in rows[MODEL_34] if not r.is_fall})
    print(f"\n  {'cenário':<24} {'janelas':>8} {'FPR 34f':>9} {'FPR 37f':>9} {'Δ':>8}")
    print("  " + "-" * 62)
    for scenario in scenarios:
        entry = {}
        for key in (MODEL_34, MODEL_37):
            sel = [r for r in rows[key]
                   if r.scenario_id == scenario and r.label != "INVALID"]
            fp = sum(1 for r in sel if r.label == "FALL")
            entry[key] = {"n": len(sel), "n_fp": fp,
                          "fpr": fp / len(sel) if sel else float("nan")}
        delta = entry[MODEL_37]["fpr"] - entry[MODEL_34]["fpr"]
        out["by_scenario"][scenario] = entry
        print(f"  {scenario:<24} {entry[MODEL_34]['n']:>8} "
              f"{pct(entry[MODEL_34]['fpr']):>9} {pct(entry[MODEL_37]['fpr']):>9} "
              f"{100*delta:>+7.1f}pp")

    # Por faixa de rotação de tronco
    print(f"\n  {'rotação do tronco':<22} {'janelas':>8} {'FPR 34f':>18} {'FPR 37f':>18}")
    print("  " + "-" * 70)
    for i, name in enumerate(BIN_NAMES):
        entry = {}
        for key in (MODEL_34, MODEL_37):
            sel = [r for r in rows[key]
                   if not r.is_fall and r.label != "INVALID"
                   and bin_of(r.delta_trunk) == i]
            fp = sum(1 for r in sel if r.label == "FALL")
            lo, hi = wilson_ci(fp, len(sel))
            entry[key] = {"n": len(sel), "n_fp": fp,
                          "fpr": fp / len(sel) if sel else float("nan"),
                          "ci": [lo, hi]}
        n = entry[MODEL_34]["n"]
        out["by_delta_bin"][name] = entry
        if n < MIN_BIN_N:
            print(f"  {name:<22} {n:>8}   (suprimido: n < {MIN_BIN_N})")
            continue
        s34, s37 = entry[MODEL_34], entry[MODEL_37]
        print(f"  {name:<22} {n:>8} "
              f"{pct(s34['fpr'])} [{100*s34['ci'][0]:4.1f}–{100*s34['ci'][1]:4.1f}]"
              f"{pct(s37['fpr'])} [{100*s37['ci'][0]:4.1f}–{100*s37['ci'][1]:4.1f}]")

    return out


# ── Consistência e sanidade ───────────────────────────────────────────────────

def assert_alert_consistency(
    rows: dict[str, list[WindowRow]], builtin: dict[str, dict[str, list[bool]]],
) -> None:
    """O recálculo com confirm_n=2 deve reproduzir o `alert` embutido.

    Se divergir, `reset_person` não está isolando os takes e o histórico vaza
    de um take para o outro.
    """
    for key in rows:
        by_take: dict[str, list[str]] = {}
        for row in rows[key]:
            by_take.setdefault(row.take_id, []).append(row.label)
        for take_id, labels in by_take.items():
            recomputed = detected(labels, confirm_n=2)
            embedded = any(builtin[key][take_id])
            if recomputed != embedded:
                raise SystemExit(
                    f"Inconsistência de alerta em {take_id} ({key}): "
                    f"recalculado={recomputed}, embutido={embedded}. "
                    "Provável vazamento de histórico entre takes."
                )
    print("\n[ok] Recálculo de confirmação reproduz o `alert` embutido (confirm_n=2).")


def sanity_check_velocity(inv: Inventory) -> None:
    """|v_y| médio em FALL deve ser ~2x o de ADL (razão medida no treino).

    Se a reamostragem 30->10fps tiver falhado, as velocidades saem ~3x menores
    e essa razão se descaracteriza — é a única checagem que pega esse bug, que
    de outra forma é silencioso e penaliza só o modelo de 37 features.
    """
    sums = {True: [], False: []}
    for take in inv.takes:
        for _, window in take.windows(stride=HEADLINE_STRIDE):
            kp = window.reshape(WINDOW_SIZE, 17, 3)
            feats = pose_features.build_features(
                kp[:, :, :2], pose_features.valid_from_conf(kp[:, :, 2]),
            )
            sums[take.is_fall].append(np.abs(feats[:, 35]).mean())
    v_fall, v_adl = float(np.mean(sums[True])), float(np.mean(sums[False]))
    ratio = v_fall / v_adl if v_adl else float("nan")
    print(f"[sanidade] |v_y| médio  FALL={v_fall:.4f}  ADL={v_adl:.4f}  razão={ratio:.2f}x")
    if not (1.2 <= ratio <= 4.0):
        print("  AVISO: razão fora do esperado (~2x). Conferir a reamostragem 30->10fps.")


# ── Artefatos ─────────────────────────────────────────────────────────────────

def write_window_rows(rows: list[WindowRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_takes_csv(
    rows: dict[str, list[WindowRow]], takes: list[PreparedTake], path: Path,
) -> None:
    labels_by_stride = {
        stride: {key: take_labels(rows[key], stride) for key in rows}
        for stride in (1, HEADLINE_STRIDE)
    }
    fields = [
        "take_id", "scenario_id", "is_fall", "model", "chosen_person_idx",
        "t30", "t10", "n_pad", "n_windows_s1", "n_valid_windows_s1",
        "n_fall_windows_s1", "max_prob_fall",
    ] + [f"detected_confirm{n}_s{HEADLINE_STRIDE}" for n in range(1, 11)]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for take in takes:
            for key in rows:
                take_rows = [r for r in rows[key] if r.take_id == take.take_id]
                labels_s1 = [r.label for r in take_rows]
                record = {
                    "take_id": take.take_id, "scenario_id": take.scenario_id,
                    "is_fall": take.is_fall, "model": key,
                    "chosen_person_idx": take.chosen_person_idx,
                    "t30": take.t30, "t10": take.t10, "n_pad": take.n_pad,
                    "n_windows_s1": len(labels_s1),
                    "n_valid_windows_s1": n_valid_windows(labels_s1),
                    "n_fall_windows_s1": sum(1 for x in labels_s1 if x == "FALL"),
                    "max_prob_fall": f"{max((r.prob_fall for r in take_rows), default=0.0):.4f}",
                }
                headline = labels_by_stride[HEADLINE_STRIDE][key].get(take.take_id, [])
                for n in range(1, 11):
                    record[f"detected_confirm{n}_s{HEADLINE_STRIDE}"] = detected(headline, n)
                writer.writerow(record)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-merge-disjoint", action="store_true")
    args = parser.parse_args()

    inv = load_inventory(args.data_dir, merge_disjoint=not args.no_merge_disjoint)
    print(f"Takes avaliáveis: {len(inv.takes)}  "
          f"(FALL {sum(1 for t in inv.takes if t.is_fall)} / "
          f"ADL {sum(1 for t in inv.takes if not t.is_fall)})")

    sanity_check_velocity(inv)

    models = build_models()
    rows, builtin = run_inference(inv, models)
    assert_alert_consistency(rows, builtin)

    labels_headline = {key: take_labels(rows[key], HEADLINE_STRIDE) for key in rows}
    results = {
        key: confusion_for(labels_headline[key], inv.takes, HEADLINE_CONFIRM_N).metrics()
        for key in rows
    }
    print_headline(results, inv.takes, "MÉTRICAS PRINCIPAIS")

    no_pad = [t for t in inv.takes if t.n_pad == 0]
    results_no_pad = {
        key: confusion_for(labels_headline[key], no_pad, HEADLINE_CONFIRM_N).metrics()
        for key in rows
    }
    print_headline(results_no_pad, no_pad,
                   "ROBUSTEZ — só takes sem padding (controle do viés de padding)")

    sweep = print_sweep(rows, inv.takes)
    scenarios = per_scenario(rows, inv.takes)
    mcnemar = print_mcnemar(rows, inv.takes)
    windows_diag = window_diagnostics(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_track_selection(inv, OUT_DIR / "track_selection.csv")
    for key in rows:
        write_window_rows(rows[key], OUT_DIR / f"windows_{key}.csv")
    write_takes_csv(rows, inv.takes, OUT_DIR / "takes.csv")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(args.data_dir),
        "merge_disjoint": not args.no_merge_disjoint,
        "headline": {"confirm_n": HEADLINE_CONFIRM_N, "stride": HEADLINE_STRIDE,
                     "window_size": WINDOW_SIZE, "deploy_fps": 10.0},
        "n_takes": len(inv.takes),
        "metrics": results,
        "metrics_no_padding": results_no_pad,
        "n_takes_no_padding": len(no_pad),
        "sweep": sweep,
        "per_scenario": scenarios,
        "mcnemar": mcnemar,
        "window_diagnostics": windows_diag,
        "reproducibility": {
            "onnx_34f_sha256": file_sha256(BENCH_DIR / "model" / "gru_2classes.onnx"),
            "onnx_37f_sha256": file_sha256(BENCH_DIR / "model" / "gru_2classes_v2.onnx"),
            "pose_features_file": pose_features.__file__,
            "pose_features_version": pose_features.FEATURES_VERSION,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")

    print(f"\nArtefatos em {OUT_DIR}")

    import captures_tables
    captures_tables.write_all(summary, OUT_DIR)


if __name__ == "__main__":
    main()
