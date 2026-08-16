"""Avalia o classificador GRU contra o ground truth de cenário da campanha teste_11_07.

Lê `scenario_labels.csv` (mapeamento arquivo -> cenário real) e os CSVs de
`output/csv_2/`, recalcula `alert` a partir de `label`/`prob_fall` por janela com um
`confirm_n` configurável (equivalente ao `ALERT_PREDS_FALL` de `classifier.py`), e
reporta recall de queda e falso-alarme de ADL por cenário e agregado.

Só lê os CSVs já gravados — não roda o modelo de novo. Serve para testar mudanças de
pós-processamento (Fase 2 do plano) sem precisar de nova captura de campo.

Uso:
    python eval_scenarios.py                    # sweep de confirm_n (2 a 6)
    python eval_scenarios.py --confirm-n 3       # detalhe por arquivo com confirm_n=3
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LABELS_PATH = BASE_DIR / "scenario_labels.csv"
CSV_DIR = BASE_DIR / "output" / "csv_2"

LABEL_TO_CLASS = {"ADL": 0, "FALL": 1}


def load_scenario_labels(path: Path = LABELS_PATH) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def recompute_alert_events(csv_path: Path, confirm_n: int) -> tuple[int, int]:
    """Recalcula alert por (person_id) ignorando janelas INVALID (mesma semântica de
    GRUFallClassifier._check_alert: histórico só avança em janelas válidas).

    Retorna (n_rows_validas, n_eventos_de_alerta) — um evento é uma transição para
    alert=True; múltiplos eventos no mesmo arquivo indicam confirmações repetidas.
    """
    hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=confirm_n))
    prev_alert: dict[str, bool] = defaultdict(bool)
    n_valid = 0
    n_events = 0

    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = row["label"]
            if label not in LABEL_TO_CLASS:
                continue  # INVALID: não altera o histórico, igual ao classifier.py
            n_valid += 1
            person_id = row.get("person_id", "0")
            pred = LABEL_TO_CLASS[label]
            h = hist[person_id]
            h.append(pred)
            alert = pred == 1 and len(h) >= confirm_n and all(p == pred for p in h)
            if alert and not prev_alert[person_id]:
                n_events += 1
            prev_alert[person_id] = alert

    return n_valid, n_events


def evaluate(confirm_n: int) -> list[dict]:
    rows = []
    for entry in load_scenario_labels():
        csv_path = CSV_DIR / entry["csv_filename"]
        if not csv_path.exists():
            continue
        n_valid, n_events = recompute_alert_events(csv_path, confirm_n)
        rows.append({
            "num": int(entry["num"]),
            "gt_label": entry["gt_label"],
            "scenario": entry["scenario"],
            "n_valid_rows": n_valid,
            "alert_events": n_events,
            "detected": n_events > 0,
        })
    return rows


def print_detail(rows: list[dict], confirm_n: int) -> None:
    print(f"confirm_n={confirm_n}")
    print(f"{'#':>3} {'GT':>4} {'cenario':<32} {'rows':>5} {'events':>6}  alertou?")
    for r in rows:
        print(f"{r['num']:>3} {r['gt_label']:>4} {r['scenario']:<32} "
              f"{r['n_valid_rows']:>5} {r['alert_events']:>6}  "
              f"{'SIM' if r['detected'] else 'NAO'}")
    print()
    print_confusion(rows)


def print_confusion(rows: list[dict]) -> dict:
    fall_rows = [r for r in rows if r["gt_label"] == "FALL"]
    adl_rows = [r for r in rows if r["gt_label"] == "ADL"]
    tp = sum(1 for r in fall_rows if r["detected"])
    fp = sum(1 for r in adl_rows if r["detected"])
    recall = tp / len(fall_rows) if fall_rows else float("nan")
    false_alarm = fp / len(adl_rows) if adl_rows else float("nan")
    print(f"FALL: {len(fall_rows)} cenarios  ->  alertou {tp}  (recall={recall:.2%})")
    print(f"ADL : {len(adl_rows)} cenarios  ->  alertou {fp}  (falso-alarme={false_alarm:.2%})")
    return {"recall_fall": recall, "false_alarm_adl": false_alarm}


def sweep(confirm_range: range) -> None:
    print(f"{'confirm_n':>9}  {'recall_FALL':>12}  {'falso_alarme_ADL':>17}")
    for n in confirm_range:
        rows = evaluate(n)
        fall_rows = [r for r in rows if r["gt_label"] == "FALL"]
        adl_rows = [r for r in rows if r["gt_label"] == "ADL"]
        tp = sum(1 for r in fall_rows if r["detected"])
        fp = sum(1 for r in adl_rows if r["detected"])
        recall = tp / len(fall_rows) if fall_rows else float("nan")
        false_alarm = fp / len(adl_rows) if adl_rows else float("nan")
        print(f"{n:>9}  {recall:>11.2%}  {false_alarm:>16.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-n", type=int, default=None,
                         help="Roda detalhe para um confirm_n específico (default: sweep 2-6)")
    parser.add_argument("--sweep-max", type=int, default=6,
                         help="Limite superior do sweep default (default: 6)")
    args = parser.parse_args()

    if not LABELS_PATH.exists():
        raise SystemExit(f"scenario_labels.csv não encontrado em {LABELS_PATH}")
    if not CSV_DIR.exists():
        raise SystemExit(f"Diretório de CSVs não encontrado: {CSV_DIR}")

    if args.confirm_n is not None:
        rows = evaluate(args.confirm_n)
        print_detail(rows, args.confirm_n)
    else:
        sweep(range(2, args.sweep_max + 1))


if __name__ == "__main__":
    main()
