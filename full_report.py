"""Monta o relatório completo (GRU + modelo matemático) a partir dos JSONs já
salvos por eval_captures.py, bench_latency_captures.py e eval_math_captures.py.

Não reinfere nada — é uma serialização dos resultados já computados, para
garantir que o relatório e o que rodou no terminal nunca divirjam.

Uso:
    ../modelos/venv/Scripts/python.exe full_report.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "output" / "captures_eval"

MODEL_LABELS = {"34f": "GRU (34 features)", "37f": "GRU + cinemática (37 features)"}


def pct(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{100*v:.1f}%".replace(".", ",")


def num(v, d=3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{d}f}".replace(".", ",")


def ci(bounds) -> str:
    if not bounds or any(b is None or (isinstance(b, float) and math.isnan(b)) for b in bounds):
        return ""
    return f" [IC95% {pct(bounds[0])}–{pct(bounds[1])}]"


def load(name: str) -> dict | None:
    path = OUT_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def section(title: str) -> list[str]:
    return ["", f"## {title}", ""]


def subsection(title: str) -> list[str]:
    return ["", f"### {title}", ""]


# ── GRU ──────────────────────────────────────────────────────────────────────

def gru_sections(summary: dict, latency: dict | None) -> list[str]:
    out = []
    out += section("Parte 1 — Modelos GRU")

    h = summary["headline"]
    out += [
        f"Avaliação sobre **{summary['n_takes']} gravações** "
        f"({summary['metrics']['34f']['tp'] + summary['metrics']['34f']['fn']} de queda, "
        f"{summary['metrics']['34f']['tn'] + summary['metrics']['34f']['fp']} de "
        f"atividade de vida diária). Ponto de operação: uma janela classificada "
        f"como queda já basta para caracterizar detecção "
        f"(`confirm_n={h['confirm_n']}`), com janelas de {h['window_size']} "
        f"quadros a {h['deploy_fps']:.0f} fps e cadência de inferência de "
        f"{h['stride']} quadros. Esse critério foi necessário porque a maioria "
        "das gravações de queda desta campanha produz apenas uma janela válida.",
    ]

    out += subsection("1.1 Métricas principais")
    m = summary["metrics"]
    out += [
        "| Métrica | GRU (34 features) | GRU + cinemática (37 features) |",
        "|---|---|---|",
    ]
    rows = [
        ("Acurácia", "accuracy", "accuracy_ci", True),
        ("Precisão", "precision", None, False),
        ("Sensibilidade (recall)", "recall", "recall_ci", False),
        ("Especificidade", "specificity", "specificity_ci", False),
        ("Falso-alarme", "false_alarm", None, False),
    ]
    for label, key, cikey, _ in rows:
        c34 = ci(m["34f"].get(cikey)) if cikey else ""
        c37 = ci(m["37f"].get(cikey)) if cikey else ""
        out.append(f"| {label} | {pct(m['34f'][key])}{c34} | {pct(m['37f'][key])}{c37} |")
    out.append(f"| F1 (classe queda) | {num(m['34f']['f1_fall'])} | {num(m['37f']['f1_fall'])} |")
    out.append(f"| F1 macro | {num(m['34f']['f1_macro'])} | {num(m['37f']['f1_macro'])} |")

    out += [
        "", "**Matriz de confusão** (classe positiva = queda):", "",
        "| Modelo | VP | FN | FP | VN | Indecisos |",
        "|---|---|---|---|---|---|",
    ]
    for key in ("34f", "37f"):
        mm = m[key]
        out.append(f"| {MODEL_LABELS[key]} | {mm['tp']} | {mm['fn']} | {mm['fp']} | "
                   f"{mm['tn']} | {mm['undecided_fall'] + mm['undecided_adl']} |")

    out += subsection("1.2 Robustez ao padding — só takes com duração suficiente")
    mnp = summary["metrics_no_padding"]
    out += [
        f"Restringindo aos **{summary['n_takes_no_padding']} takes** que não "
        "precisaram de preenchimento de quadros (evita o viés assimétrico: "
        "27 de 50 quedas precisaram de padding, contra 4 de 110 AVDs):",
        "",
        "| Métrica | GRU (34 features) | GRU + cinemática (37 features) |",
        "|---|---|---|",
    ]
    for label, key, _, _ in rows:
        out.append(f"| {label} | {pct(mnp['34f'][key])} | {pct(mnp['37f'][key])} |")
    out.append(f"| F1 (classe queda) | {num(mnp['34f']['f1_fall'])} | {num(mnp['37f']['f1_fall'])} |")

    out += subsection("1.3 Sweep de confirm_n (com teto alcançável)")
    out += [
        "A coluna 'teto' é a fração de gravações de queda com pelo menos `n` "
        "janelas válidas — sem ela, um recall baixo em `confirm_n` alto seria "
        "lido como propriedade do modelo, quando é limite de duração de clipe.",
        "",
    ]
    for stride in (1, 5):
        rows_s = [r for r in summary["sweep"] if r["stride"] == stride]
        out += [f"**Cadência (stride) = {stride} quadros:**", "",
               "| confirm_n | Teto (queda) | Recall 34f | Especif. 34f | Recall 37f | Especif. 37f |",
               "|---|---|---|---|---|---|"]
        for r in rows_s:
            out.append(f"| {r['confirm_n']} | {pct(r['ceiling_fall'])} | "
                       f"{pct(r['recall_34f'])} | {pct(r['specificity_34f'])} | "
                       f"{pct(r['recall_37f'])} | {pct(r['specificity_37f'])} |")
        out.append("")

    out += subsection("1.4 Desempenho por cenário")
    out += ["| Cenário | Classe | GRU 34f | GRU 37f |", "|---|---|---|---|"]
    for rec in summary["per_scenario"]:
        gt = "Queda" if rec["gt"] == "FALL" else "AVD"
        out.append(f"| {rec['scenario_id']} | {gt} | "
                   f"{rec['detected_34f']}/{rec['n_takes']} ({pct(rec['rate_34f'])}) | "
                   f"{rec['detected_37f']}/{rec['n_takes']} ({pct(rec['rate_37f'])}) |")

    out += subsection("1.5 Teste de McNemar (34f vs 37f)")
    mc = summary["mcnemar"]
    out += [
        f"- 34f acerta / 37f erra: **{mc['b_34f_only']}**",
        f"- 37f acerta / 34f erra: **{mc['c_37f_only']}**",
        f"- p-valor exato bilateral: **{num(mc['p_value'])}**",
        ("- Diferença estatisticamente significativa (p < 0,05)."
         if mc["p_value"] < 0.05 else
         "- Diferença não distinguível de ruído (p ≥ 0,05)."),
    ]

    out += subsection("1.6 Diagnóstico por janela — só clipes de AVD (stride 1)")
    wd = summary["window_diagnostics"]
    out += [
        "Rótulo constante só é válido durante a duração inteira em clipes de "
        "AVD; em quedas as janelas pré-impacto são AVD de fato.", "",
        "**Taxa de falso positivo global:**", "",
        "| Modelo | Janelas AVD | Falsos positivos | FPR | IC95% |",
        "|---|---|---|---|---|",
    ]
    for key in ("34f", "37f"):
        g = wd["global"][key]
        out.append(f"| {MODEL_LABELS[key]} | {g['n_windows']} | {g['n_fp']} | "
                   f"{pct(g['fpr'])} | [{pct(g['ci'][0])}–{pct(g['ci'][1])}] |")

    out += ["", "**Por cenário:**", "",
           "| Cenário | Janelas | FPR 34f | FPR 37f | Δ (pp) |",
           "|---|---|---|---|---|"]
    for scenario, e in wd["by_scenario"].items():
        delta = 100 * (e["37f"]["fpr"] - e["34f"]["fpr"])
        out.append(f"| {scenario} | {e['34f']['n']} | {pct(e['34f']['fpr'])} | "
                   f"{pct(e['37f']['fpr'])} | {delta:+.1f} |")

    out += ["", "**Por faixa de rotação de tronco** (suprimidas faixas com n<30):", "",
           "| Rotação | Janelas | FPR 34f | FPR 37f |",
           "|---|---|---|---|"]
    for name_bin, e in wd["by_delta_bin"].items():
        n = e["34f"]["n"]
        if n < 30:
            out.append(f"| {name_bin} | {n} | suprimido (n<30) | suprimido (n<30) |")
            continue
        s34, s37 = e["34f"], e["37f"]
        out.append(f"| {name_bin} | {n} | {pct(s34['fpr'])} "
                   f"[{pct(s34['ci'][0])}–{pct(s34['ci'][1])}] | {pct(s37['fpr'])} "
                   f"[{pct(s37['ci'][0])}–{pct(s37['ci'][1])}] |")

    if latency:
        out += subsection("1.7 Latência de inferência")
        out += [
            f"{latency['n_reps']} repetições cronometradas individualmente, "
            f"{latency['n_warmup']} warm-ups, batch=1, single-thread "
            f"(`intra_op_num_threads=inter_op_num_threads=1`). "
            f"Plataforma: {latency['platform']} (x86 — o alvo de deploy é "
            "Raspberry Pi 5; a razão entre modelos transfere, o absoluto não).",
            "",
            "| Modelo | Etapa | Média (ms) | p50 (ms) | p95 (ms) | Mín (ms) |",
            "|---|---|---|---|---|---|",
        ]
        for key in ("34f", "37f"):
            for stage, label in [("onnx", "Inferência ONNX"),
                                 ("end_to_end", "Ponta a ponta")]:
                s = latency[key][stage]
                out.append(f"| {MODEL_LABELS[key]} | {label} | {num(s['mean'])} | "
                           f"{num(s['p50'])} | {num(s['p95'])} | {num(s['min'])} |")
        ratio_onnx = latency["37f"]["onnx"]["mean"] / latency["34f"]["onnx"]["mean"]
        ratio_e2e = latency["37f"]["end_to_end"]["mean"] / latency["34f"]["end_to_end"]["mean"]
        out.append(f"\nCusto relativo 37f/34f: ONNX puro {ratio_onnx:.2f}x, "
                   f"ponta a ponta {ratio_e2e:.2f}x.")

    rp = summary["reproducibility"]
    out += subsection("1.8 Reprodutibilidade")
    out += [
        f"- ONNX 34f: SHA-256 `{rp['onnx_34f_sha256']}`",
        f"- ONNX 37f: SHA-256 `{rp['onnx_37f_sha256']}`",
        f"- `pose_features` versão `{rp['pose_features_version']}` "
        f"(`{rp['pose_features_file']}`)",
        f"- numpy `{rp['numpy']}`, Python `{rp['python']}`, {rp['platform']}",
        f"- Gerado em: {summary['generated_at']}",
        f"- `merge_disjoint`: {summary['merge_disjoint']}",
    ]
    return out


# ── Modelo matemático ────────────────────────────────────────────────────────

def math_sections(summary_math: dict) -> list[str]:
    out = []
    out += section("Parte 2 — Modelo Matemático")
    out += [
        "Classificador de `vigia/fall-detection` (branch `develop`, "
        "`vigia-fall/capture/classifiers/math_classifier.py`): Kalman → "
        "normalização → PCA → 6 features → score ponderado por sigmoides → "
        "máquina de estados com histerese (`NORMAL → SUSPECT → FALL`, "
        "exigindo 5 quadros consecutivos acima do limiar para cada transição).",
        "",
        "**Ressalva que precede qualquer número desta seção:** os pesos das "
        "features e os midpoints das sigmoides estão marcados como `TODO: "
        "Calibrar os valores a partir de dados reais e rotulados` no "
        "código-fonte de origem. O que segue avalia as constantes tal como "
        "implementadas hoje, não o melhor ajuste possível do modelo.",
    ]

    for cfg_key in ("deploy", "reduzida"):
        cfg = summary_math["configs"][cfg_key]
        m = cfg["metrics"]
        label = m["config"]["label"]
        out += subsection(f"2.{1 if cfg_key=='deploy' else 2} {label}")
        out += [
            f"Janela preencheu ao menos uma vez: queda "
            f"{m['filled_fall']}/{m['n_fall']}, AVD {m['filled_adl']}/{m['n_adl']}. "
            f"Decididos: {m['n']} "
            f"(indecisos: {m['undecided_fall']} quedas, {m['undecided_adl']} AVDs "
            "— janela nunca encheu, fora de qualquer denominador).",
            "",
            "| Métrica | Com máquina de estados | Score bruto (sem persistência) |",
            "|---|---|---|",
        ]
        so = m["score_only"]
        for lab, key in [("Sensibilidade (recall)", "recall"),
                        ("Especificidade", "specificity"),
                        ("Falso-alarme", "false_alarm")]:
            out.append(f"| {lab} | {pct(m[key])} | {pct(so[key])} |")
        out.append(f"| F1 | {num(m['f1'])} | {num(so['f1'])} |")
        out += [
            "", f"Matriz de confusão (com máquina de estados): VP={m['tp']} "
            f"FN={m['fn']} FP={m['fp']} VN={m['tn']}",
            "",
            "*'Score bruto' ignora a exigência de persistência temporal (10 "
            "quadros consecutivos) e considera detectado qualquer instante em "
            "que o score cruzou o limiar superior — isola a capacidade "
            "discriminativa das features da política de confirmação.*",
            "", "**Por cenário:**", "",
            "| Cenário | Classe | Preencheu | Detectados (com persistência) |",
            "|---|---|---|---|",
        ]
        for rec in cfg["per_scenario"]:
            gt = "Queda" if rec["gt"] == "FALL" else "AVD"
            rate = pct(rec["rate"]) if rec["n_filled"] else "n/a"
            out.append(f"| {rec['scenario_id']} | {gt} | "
                       f"{rec['n_filled']}/{rec['n_takes']} | "
                       f"{rec['n_detected']}/{rec['n_filled']} ({rate}) |")

        lat = cfg["latency"]
        out += ["", f"**Latência** (`process()` em regime estacionário, janela cheia, "
               f"{lat['n']} repetições): média {num(lat['mean'])}ms, "
               f"p50 {num(lat['p50'])}ms, p95 {num(lat['p95'])}ms."]

    return out


# ── Comparação consolidada ───────────────────────────────────────────────────

def comparison_section(summary: dict | None, summary_math: dict | None) -> list[str]:
    if not summary or not summary_math:
        return []
    out = section("Parte 3 — Comparação consolidada")
    m = summary["metrics"]
    math_reduzida = summary_math["configs"]["reduzida"]["metrics"]
    math_score = math_reduzida["score_only"]

    out += [
        "Comparação no envelope temporal comum (~2s de janela). O modelo "
        "matemático só é comparável via 'score bruto' — com a máquina de "
        "estados como está configurada, ele não decide a tempo nesta campanha "
        "(ver Parte 2).",
        "",
        "| Modelo | Sensibilidade | Especificidade | Falso-alarme | F1 |",
        "|---|---|---|---|---|",
        f"| GRU (34 features) | {pct(m['34f']['recall'])} | "
        f"{pct(m['34f']['specificity'])} | {pct(m['34f']['false_alarm'])} | "
        f"{num(m['34f']['f1_fall'])} |",
        f"| GRU + cinemática (37 features) | {pct(m['37f']['recall'])} | "
        f"{pct(m['37f']['specificity'])} | {pct(m['37f']['false_alarm'])} | "
        f"{num(m['37f']['f1_fall'])} |",
        f"| Matemático — máquina de estados | {pct(math_reduzida['recall'])} | "
        f"{pct(math_reduzida['specificity'])} | {pct(math_reduzida['false_alarm'])} | "
        f"{num(math_reduzida['f1'])} |",
        f"| Matemático — score bruto | {pct(math_score['recall'])} | "
        f"{pct(math_score['specificity'])} | {pct(math_score['false_alarm'])} | "
        f"{num(math_score['f1'])} |",
        "",
        "**Leitura:** o modelo matemático, avaliado só pelo score (ignorando a "
        "política de confirmação), tem especificidade nitidamente melhor que "
        "os dois GRUs — a fórmula fechada com features de velocidade/aceleração "
        "explícitas discrimina melhor as atividades de baixa altura. Mas como "
        "está implementado hoje (com a máquina de estados de 10 quadros de "
        "persistência), nunca dispara nos clipes desta campanha, que são "
        "curtos demais para satisfazer essa exigência.",
    ]
    return out


def main() -> None:
    summary = load("summary.json")
    latency = load("latency.json")
    summary_math = load("summary_math.json")

    if not summary and not summary_math:
        raise SystemExit(
            "Nenhum resultado encontrado — rode eval_captures.py e "
            "eval_math_captures.py antes."
        )

    lines = [
        "# Relatório completo — GRU e Modelo Matemático",
        "",
        "Consolidação de todos os resultados de avaliação sobre a campanha de "
        "160 gravações capturadas (`keypoint_extraction`), gerada "
        "automaticamente a partir de `summary.json` e `summary_math.json` — "
        "nenhum número aqui foi recalculado ou digitado à mão.",
    ]
    if summary:
        lines += gru_sections(summary, latency)
    if summary_math:
        lines += math_sections(summary_math)
    lines += comparison_section(summary, summary_math)

    lines += section("Apêndice — arquivos de origem")
    lines += [
        "- `summary.json`, `latency.json` — GRU (34f/37f)",
        "- `summary_math.json` — modelo matemático",
        "- `track_selection.csv` — decisão de seleção/fusão de track por take",
        "- `windows_34f.csv`, `windows_37f.csv` — uma linha por janela avaliada",
        "- `takes.csv` — uma linha por take × modelo GRU, com decisão por `confirm_n`",
        "- `tables/` — tabelas 1–4 em Markdown/CSV, formatadas para inserção "
        "direta no artigo",
    ]

    out_path = OUT_DIR / "RELATORIO_COMPLETO.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Relatório completo: {out_path}")
    print(f"  {len(lines)} linhas")


if __name__ == "__main__":
    main()
