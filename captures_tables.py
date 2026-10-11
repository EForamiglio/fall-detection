"""Geração das tabelas acadêmicas em Markdown + CSV, para o artigo.

As tabelas são serializações do `summary.json` produzido por `eval_captures.py`
— nunca um segundo cálculo das métricas. Isso garante que o que vai para o
artigo é exatamente o que foi impresso no terminal.

Convenções aplicadas (é o que diferencia tabela publicável de dump de terminal):
legenda acima, notas de rodapé abaixo, separador decimal por vírgula,
N e denominador explícitos em toda proporção, IC de Wilson 95% nas proporções
de headline, e melhor valor em negrito com ressalva do teste de McNemar.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

MODEL_34, MODEL_37 = "34f", "37f"
MODEL_LABELS = {
    MODEL_34: "GRU (34 features)",
    MODEL_37: "GRU + cinemática (37 features)",
}

SCENARIO_LABELS = {
    "queda_lado_esquerdo": "Queda lateral (esquerda)",
    "queda_lado_direito": "Queda lateral (direita)",
    "queda_costas_esquerdo": "Queda de costas (esquerda)",
    "queda_costas_direito": "Queda de costas (direita)",
    "queda_de_frente": "Queda frontal",
    "andando": "Caminhar",
    "sentando": "Sentar-se",
    "deitando": "Deitar-se",
    "agachando_chao": "Agachar (objeto no chão)",
    "agachando_banco": "Agachar (objeto em banco)",
    "varrendo": "Varrer",
    "tomando_banho": "Tomar banho",
    "vestindo_roupa": "Vestir-se",
    "abracando": "Abraçar",
    "duas_pessoas_andando": "Duas pessoas caminhando",
    "tropeco": "Tropeço (sem queda)",
}

_METRIC_ROWS = [
    ("Acurácia", "accuracy", "accuracy_ci", ("tp", "tn"), "n"),
    ("Precisão", "precision", None, ("tp",), None),
    ("Sensibilidade (recall)", "recall", "recall_ci", ("tp",), None),
    ("Especificidade", "specificity", "specificity_ci", ("tn",), None),
    ("Taxa de falso alarme", "false_alarm", None, ("fp",), None),
]


def _num(value: float, decimals: int = 1) -> str:
    """Formata com vírgula decimal, como manda a norma em português."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{decimals}f}".replace(".", ",")


def _pct(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{_num(100 * value)}%"


def _ci(bounds) -> str:
    if not bounds or any(b is None or math.isnan(b) for b in bounds):
        return ""
    return f" [IC95% {_num(100 * bounds[0])}–{_num(100 * bounds[1])}]"


def _numerator(metrics: dict, keys: tuple[str, ...]) -> int:
    return sum(int(metrics[k]) for k in keys)


def _denominator(metrics: dict, key: str | None, metric_key: str) -> int:
    if key:
        return int(metrics[key])
    if metric_key == "precision":
        return int(metrics["tp"]) + int(metrics["fp"])
    if metric_key == "recall":
        return int(metrics["tp"]) + int(metrics["fn"])
    if metric_key in ("specificity", "false_alarm"):
        return int(metrics["tn"]) + int(metrics["fp"])
    return int(metrics["n"])


def _cell(metrics: dict, metric_key: str, ci_key: str | None,
          num_keys: tuple[str, ...], den_key: str | None, bold: bool) -> str:
    value = metrics[metric_key]
    num = _numerator(metrics, num_keys)
    den = _denominator(metrics, den_key, metric_key)
    text = f"{_pct(value)} ({num}/{den})"
    if bold:
        text = f"**{text}**"
    return text + _ci(metrics.get(ci_key) if ci_key else None)


def _best(a: float, b: float, lower_is_better: bool = False) -> str | None:
    """Qual modelo tem o melhor valor, ou None em caso de empate."""
    if any(isinstance(v, float) and math.isnan(v) for v in (a, b)):
        return None
    if a == b:
        return None
    if lower_is_better:
        return MODEL_34 if a < b else MODEL_37
    return MODEL_34 if a > b else MODEL_37


def table1_metricas(summary: dict) -> tuple[str, list[list[str]]]:
    m = summary["metrics"]
    n_takes = summary["n_takes"]
    n_fall = m[MODEL_34]["tp"] + m[MODEL_34]["fn"]
    n_adl = m[MODEL_34]["tn"] + m[MODEL_34]["fp"]

    lines = [
        f"**Tabela 1 —** Desempenho dos modelos GRU na detecção de quedas "
        f"(N = {n_takes} gravações: {n_fall} quedas e {n_adl} atividades de vida diária).",
        "",
        f"| Métrica | {MODEL_LABELS[MODEL_34]} | {MODEL_LABELS[MODEL_37]} |",
        "|---|---|---|",
    ]
    csv_rows = [["metrica", "modelo_34f", "modelo_37f"]]

    for label, key, ci_key, num_keys, den_key in _METRIC_ROWS:
        winner = _best(m[MODEL_34][key], m[MODEL_37][key],
                       lower_is_better=(key == "false_alarm"))
        cells = {
            model: _cell(m[model], key, ci_key, num_keys, den_key, winner == model)
            for model in (MODEL_34, MODEL_37)
        }
        lines.append(f"| {label} | {cells[MODEL_34]} | {cells[MODEL_37]} |")
        csv_rows.append([label, _pct(m[MODEL_34][key]), _pct(m[MODEL_37][key])])

    for label, key in [("F1 (classe queda)", "f1_fall"), ("F1 macro", "f1_macro")]:
        winner = _best(m[MODEL_34][key], m[MODEL_37][key])
        cells = {}
        for model in (MODEL_34, MODEL_37):
            text = _num(m[model][key], 3)
            cells[model] = f"**{text}**" if winner == model else text
        lines.append(f"| {label} | {cells[MODEL_34]} | {cells[MODEL_37]} |")
        csv_rows.append([label, _num(m[MODEL_34][key], 3), _num(m[MODEL_37][key], 3)])

    lines += [
        "",
        "**Matriz de confusão** (classe positiva = queda):",
        "",
        "| Modelo | Verdadeiros positivos | Falsos negativos | "
        "Falsos positivos | Verdadeiros negativos |",
        "|---|---|---|---|---|",
    ]
    for model in (MODEL_34, MODEL_37):
        mm = m[model]
        lines.append(f"| {MODEL_LABELS[model]} | {mm['tp']} | {mm['fn']} | "
                     f"{mm['fp']} | {mm['tn']} |")
        csv_rows.append([f"confusao_{model}",
                         f"VP={mm['tp']};FN={mm['fn']}",
                         f"FP={mm['fp']};VN={mm['tn']}"])

    mcnemar = summary["mcnemar"]
    no_pad = summary["metrics_no_padding"]
    lines += [
        "",
        f"*Ponto de operação: uma única janela classificada como queda é suficiente "
        f"para caracterizar detecção (confirm_n = 1), com janelas de "
        f"{summary['headline']['window_size']} quadros a "
        f"{_num(summary['headline']['deploy_fps'], 0)} fps e cadência de inferência de "
        f"{summary['headline']['stride']} quadros. Esse critério foi adotado porque a "
        f"maioria das gravações de queda desta campanha produz apenas uma janela "
        f"válida, o que impediria o uso do critério de confirmação dupla empregado "
        f"em produção.*",
        "",
        f"*Valores em negrito indicam o melhor desempenho por métrica. O teste de "
        f"McNemar sobre as decisões por gravação resultou em p = "
        f"{_num(mcnemar['p_value'], 3)} "
        f"({mcnemar['b_34f_only']} discordâncias favoráveis ao modelo de 34 features "
        f"e {mcnemar['c_37f_only']} ao de 37), "
        + ("de modo que as diferenças observadas não são estatisticamente "
           "distinguíveis de variação aleatória.*"
           if mcnemar["p_value"] >= 0.05 else
           "indicando diferença estatisticamente significativa.*"),
        "",
        f"*Análise de robustez: restringindo a amostra às "
        f"{summary['n_takes_no_padding']} gravações com duração suficiente para "
        f"dispensar preenchimento de quadros, a acurácia foi de "
        f"{_pct(no_pad[MODEL_34]['accuracy'])} (34 features) e "
        f"{_pct(no_pad[MODEL_37]['accuracy'])} (37 features).*",
    ]
    return "\n".join(lines), csv_rows


def table2_por_cenario(summary: dict) -> tuple[str, list[list[str]]]:
    records = summary["per_scenario"]
    lines = [
        "**Tabela 2 —** Taxa de detecção por cenário "
        "(10 gravações por cenário; confirm_n = 1).",
        "",
        f"| Cenário | Classe de referência | {MODEL_LABELS[MODEL_34]} | "
        f"{MODEL_LABELS[MODEL_37]} |",
        "|---|---|---|---|",
    ]
    csv_rows = [["cenario", "classe", "detectados_34f", "taxa_34f",
                 "detectados_37f", "taxa_37f", "n_takes"]]

    for rec in records:
        label = SCENARIO_LABELS.get(rec["scenario_id"], rec["scenario_id"])
        gt = "Queda" if rec["gt"] == "FALL" else "AVD"
        n = rec["n_takes"]
        cells = []
        for model in (MODEL_34, MODEL_37):
            hits = rec[f"detected_{model}"]
            cells.append(f"{hits}/{n} ({_pct(rec[f'rate_{model}'])})")
        lines.append(f"| {label} | {gt} | {cells[0]} | {cells[1]} |")
        csv_rows.append([
            label, gt, rec["detected_34f"], _pct(rec["rate_34f"]),
            rec["detected_37f"], _pct(rec["rate_37f"]), n,
        ])

    lines += [
        "",
        "*Para cenários de queda, a taxa corresponde à sensibilidade "
        "(proporção de quedas detectadas); para cenários de atividade de vida "
        "diária (AVD), corresponde à taxa de falso alarme (proporção de "
        "gravações em que o sistema alertou indevidamente).*",
    ]
    return "\n".join(lines), csv_rows


def table3_latencia(latency: dict) -> tuple[str, list[list[str]]]:
    lines = [
        "**Tabela 3 —** Latência de inferência por janela "
        f"({latency['n_reps']} repetições, execução em uma única thread).",
        "",
        "| Modelo | Etapa medida | Média (ms) | Mediana (ms) | p95 (ms) | Mínimo (ms) |",
        "|---|---|---|---|---|---|",
    ]
    csv_rows = [["modelo", "etapa", "media_ms", "mediana_ms", "p95_ms", "minimo_ms"]]

    stage_labels = {
        "onnx": "Inferência ONNX",
        "end_to_end": "Pipeline completo (normalização + inferência)",
    }
    for model in (MODEL_34, MODEL_37):
        for stage in ("onnx", "end_to_end"):
            stats = latency[model][stage]
            lines.append(
                f"| {MODEL_LABELS[model]} | {stage_labels[stage]} | "
                f"{_num(stats['mean'], 3)} | {_num(stats['p50'], 3)} | "
                f"{_num(stats['p95'], 3)} | {_num(stats['min'], 3)} |"
            )
            csv_rows.append([
                MODEL_LABELS[model], stage_labels[stage],
                _num(stats["mean"], 3), _num(stats["p50"], 3),
                _num(stats["p95"], 3), _num(stats["min"], 3),
            ])

    lines += [
        "",
        f"*Medições realizadas em arquitetura x86 ({latency['platform']}), "
        "enquanto o destino de implantação é um Raspberry Pi 5. A razão entre os "
        "dois modelos é transferível; os valores absolutos, não.*",
        "",
        "*O pipeline completo inclui a normalização dos keypoints e, no modelo de "
        "37 features, o cálculo dos canais de cinemática — é a medida comparável "
        "ao custo real por janela em operação.*",
    ]
    return "\n".join(lines), csv_rows


def table4_modelo_matematico(summary_math: dict) -> tuple[str, list[list[str]]]:
    configs = summary_math["configs"]
    lines = [
        "**Tabela 4 —** Desempenho do classificador matemático (Kalman + "
        "features geométricas + score ponderado) sobre as mesmas 160 gravações, "
        "em duas configurações de janela.",
        "",
        "| Configuração | N decididos | Sensibilidade | Especificidade | "
        "Falso alarme | F1 |",
        "|---|---|---|---|---|---|",
    ]
    csv_rows = [["configuracao", "n_decididos", "sensibilidade", "especificidade",
                 "falso_alarme", "f1", "variante"]]

    for name in ("deploy", "reduzida"):
        m = configs[name]["metrics"]
        label = m["config"]["label"]
        for variant, mm in [("Com máquina de estados", m),
                            ("Score bruto (sem persistência)", m["score_only"])]:
            lines.append(
                f"| {label} — {variant} | {mm['n']} | {_pct(mm['recall'])} | "
                f"{_pct(mm['specificity'])} | {_pct(mm['false_alarm'])} | "
                f"{_num(mm['f1'], 3)} |"
            )
            csv_rows.append([label, mm["n"], _pct(mm["recall"]),
                             _pct(mm["specificity"]), _pct(mm["false_alarm"]),
                             _num(mm["f1"], 3), variant])

    deploy_undecided_fall = configs["deploy"]["metrics"]["undecided_fall"]
    deploy_n_fall = configs["deploy"]["metrics"]["n_fall"]
    lines += [
        "",
        f"*Os pesos e os limiares das funções sigmoides usados no cálculo do "
        f"score estão marcados como não calibrados no código-fonte de origem "
        f"('a calibrar a partir de dados reais e rotulados'); os valores acima "
        f"avaliam as constantes tal como implementadas, não o melhor ajuste "
        f"possível do modelo.*",
        "",
        f"*Na configuração de deploy (janela de 2,5 s), "
        f"{deploy_undecided_fall} das {deploy_n_fall} gravações de queda não "
        f"preencheram a janela mínima e ficaram fora do denominador — a "
        f"maioria das quedas desta campanha é mais curta que a janela "
        f"configurada.*",
        "",
        "*'Score bruto' avalia apenas se o score ultrapassou o limiar superior "
        "em algum instante, ignorando a exigência de persistência temporal da "
        "máquina de estados (10 quadros consecutivos); isola a capacidade "
        "discriminativa das features da política de confirmação.*",
    ]
    return "\n".join(lines), csv_rows


def _write(path: Path, markdown: str, csv_rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".md").write_text(markdown + "\n", encoding="utf-8")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)


def write_all(summary: dict, out_dir: Path) -> None:
    """Tabelas 1 e 2 + relatório consolidado (inclui 3 e 4 se já geradas)."""
    tables_dir = out_dir / "tables"

    md1, csv1 = table1_metricas(summary)
    _write(tables_dir / "tabela1_metricas", md1, csv1)

    md2, csv2 = table2_por_cenario(summary)
    _write(tables_dir / "tabela2_por_cenario", md2, csv2)

    parts = [
        "# Resultados — avaliação dos modelos de detecção de queda",
        "",
        "Tabelas prontas para o artigo. Geradas a partir de `summary.json` / "
        "`summary_math.json`; os números são os mesmos impressos pelos "
        "scripts de avaliação.",
        "", md1, "", md2,
    ]

    latency_md = tables_dir / "tabela3_latencia.md"
    if latency_md.exists():
        parts += ["", latency_md.read_text(encoding="utf-8").strip()]
    else:
        parts += ["", "*Tabela 3 (latência): rode `bench_latency_captures.py` "
                  "e este relatório será regenerado com ela.*"]

    math_md = tables_dir / "tabela4_modelo_matematico.md"
    if math_md.exists():
        parts += ["", math_md.read_text(encoding="utf-8").strip()]

    (tables_dir / "relatorio_tabelas.md").write_text(
        "\n".join(parts) + "\n", encoding="utf-8")
    print(f"Tabelas acadêmicas em {tables_dir}")


def write_math_table(summary_math: dict, out_dir: Path) -> None:
    """Tabela 4 (modelo matemático) + regeneração do relatório consolidado."""
    tables_dir = out_dir / "tables"
    md4, csv4 = table4_modelo_matematico(summary_math)
    _write(tables_dir / "tabela4_modelo_matematico", md4, csv4)

    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        write_all(json.loads(summary_path.read_text(encoding="utf-8")), out_dir)
    else:
        print(f"Tabela 4 em {tables_dir} (rode eval_captures.py para o relatório completo)")


def write_latency_tables(latency: dict, out_dir: Path) -> None:
    """Tabela 3 + regeneração do relatório consolidado, se o summary existir."""
    tables_dir = out_dir / "tables"
    md3, csv3 = table3_latencia(latency)
    _write(tables_dir / "tabela3_latencia", md3, csv3)

    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        write_all(json.loads(summary_path.read_text(encoding="utf-8")), out_dir)
    else:
        print(f"Tabela 3 em {tables_dir} (rode eval_captures.py para o relatório)")
