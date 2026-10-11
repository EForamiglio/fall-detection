> **ATENÇÃO — NÃO EXECUTAR ESTE GUIA POR ENQUANTO.**
> Foi escrito antes da avaliação de campo. Nessa avaliação (160 gravações) o modelo de
> 37 features teve desempenho **pior** que o de 34 (falso alarme 54,5% contra 41,8%;
> McNemar p = 0,009). Porém aquela análise assumiu 30 fps e reamostrou por índice de
> frame, enquanto o fps real de gravação foi medido entre 12 e 22 — isso infla as features
> de velocidade do 37f, então **a comparação provavelmente está contaminada** e precisa
> ser refeita com reamostragem por tempo (ver `keypoint_extraction/README.md`, seção
> "Notas para continuidade"). Até lá, mantenha o `gru_2classes.onnx` (34f) em produção.

# Integração do GRU v2 (37 features) — guia para o `vigia/fall-detection`

Documento para o Pedro. Descreve o que muda no deploy, por quê, e como verificar
que a integração está correta antes de confiar no resultado.

## Resumo em uma linha

O modelo passa a receber 3 canais a mais (velocidade do corpo), a entrada muda de
`(1, 20, 34)` para `(1, 20, 37)`, e o cálculo das features sai de dentro do
`fall_classifier.py` para um módulo compartilhado com o treino.

## Por que a mudança

A normalização atual subtrai o centro do quadril e divide pela distância
ombro-quadril **a cada frame**. O efeito colateral é que o quadril fica em
`(0,0)` e a escala em `1.0` em todos os frames — ou seja, a translação do corpo é
apagada por completo. O modelo nunca teve acesso à velocidade de descida, que é
justamente o que distingue uma queda de um agachamento controlado.

Medições que motivaram a correção:

| medição | valor |
|---|---|
| falso positivo, tronco estável (val/test) | 8-10% |
| falso positivo, tronco em rotação | 26-35% |
| falso alarme em campo, cenários ADL (49 testes) | 76% |

O modelo antigo aprendeu, corretamente para os dados que recebeu, que "tronco
girando = queda". Como sentar, deitar, agachar e varrer também giram o tronco,
eles disparam alarme.

Efeito medido da correção, com 3 seeds por configuração:

| | 34 features | 37 features |
|---|---|---|
| FP em janelas com rotação (test) | 26,6% | **21,3%** |
| razão rotação/estável | 3,3x | **2,5x** |
| held-out F1 macro | 0,839 | 0,829 |
| latência ONNX (CPU) | 0,10 ms | 0,107 ms |

A correção ajuda, mas não resolve sozinha — janelas com rotação ainda erram 2,5x
mais. Por isso a validação de campo é necessária antes de considerar isso pronto.

## O que copiar

| origem | destino |
|---|---|
| `modelos/data_prep/pose_features.py` | `src/app/core/pose_features.py` |
| `modelos/cv25_fallvision/models/gru_2classes_v2.onnx` | `model/gru_2classes_v2.onnx` |

O `gru_2classes.onnx` antigo **não deve ser removido** — os dois formatos são
incompatíveis, e manter o antigo permite rollback imediato.

## Mudanças no `src/app/core/fall_classifier.py`

Três pontos, todos dentro de `GRUFallClassifier`:

**1. Caminho padrão do modelo**

```python
# antes
onnx_path = str(repo_or_bundle_root() / "model" / "gru_2classes.onnx")
# depois
onnx_path = str(repo_or_bundle_root() / "model" / "gru_2classes_v2.onnx")
```

**2. Shape da entrada em `predict`**

```python
# antes
x = normalized.reshape(1, -1, 34).astype(np.float32)
# depois
x = normalized.reshape(1, -1, 37).astype(np.float32)
```

**3. Corpo de `_normalize_window`**

O cálculo manual é substituído por uma chamada ao módulo compartilhado. A regra
de descarte (menos de 20% de frames válidos → `None`) é preservada:

```python
from app.core.pose_features import FEATURE_DIM, build_features, valid_from_conf

def _normalize_window(self, window: np.ndarray) -> np.ndarray | None:
    kp_full = np.asarray(window, dtype=np.float32).reshape(-1, 17, 3)
    kp_xy   = kp_full[:, :, :2]
    kp_conf = kp_full[:, :, 2]

    frame_valid = np.all(kp_conf[:, self._REQUIRED_JOINTS] > 0, axis=1)
    if frame_valid.mean() < 0.20:
        return None

    return build_features(kp_xy, valid_from_conf(kp_conf))   # (T, 37)
```

Uma referência funcional completa está em `classifier_v2.py` neste repositório.
`frame_consumer.py`, `action_classifier.py`, a lógica de alerta e o tracking
**não mudam**.

## Pré-requisito que já está satisfeito

`WINDOW_SIZE = 20` em `frame_consumer.py`. Isso agora importa mais do que antes:
a escala de referência da velocidade é calculada **sobre a janela**, então uma
janela de tamanho diferente produziria features com escala diferente da do
treino. Se o `WINDOW_SIZE` mudar algum dia, o modelo precisa ser retreinado.

(A docstring de `GRUFallClassifier` ainda diz "janela (60, 51)" — está
desatualizada desde a mudança de 30 para 20 frames; vale corrigir de passagem.)

## Verificação

**1. Sincronia entre treino e deploy.** O `pose_features.py` do deploy precisa
ser cópia exata do usado no treino. Se divergirem, o modelo degrada em campo sem
gerar erro nenhum. Confira `FEATURES_VERSION` nos dois lados:

```bash
grep FEATURES_VERSION src/app/core/pose_features.py
```

**2. Shape aceito pelo modelo.**

```python
import onnxruntime as ort
s = ort.InferenceSession("model/gru_2classes_v2.onnx")
print(s.get_inputs()[0].shape)   # ['batch_size', 20, 37]
```

O `classifier_v2.py` já faz essa checagem no `__init__` e levanta erro explícito
se o modelo e o módulo de features estiverem fora de sincronia.

**3. Fumaça ponta a ponta.** Rodar o `fall-classifier-bench` com o modelo novo e
confirmar que as predições saem e que `n_valid_frames` continua coerente.

## Rollback

Reverter os três pontos acima e voltar o caminho para `gru_2classes.onnx`. Como o
arquivo antigo permanece no repositório, o rollback não depende de rebuild.

## O que ainda não foi feito

A validação de campo. Os 49 testes já gravados **não podem ser reprocessados**
com o modelo novo, porque os CSVs guardam apenas as predições, não os keypoints
brutos. É necessária uma nova campanha com o mesmo protocolo de cenários para
comparar contra os 76% de falso alarme da baseline.
