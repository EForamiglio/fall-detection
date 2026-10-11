"""Construcao do vetor de features a partir de keypoints em PIXELS.

MODULO COMPARTILHADO — versao canonica. Uma copia identica precisa existir no
deploy (fall-classifier-bench/classifier.py e vigia/fall-detection). Se treino e
deploy calcularem essas features de forma diferente, o modelo degrada em campo
SEM emitir erro nenhum. Confira FEATURES_VERSION nos dois lados.

--------------------------------------------------------------------------------
Por que este modulo existe
--------------------------------------------------------------------------------
A normalizacao original subtraia o centro do quadril e dividia pela distancia
ombro-quadril A CADA FRAME:

    out[t] = (kp[t] - hip_c[t]) / sh_dist[t]

Isso deixa o quadril exatamente em (0,0) e a escala exatamente em 1.0 em TODOS os
frames — ou seja, apaga completamente a translacao do corpo e a variacao de
tamanho aparente. O modelo nunca via a velocidade de descida, que e justamente o
que separa uma queda de um agachamento controlado. Medimos o efeito disso: o GRU
aprendeu "tronco girando = queda" (falso positivo de 8-10% com tronco estavel
contra 26-35% com rotacao, em val/test).

A correcao e ADITIVA: os 34 valores de pose continuam sendo calculados
exatamente como antes (indices 0..33, significado inalterado), e 3 canais de
movimento sao acrescentados no fim (indices 34..36).

    [0..33]  pose normalizada hip-centered   (17 joints x 2 coords)
    [34]     v_x      velocidade horizontal do quadril
    [35]     v_y      velocidade vertical do quadril
    [36]     d_scale  variacao do tamanho aparente do corpo

--------------------------------------------------------------------------------
Decisoes de projeto
--------------------------------------------------------------------------------
* scale_ref e UM valor por janela (mediana da distancia ombro-quadril), nao um
  valor por frame. Dividir por uma escala por frame reintroduziria exatamente o
  problema que estamos corrigindo. Com a mediana da janela, a velocidade fica
  adimensional (comprimentos de tronco por frame) e continua invariante a
  distancia da camera, que era o objetivo original da normalizacao.

* IMPORTANTE: scale_ref precisa ser calculado sobre o MESMO span no treino e no
  deploy. No deploy o span e a janela de 20 frames, entao no treino as features
  de movimento tambem precisam ser calculadas POR JANELA, nunca sobre o clipe
  inteiro.

* A velocidade so e calculada entre frames em que os DOIS quadris sao validos.
  Se um quadril pisca para invalido, o centro do quadril salta para a metade do
  caminho ate a origem e produziria um pico de velocidade falso.

* Velocidades sao limitadas a +/- VELOCITY_CLIP para que uma troca de ID do
  tracker ou um salto espurio do YOLO nao desestabilize o treino. Medido em
  clipes reais a 10fps: mediana 0.081 e p90 0.481 comprimentos de tronco por
  frame, entao 1.5 e folgado para uma queda real e ainda corta teleportes.

* Continuidade de pessoa NAO e tratada aqui. No deploy, o main.py ja usa o
  tracker BotSort do YOLO e mantem um buffer por person_id, entao a identidade
  ja vem estavel. O problema existe apenas na preparacao dos dados de treino,
  onde o parsing escolhia a pessoa quadro a quadro sem continuidade — ver
  extract_pixel_sequences.py.
"""

from __future__ import annotations

import numpy as np

FEATURES_VERSION = "1.0.0"

POSE_DIM = 34
MOTION_DIM = 3
FEATURE_DIM = POSE_DIM + MOTION_DIM      # 37

# Indices COCO-17
SHO_L, SHO_R = 5, 6
HIP_L, HIP_R = 11, 12

VELOCITY_CLIP = 1.5   # comprimentos de tronco por frame (ver docstring)
_EPS = 1e-6


def _hip_shoulder_centers(
    kp_px: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna (hip_c, sh_dist, both_hips_valid) por frame.

    hip_c    : (T, 2) centro do quadril em pixels
    sh_dist  : (T,)   distancia ombro-quadril em pixels
    both_ok  : (T,)   True quando os dois quadris E os dois ombros sao validos
    """
    hip_c = (kp_px[:, HIP_L, :] + kp_px[:, HIP_R, :]) * 0.5
    sho_c = (kp_px[:, SHO_L, :] + kp_px[:, SHO_R, :]) * 0.5
    sh_dist = np.linalg.norm(sho_c - hip_c, axis=-1)

    both_ok = (
        valid[:, HIP_L] & valid[:, HIP_R] & valid[:, SHO_L] & valid[:, SHO_R]
        & (sh_dist > _EPS)
    )
    return hip_c, sh_dist, both_ok


def normalize_pose(kp_px: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(T, 17, 2) pixels -> (T, 34) hip-centered, escala ombro-quadril.

    Um frame so e considerado valido quando os QUATRO joints do tronco
    (ombros + quadris) sao validos — a mesma regra do deploy
    (GRUFallClassifier._REQUIRED_JOINTS em classifier.py) e do pipeline do
    FallVision, que responde por 73% das janelas de treino.

    ATENCAO: o pipeline do MCFD usava uma regra mais permissiva (bastava UM dos
    quadris). Isso criava uma divergencia entre treino e deploy em 21% dos dados
    de treino, na qual frames aceitos no treino seriam descartados em producao —
    e, pior, com o centro do quadril deslocado, ja que um quadril ausente vale
    (0,0) na media. Padronizamos aqui pela regra do deploy.
    """
    T = kp_px.shape[0]
    out = np.zeros((T, 17, 2), dtype=np.float32)

    hip_c = (kp_px[:, HIP_L, :] + kp_px[:, HIP_R, :]) * 0.5        # (T, 2)
    sho_c = (kp_px[:, SHO_L, :] + kp_px[:, SHO_R, :]) * 0.5 - hip_c
    sh_dist = np.hypot(sho_c[:, 0], sho_c[:, 1])                    # (T,)

    frame_ok = (valid[:, HIP_L] & valid[:, HIP_R]
                & valid[:, SHO_L] & valid[:, SHO_R]
                & (sh_dist >= _EPS))
    if frame_ok.any():
        centered = kp_px[frame_ok] - hip_c[frame_ok][:, np.newaxis, :]
        scaled = centered / sh_dist[frame_ok][:, np.newaxis, np.newaxis]
        # Joints invalidos estavam no pixel (0,0); sem re-zerar viram valores
        # espurios apos a centralizacao.
        out[frame_ok] = scaled * valid[frame_ok][:, :, np.newaxis]

    return out.reshape(T, POSE_DIM)


def motion_features(kp_px: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(T, 17, 2) pixels -> (T, 3) = [v_x, v_y, d_scale].

    O span recebido DEVE ser a janela que o modelo vera (ver docstring do modulo).
    Frame 0 e frames sem par valido recebem 0.
    """
    T = kp_px.shape[0]
    out = np.zeros((T, MOTION_DIM), dtype=np.float32)
    if T < 2:
        return out

    hip_c, sh_dist, both_ok = _hip_shoulder_centers(kp_px, valid)

    if both_ok.any():
        scale_ref = float(np.median(sh_dist[both_ok]))
    else:
        return out                      # sem referencia confiavel de escala
    if scale_ref < _EPS:
        return out

    pair_ok = both_ok[1:] & both_ok[:-1]          # (T-1,) frames t e t-1 validos
    d_hip = (hip_c[1:] - hip_c[:-1]) / scale_ref  # (T-1, 2)
    d_sc = (sh_dist[1:] - sh_dist[:-1]) / scale_ref

    out[1:, 0] = np.where(pair_ok, d_hip[:, 0], 0.0)
    out[1:, 1] = np.where(pair_ok, d_hip[:, 1], 0.0)
    out[1:, 2] = np.where(pair_ok, d_sc, 0.0)

    return np.clip(out, -VELOCITY_CLIP, VELOCITY_CLIP).astype(np.float32)


def build_features(kp_px: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(T, 17, 2) pixels + (T, 17) validade -> (T, 37) pronto para o modelo.

    Chame POR JANELA, com o mesmo span usado no deploy.
    """
    pose = normalize_pose(kp_px, valid)
    motion = motion_features(kp_px, valid)
    return np.concatenate([pose, motion], axis=1).astype(np.float32)


def valid_from_conf(conf: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """(T, 17) confiancas do YOLO -> mascara booleana. Uso no deploy."""
    return conf > threshold


def valid_from_zeros(kp_px: np.ndarray) -> np.ndarray:
    """(T, 17, 2) -> mascara booleana. Uso no data prep, onde joints invalidos
    ja foram zerados na etapa de parsing."""
    return np.any(kp_px != 0, axis=-1)
