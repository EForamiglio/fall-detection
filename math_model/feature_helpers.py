""" "
Helpers para extração de features
"""

import math

import numpy as np

from .capture_constants import COM_SHOULDER_WEIGHT


def get_linear_speed(
    current_position: float,
    current_timestamp: float,
    previous_position: float,
    previous_timestamp: float,
) -> float:
    """
    Calcula a velocidade linear de um corpo a partir de suas coordenadas e do tempo.
    """
    # v(t) = (p(t) - p(t-1)) / (t - t-1)

    return (current_position - previous_position) / (
        current_timestamp - previous_timestamp
    )


def get_angular_speed(
    current_position: tuple[float, float],
    current_timestamp: float,
    previous_position: tuple[float, float],
    previous_timestamp: float,
) -> float:
    """
    Calcula a velocidade angular (rad/s) do vetor posição de um ponto,
    a partir da variação do ângulo entre dois instantes.
    """
    # w(t) = (theta(t) - theta(t-1)) / (t - t-1)

    current_angle = math.atan2(current_position[1], current_position[0])
    previous_angle = math.atan2(previous_position[1], previous_position[0])

    # normaliza a variação para [-pi, pi] p/ evitar saltos de 2*pi
    delta_angle = (current_angle - previous_angle + math.pi) % (2 * math.pi) - math.pi

    return delta_angle / (current_timestamp - previous_timestamp)


def get_linear_acceleration(
    current_speed: float,
    current_timestamp: float,
    previous_speed: float,
    previous_timestamp: float,
) -> float:
    """
    Calcula a aceleração linear de um corpo a partir da variação da velocidade no tempo.
    """
    # a(t) = (v(t) - v(t-1)) / (t - t-1)

    return (current_speed - previous_speed) / (current_timestamp - previous_timestamp)


def get_angular_acceleration(
    current_angular_speed: float,
    current_timestamp: float,
    previous_angular_speed: float,
    previous_timestamp: float,
) -> float:
    """
    Calcula a aceleração angular (rad/s^2) a partir da variação da
    velocidade angular no tempo.
    """
    # alpha(t) = (w(t) - w(t-1)) / (t - t-1)

    return (current_angular_speed - previous_angular_speed) / (
        current_timestamp - previous_timestamp
    )


def get_trunk_angle(
    shoulder_center: tuple[float, float],
    hip_center: tuple[float, float],
) -> float:
    """
    Calcula o ângulo do tronco de um corpo a partir das coordenadas do ombro e do quadril.
    Retorna o ângulo em radianos com o posicionamento do tronco em relação ao eixo vertical
    """

    shoulder_x, shoulder_y = shoulder_center
    hip_x, hip_y = hip_center

    trunk_angle = math.atan2((hip_x - shoulder_x), (hip_y - shoulder_y))

    return trunk_angle


def get_center_of_mass(
    shoulder_center: tuple[float, float],
    hip_center: tuple[float, float],
    shoulder_weight: float = COM_SHOULDER_WEIGHT,
) -> tuple[float, float]:
    """
    Aproxima o centro de massa do tronco como média ponderada entre
    o centro dos ombros e o centro do quadril.

    Pesos biomecânicos típicos: ~0.6 no ombro (tronco superior + cabeça)
    e o restante no quadril. Retorna a posição no mesmo espaço dos centros
    de entrada (pixels da imagem, antes da normalização).
    """
    hip_weight = 1.0 - shoulder_weight
    return (
        shoulder_weight * shoulder_center[0] + hip_weight * hip_center[0],
        shoulder_weight * shoulder_center[1] + hip_weight * hip_center[1],
    )


def get_pca_features(
    coordinates: dict[int, tuple[float, float]],
) -> tuple[float, float]:
    """
    Analisa a nuvem de keypoints de um frame via PCA (Análise de Componentes Principais).

    Roda o PCA sobre os pontos daquele instante e usa os autovalores da matriz de
    covariância para descrever a forma da silhueta:

    Retorna uma tupla (aspect_ratio, principal_angle):
        - aspect_ratio: razão entre o eixo principal e o secundário (sqrt(lambda1 / lambda2)).
          Valores altos => silhueta alongada em uma direção (ex.: em pé);
          valores próximos de 1 => silhueta achatada/espalhada (ex.: no chão).
        - principal_angle: ângulo (rad) do eixo principal em relação ao eixo horizontal,
          indicando a orientação predominante do corpo.
    """

    points = np.array(list(coordinates.values()), dtype=float)

    # PCA precisa de ao menos 2 pontos para estimar a covariância
    if points.shape[0] < 2:
        return 1.0, 0.0

    centered = points - points.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)

    # eigh retorna autovalores em ordem crescente (matriz simétrica)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    minor_eigenvalue, major_eigenvalue = eigenvalues

    # clamp evita divisão por zero quando a nuvem é degenerada (pontos ~colineares)
    minor_eigenvalue = max(float(minor_eigenvalue), 1e-12)
    aspect_ratio = math.sqrt(float(major_eigenvalue) / minor_eigenvalue)

    # autovetor associado ao maior autovalor = direção do eixo principal
    principal_axis = eigenvectors[:, -1]
    principal_angle = math.atan2(float(principal_axis[1]), float(principal_axis[0]))

    return aspect_ratio, principal_angle


def align_pca_angle(current_angle: float, previous_angle: float | None) -> float:
    """
    Remove o flip de sinal do eixo PCA (período π) e mantém o ângulo
    contínuo em relação ao frame anterior, no intervalo [-π, π].
    """
    if previous_angle is None:
        return current_angle
    # escolhe current ou current+π — o mais próximo do ângulo anterior
    candidates = (current_angle, current_angle + math.pi)
    best = min(
        candidates,
        key=lambda a: abs((a - previous_angle + math.pi) % (2 * math.pi) - math.pi),
    )
    # re-wrap em [-π, π]
    return (best + math.pi) % (2 * math.pi) - math.pi


def get_angle_speed(
    current_angle: float,
    current_timestamp: float,
    previous_angle: float,
    previous_timestamp: float,
) -> float:
    """Velocidade angular a partir de ângulos já alinhados (wrap 2π no delta)."""
    delta = (current_angle - previous_angle + math.pi) % (2 * math.pi) - math.pi
    return delta / (current_timestamp - previous_timestamp)


def sigmoid_normalize(x: float, midpoint: float, steepness: float) -> float:
    """
    Mapeia x para [0,1]
    midpoint: valor de x onde f(x) = 0.5 (o "limiar de ambiguidade")
    steepness: controla quão abrupta é a transição
    """
    return 1 / (1 + math.exp(-steepness * (x - midpoint)))


def compute_fall_score(features: dict[str, float], weights: dict[str, float]) -> float:
    """
    Calcula o score de queda a partir das features e dos pesos
    """
    return sum(weights[k] * features[k] for k in features)
