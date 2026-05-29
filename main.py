"""fall-classifier-bench — banco de testes isolado pra validar GRU 2-classes.

Loop single-process: captura → YOLO pose → buffer → GRU → overlay/CSV/logs.

Uso típico:
    python main.py --source 0                          # webcam, com display
    python main.py --source 0 --no-display             # headless, só CSV+logs
    python main.py --source 0 --yolo-model yolov8n-pose.pt --imgsz 320
    python main.py --source video.mp4 --tag bench_v1   # vídeo gravado
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from loguru import logger

from classifier import GRUFallClassifier
from metrics import MetricsLogger
from overlay import draw_hud, draw_person_label, draw_skeleton
from pose_detector import PoseDetector

# Constantes alinhadas com o treino e com vigia
WINDOW_SIZE = 20         # 20 frames @ 10fps deploy = 2s de contexto
GRU_INTERVAL = 0.5       # mínimo entre inferências GRU por pessoa (s)
PERSON_TIMEOUT = 3.0     # remover buffer após pessoa ausente por X s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bench do classificador GRU 2-classes")
    p.add_argument("--source", default="0",
                   help="Índice da webcam (ex: 0) ou caminho de vídeo. Default: 0")
    p.add_argument("--yolo-model", default="yolov8n-pose.pt",
                   help="Modelo YOLO pose. Default: yolov8n-pose.pt "
                        "(YOLOv8 nano — consistente com o treino do GRU).")
    p.add_argument("--yolo-conf", type=float, default=0.25,
                   help="Threshold de confiança do YOLO. Default: 0.25")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Tamanho de input do YOLO (640 ou 320). Default: 640")
    p.add_argument("--device", default="cpu",
                   help="Device do YOLO: cpu, cuda. Default: cpu")
    p.add_argument("--tag", default="run",
                   help="Tag pra identificar a sessão nos outputs. Default: run")
    p.add_argument("--no-display", action="store_true",
                   help="Não abre janela cv2.imshow mesmo com DISPLAY setado")
    p.add_argument("--save-frames", type=int, metavar="N",
                   help="Salva 1 JPG anotado a cada N segundos em output/frames/")
    p.add_argument("--save-video", metavar="PATH",
                   help="Grava vídeo anotado em PATH (.mp4)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _setup_logging(log_path: Path, level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")
    logger.add(log_path, level="DEBUG", rotation=None,
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {message}")


def _resolve_source(src: str):
    """Aceita índice numérico ou path de vídeo."""
    try:
        return int(src)
    except ValueError:
        return src


def _should_display(no_display: bool) -> bool:
    if no_display:
        return False
    return bool(os.environ.get("DISPLAY")) or sys.platform == "win32"


def main() -> int:
    args = parse_args()
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}"
    out_dir = Path(__file__).resolve().parent / "output"

    csv_path = out_dir / "csv" / f"{run_id}.csv"
    log_path = out_dir / "logs" / f"{run_id}.log"
    _setup_logging(log_path, args.log_level)

    logger.info("=" * 60)
    logger.info("fall-classifier-bench | run_id={}", run_id)
    logger.info("source={} model={} imgsz={} conf={} device={}",
                args.source, args.yolo_model, args.imgsz, args.yolo_conf, args.device)

    # --- inicialização ---
    source = _resolve_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Não foi possível abrir source={!r}", source)
        return 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    logger.info("câmera OK: {}x{} @ {:.1f}fps reportado", w, h, cap_fps)

    detector = PoseDetector(
        model_path=args.yolo_model,
        device=args.device,
        conf=args.yolo_conf,
        imgsz=args.imgsz,
    )
    logger.info("YOLO pose carregado: {}", args.yolo_model)

    classifier = GRUFallClassifier()
    logger.info("GRU 2-classes carregado")

    metrics = MetricsLogger(csv_path)
    logger.info("CSV: {}", csv_path)

    display = _should_display(args.no_display)
    logger.info("display={} no_display_flag={}", display, args.no_display)

    video_writer = None
    if args.save_video:
        video_path = Path(args.save_video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (w, h))
        logger.info("gravação de vídeo: {}", video_path)

    frames_dir = None
    if args.save_frames:
        frames_dir = out_dir / "frames" / run_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        logger.info("snapshots em {} a cada {}s", frames_dir, args.save_frames)

    # --- buffers por pessoa ---
    buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
    last_inference: dict[int, float] = {}
    last_seen: dict[int, float] = {}
    last_pred_by_pid: dict[int, dict | None] = {}

    # --- rolling stats pro HUD ---
    fps_window = deque(maxlen=30)
    t_yolo_window = deque(maxlen=30)
    t_gru_window = deque(maxlen=30)

    frame_idx = 0
    last_snapshot_t = 0.0
    started_at = perf_counter()

    try:
        while True:
            loop_t0 = perf_counter()

            t0 = perf_counter()
            ok, frame = cap.read()
            t_capture_ms = (perf_counter() - t0) * 1000

            if not ok:
                logger.warning("fim do stream ou erro de leitura no frame {}", frame_idx)
                break

            frame_idx += 1
            now = perf_counter()

            # YOLO
            t0 = perf_counter()
            detections = detector.detect(frame)
            t_yolo_ms = (perf_counter() - t0) * 1000
            t_yolo_window.append(t_yolo_ms)

            # Atualiza buffers + roda GRU quando janela cheia + cooldown
            active_pids = set()
            t_gru_last = 0.0
            for det in detections:
                active_pids.add(det.person_id)
                last_seen[det.person_id] = now
                buffers[det.person_id].append(det.kpts_flat)

                is_full = len(buffers[det.person_id]) == WINDOW_SIZE
                cooldown_ok = (now - last_inference.get(det.person_id, 0.0)) >= GRU_INTERVAL

                if is_full and cooldown_ok:
                    window = np.array(buffers[det.person_id])
                    t0 = perf_counter()
                    pred = classifier.predict(window, person_id=det.person_id)
                    t_gru_ms = (perf_counter() - t0) * 1000
                    t_gru_last = t_gru_ms
                    t_gru_window.append(t_gru_ms)
                    last_inference[det.person_id] = now
                    last_pred_by_pid[det.person_id] = pred

                    if pred is None:
                        logger.debug("ID={} janela inválida (keypoints insuficientes)",
                                     det.person_id)
                        metrics.record(
                            frame_idx=frame_idx, person_id=det.person_id,
                            t_capture_ms=t_capture_ms, t_yolo_ms=t_yolo_ms,
                            t_gru_ms=t_gru_ms, n_valid_kpts=det.n_valid_kpts,
                            n_valid_frames_window=0,
                            label="INVALID", prob_adl=0.0, prob_fall=0.0,
                            alert=False, n_detections_total=len(detections),
                        )
                    else:
                        if pred["alert"]:
                            logger.info(
                                "FALL CONFIRMADO | ID={} probs=[{:.2f},{:.2f}] "
                                "t_yolo={:.0f}ms t_gru={:.1f}ms",
                                det.person_id, pred["probs"][0], pred["probs"][1],
                                t_yolo_ms, t_gru_ms,
                            )
                        elif pred["label"] == "FALL":
                            logger.info(
                                "FALL pending | ID={} probs=[{:.2f},{:.2f}]",
                                det.person_id, pred["probs"][0], pred["probs"][1],
                            )
                        else:
                            logger.debug(
                                "ADL | ID={} probs=[{:.2f},{:.2f}]",
                                det.person_id, pred["probs"][0], pred["probs"][1],
                            )

                        metrics.record(
                            frame_idx=frame_idx, person_id=det.person_id,
                            t_capture_ms=t_capture_ms, t_yolo_ms=t_yolo_ms,
                            t_gru_ms=t_gru_ms, n_valid_kpts=det.n_valid_kpts,
                            n_valid_frames_window=pred["n_valid_frames"],
                            label=pred["label"],
                            prob_adl=pred["probs"][0], prob_fall=pred["probs"][1],
                            alert=pred["alert"], n_detections_total=len(detections),
                        )

            # Cleanup de pessoas ausentes
            for pid in list(buffers.keys()):
                if pid not in active_pids and (now - last_seen.get(pid, now)) >= PERSON_TIMEOUT:
                    del buffers[pid]
                    last_inference.pop(pid, None)
                    last_seen.pop(pid, None)
                    last_pred_by_pid.pop(pid, None)
                    classifier.reset_person(pid)
                    logger.debug("buffer do ID {} removido por inatividade", pid)

            # FPS rolling
            loop_dt = perf_counter() - loop_t0
            if loop_dt > 0:
                fps_window.append(1.0 / loop_dt)
            fps_avg = sum(fps_window) / len(fps_window) if fps_window else 0.0
            t_yolo_avg = sum(t_yolo_window) / len(t_yolo_window) if t_yolo_window else 0.0
            t_gru_avg = sum(t_gru_window) / len(t_gru_window) if t_gru_window else 0.0

            # Overlay
            need_overlay = display or video_writer is not None or frames_dir is not None
            if need_overlay:
                for det in detections:
                    draw_skeleton(frame, det.kpts_flat)
                    draw_person_label(
                        frame, det.bbox, det.person_id,
                        last_pred_by_pid.get(det.person_id),
                    )
                draw_hud(frame, fps_avg, t_yolo_avg, t_gru_avg,
                         len(detections), args.yolo_model)

            # Saída
            if display:
                cv2.imshow("fall-classifier-bench", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("encerrado pelo usuário (q)")
                    break

            if video_writer is not None:
                video_writer.write(frame)

            if frames_dir is not None and args.save_frames:
                if (now - last_snapshot_t) >= args.save_frames:
                    snap_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(snap_path), frame)
                    last_snapshot_t = now

    except KeyboardInterrupt:
        logger.info("encerrado por Ctrl+C")

    finally:
        elapsed = perf_counter() - started_at
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        logger.info("=" * 60)
        logger.info("Sessão encerrada: {} frames em {:.1f}s = {:.1f} FPS",
                    frame_idx, elapsed, avg_fps)
        logger.info("CSV: {}", csv_path)
        logger.info("LOG: {}", log_path)

        cap.release()
        if video_writer is not None:
            video_writer.release()
        if display:
            cv2.destroyAllWindows()
        metrics.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
