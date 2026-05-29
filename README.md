# fall-classifier-bench

Banco de testes isolado pra validar o classificador GRU de quedas (ADL/FALL).
Single-process, instrumentado, sem dependências de FIWARE/MQTT/ZMQ.

## Por que existe

Validar o classificador rodando dentro do `vigia/fall-detection` é difícil porque
muitos problemas surgem da arquitetura multi-processo (race conditions, ZMQ, MQTT,
device sync FIWARE). Este projeto isola apenas: **captura → YOLO pose → GRU**.

Permite responder com clareza:
- O YOLO está extraindo keypoints decentes na minha cena?
- A GRU classifica certo quando os keypoints são bons?
- Falsos positivos vêm do YOLO ou da GRU?
- Onde está o gargalo de FPS na Raspi?

## Estrutura

```
fall-classifier-bench/
├── main.py             # loop principal
├── pose_detector.py    # wrapper YOLO + tracker
├── classifier.py       # GRU 2-classes + normalização
├── overlay.py          # desenho cv2 (skeleton, label, HUD)
├── metrics.py          # CSV append por classificação
├── model/
│   └── gru_2classes.onnx
└── output/             # gerado em runtime (gitignored)
    ├── csv/
    ├── logs/
    ├── frames/         # só se --save-frames
    └── video/          # só se --save-video
```

## Instalação

### Windows (dev)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Raspberry Pi 5 (Debian Trixie, Python 3.13)
```bash
# Setup one-time
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0 libsm6 libxext6
sudo usermod -a -G video $USER   # logout/login após

# Projeto
cd ~
git clone <repo-url> fall-classifier-bench
cd fall-classifier-bench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# Validação visual com HDMI conectado (default)
python main.py --source 0

# SSH headless, só CSV+logs
python main.py --source 0 --no-display

# Sessão arquivada
python main.py --source 0 --no-display \
    --save-video output/video/teste.mp4 --save-frames 10 \
    --tag baseline

# Comparativo de modelo (rodar em sequência)
python main.py --source 0 --no-display --yolo-model yolov8n-pose.pt --tag v8n_baseline
python main.py --source 0 --no-display --yolo-model yolov8s-pose.pt --tag v8s
python main.py --source 0 --no-display --yolo-model yolo26s-pose.pt --tag yolo26s_pedro
python main.py --source 0 --no-display --yolo-model yolov8n-pose.pt --imgsz 320 --tag v8n_320
```

**Por que `yolov8n-pose.pt` é o default?** Os notebooks de treino do GRU
(em `modelos/data_prep/` e `modelos/training/`) usaram esse modelo pra extrair
keypoints. Pra isolar a variável YOLO e estabelecer uma baseline limpa,
o bench começa com o mesmo modelo. Outros entram no protocolo comparativo
pra medir o impacto de drift de distribuição.

## Flags

| Flag | Default | Descrição |
|---|---|---|
| `--source` | `0` | Índice de webcam ou path de vídeo |
| `--yolo-model` | `yolov8n-pose.pt` | Modelo YOLO pose (baixa do CDN se não local) |
| `--yolo-conf` | `0.25` | Threshold de confiança do YOLO |
| `--imgsz` | `640` | Resolução de input do YOLO |
| `--device` | `cpu` | Device de inferência |
| `--tag` | `run` | Identificador da sessão nos outputs |
| `--no-display` | off | Pula cv2.imshow mesmo com DISPLAY setado |
| `--save-frames N` | off | Salva JPG a cada N segundos |
| `--save-video PATH` | off | Grava vídeo anotado |
| `--log-level` | `INFO` | DEBUG, INFO, WARNING, ERROR |

## Tecla durante execução

- `q` — encerra

## Análise dos CSVs

```bash
# Na Raspi → Windows
scp -r vigia@raspi:~/fall-classifier-bench/output/csv ./local-analysis/

# No Windows
jupyter lab analysis.ipynb
```

## Colunas do CSV

| Coluna | Descrição |
|---|---|
| `timestamp` | ISO 8601 com ms |
| `frame_idx` | Índice do frame desde o início |
| `person_id` | ID do tracker BotSort |
| `t_capture_ms` | Tempo de `cap.read()` |
| `t_yolo_ms` | Tempo de inferência YOLO pose |
| `t_gru_ms` | Tempo de inferência GRU ONNX |
| `n_valid_kpts` | Keypoints com conf > threshold (0-17) |
| `n_valid_frames_window` | Frames com joints obrigatórios na janela (0-20) |
| `label` | `ADL`, `FALL` ou `INVALID` (janela com keypoints insuficientes) |
| `prob_adl`, `prob_fall` | Probabilidades softmax do GRU |
| `alert` | 1 se 2 FALLs consecutivos confirmaram queda |
| `n_detections_total` | Pessoas detectadas no frame |
# fall-detection
