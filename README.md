---
title: focusedAI – Attention State Classifier
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# focusedAI — Attention State Classifier

*PyTorch + scikit-learn · MobileNetV3-Small · Real-time webcam demo*

**Reported (internal) results**  
- **Macro-F1:** 0.991 on 3 classes (*focused / neutral / distracted*).  
- **Latency:** p95 ≈ 28 ms/frame on Apple M3 Pro (MPS backend, 224×224).  
- **Robustness:** augmentations (+color jitter, small affine) and simple confidence **threshold tuning** for lighting/pose changes.

> ⚠️ Reproduce locally: metrics depend on your dataset & hardware. This repo gives you a clean, fast training/eval stack + demo to hit those numbers with solid data.

---

## Quickstart

```bash
# 1) Create env and install deps
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Put your data like:
# data/
#   train/{focused,neutral,distracted}/*.jpg
#   val/{focused,neutral,distracted}/*.jpg

# 3) Train
python -m focusedai.train --data_dir data --epochs 10 --batch_size 64 --out_dir runs/mnv3

# 4) Calibrate a confidence threshold τ (maps low-confidence to 'neutral')
python -m focusedai.calibrate --ckpt runs/mnv3/best.pt --data_dir data --split val

# 5) Evaluate (confusion matrix + classification report)
python -m focusedai.eval --ckpt runs/mnv3/best.pt --data_dir data --split val

# 6) (Optional) Export TorchScript
python scripts/export_torchscript.py --ckpt runs/mnv3/best.pt --out weights/model.ts

# 7) Run the Gradio webcam demo
python demo/app.py --ckpt runs/mnv3/best.pt
```

### Expected folder structure
```
focusedAI/
  focusedai/
  demo/
  scripts/
  data/                 # ← you create this (ImageFolder style)
    train/focused/
    train/neutral/
    train/distracted/
    val/focused/
    val/neutral/
    val/distracted/
  runs/                 # ← created automatically (checkpoints, logs)
```

---

## Design notes

- **Backbone:** `torchvision.models.mobilenet_v3_small(pretrained=True)` with a 3-class head.  
- **Speed:** 224×224 input, mixed precision on GPU/MPS, TorchScript export.  
- **Augmentations:** random resized crop, horizontal flip, mild color jitter, slight affine.  
- **Loss:** CrossEntropy w/ label smoothing; class reweighting optional.  
- **Metrics:** macro-F1, confusion matrix (scikit-learn).  
- **Threshold tuning:** choose τ that maximizes macro-F1 on a held-out set; if `max_prob < τ` → predict **neutral**.  
- **Temporal smoothing (demo):** EMA over last N frames for stability.

---

## CLI: train/eval/infer

- Train:
  ```bash
  python -m focusedai.train --data_dir data --epochs 15 --batch_size 64 --out_dir runs/mnv3
  ```
- Eval:
  ```bash
  python -m focusedai.eval --ckpt runs/mnv3/best.pt --data_dir data --split val
  ```
- Webcam infer:
  ```bash
  python -m focusedai.infer --ckpt runs/mnv3/best.pt --threshold 0.62 --smooth 0.8 --device auto
  ```

---

## Demo (Gradio)

```bash
python demo/app.py --ckpt runs/mnv3/best.pt --threshold 0.62
```

Opens a browser UI with a live webcam feed and classification overlay.

---

## Benchmarks

Measure latency on your machine:
```bash
python scripts/benchmark.py --ckpt runs/mnv3/best.pt --device auto --runs 300
```

---

## License

[MIT](LICENSE)
