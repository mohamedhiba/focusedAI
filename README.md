---
title: focusedAI – Attention State Classifier
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---
# focusedAI — Real-Time Attention State Classifier

[![GitHub](https://img.shields.io/badge/GitHub-focusedAI-black?logo=github)](https://github.com/mohamedhiba/focusedAI)
[![Model – Hugging Face](https://img.shields.io/badge/Model-focusedAI--engage--v2-yellow?logo=huggingface)](https://huggingface.co/Mohamedhiba/focusedAI-engage-v2)
[![Live Demo – Gradio](https://img.shields.io/badge/Live%20Demo-HF%20Spaces-blue?logo=databricks)](https://huggingface.co/spaces/Mohamedhiba/focusedAI-demo)

**focusedAI** is a small, fast **attention state classifier** that runs in real time on webcam video and predicts:

- **focused**
- **neutral**
- **distracted**

It’s built around a **MobileNetV3-Small** backbone in PyTorch, plus calibration and light “decision policy” logic to make the live behavior stable enough to demo.

---

## ✨ Highlights

- **3-class attention detection**: `focused / neutral / distracted`
- **Backbone**: MobileNetV3-Small (torchvision)
- **Latency**: ~4–5 ms p95 on Apple M3 Pro (MPS) at 224×224
- **Metrics (validation)** on a student engagement dataset:
  - **Macro-F1 ≈ 0.918**
  - High precision/recall for focused/distracted; neutral is tuned to catch idle/drowsy
- **Calibrated**:
  - Softmax **temperature scaling**
  - Per-class **logit bias** to fight “everything looks neutral”
  - Optional **threshold policy** and **hysteresis** to avoid flicker
- **End-to-end demo**:
  - Gradio webcam app
  - Hosted on **Hugging Face Spaces**
  - `app.py` auto-downloads the model from Hugging Face, no weights in repo

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/mohamedhiba/focusedAI.git
cd focusedAI

python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### **2. Run the live demo (local)**

  

The repo’s root app.py is set up as the **Space entrypoint**, but it also works locally:

```
export PYTHONPATH=$PWD

# Uses HF model "Mohamedhiba/focusedAI-engage-v2" and best.pt by default
python app.py
```

If you want to skip the HF download and point directly to a local checkpoint:

```
export PYTHONPATH=$PWD
python demo/app.py --ckpt runs/engage_v2/best.pt \
  --crop none --preproc center --policy argmax --smooth 0.0 \
  --temp 0.60 --bias_f 0.50 --bias_d 0.10 --bias_n -0.40
```

### **3. Try the hosted demo**

  

> **Live Gradio demo (Hugging Face Spaces):**

> https://huggingface.co/spaces/Mohamedhiba/focusedAI-demo

  

No setup needed — just open the link and allow webcam access.

---

## **📊 Results**

  

The final model (engage_v2/best.pt) is trained on a curated subset of a **student engagement** dataset. The original dataset has several engagement-related categories; they are remapped to 3 classes:

- **focused**: directly engaged/attentive faces
    
- **distracted**: looking away, clearly not on task, confused/frustrated
    
- **neutral**: idle/drowsy, low arousal but not obviously engaged or distracted
    

  

Validation performance (3-class, macro F1):

```
Confusion matrix (val)
======================

[[246   0  30]
 [  0 178   9]
 [  0   0  80]]

distracted: P=1.00, R=0.89, F1=0.94
focused:    P=1.00, R=0.95, F1=0.98
neutral:    P=0.67, R=1.00, F1=0.80
macro-F1:   0.918
```

Latency benchmark (scripts/benchmark.py) on an M3 Pro (MPS):

- **mean** ≈ 4 ms
    
- **p95** ≈ 5 ms
    
    (1×224×224 RGB input, batch size 1, MobileNetV3-Small + 3-class head)
    

---

## **🧱 Repository Structure**

```
focusedAI/
├─ app.py                  # HF Spaces / local demo entrypoint (downloads weights)
├─ README.md
├─ requirements.txt
├─ LICENSE
│
├─ focusedai/              # Python package
│  ├─ __init__.py
│  ├─ model.py             # MobileNetV3-Small + classifier head, load/save
│  ├─ data.py              # Dataset + transforms (ImageFolder, train/val loaders)
│  ├─ train.py             # Training loop (PyTorch + sklearn metrics)
│  ├─ eval.py              # Evaluation, confusion matrix, macro-F1
│  ├─ calibrate.py         # Threshold search / calibration helpers
│  ├─ infer.py             # Simple inference utilities
│  └─ utils.py             # Device selection, seeding, misc helpers
│
├─ demo/
│  ├─ __init__.py
│  └─ app.py               # Gradio webcam demo (policy logic etc.)
│
└─ scripts/
   ├─ __init__.py
   ├─ benchmark.py         # Micro-benchmark of forward pass (latency)
   ├─ collect.py           # Script to collect webcam data to folders
   ├─ export_torchscript.py# Optional TorchScript export (for deployment)
   └─ prepare_statefarm.py # Helper for StateFarm distracted driver dataset (unused in final model)
```

> The project evolved through several experiments (StateFarm, custom webcam captures, student engagement datasets). The final engage_v2 checkpoint uses the **student engagement** mapping with focused/neutral/distracted.

---

## **🧠 Model & Training**

  

### **Architecture**

- **Backbone:** torchvision.models.mobilenet_v3_small
    
- **Classifier head:** replaces final linear layer with a 3-class head
    
- **Input:** 224×224 RGB
    
- **Transforms:**
    
    - Resize → CenterCrop (or SquarePad + Resize)
        
    - Random horizontal flip, light color jitter
        
    - Normalize (ImageNet mean/std)
        
    

  

### **Training loop**

  

Training script:

```
export PYTHONPATH=$PWD

python -m focusedai.train \
  --data_dir data_engagement_v2 \
  --epochs 6 \
  --batch_size 64 \
  --out_dir runs/engage_v2
```

General settings:

- Loss: Cross-entropy
    
- Optimizer: AdamW
    
- Scheduler: simple step / cosine (configurable)
    
- Device: --device auto chooses CUDA / MPS / CPU
    
- Metrics: macro-F1, per-class precision/recall, confusion matrix
    
- Best checkpoint: saves to <out_dir>/best.pt based on validation macro-F1
    

  

### **Dataset layout**

  

Training assumes a standard ImageFolder structure:

```
data_engagement_v2/
├─ train/
│  ├─ distracted/
│  ├─ focused/
│  └─ neutral/
└─ val/
   ├─ distracted/
   ├─ focused/
   └─ neutral/
```

You can plug in any dataset that you can reorganize into this structure.

---

## **🎯 Calibration & Evaluation**

  

The training script already prints validation stats at the end of every epoch, plus a final confusion matrix.

  

There’s also a small calibration utility (focusedai/calibrate.py) to search for a threshold that maximizes **macro-F1** on the validation set:

```
export PYTHONPATH=$PWD

python -m focusedai.calibrate \
  --ckpt runs/engage_v2/best.pt \
  --data_dir data_engagement_v2 \
  --split val \
  --neutral_index 2 \
  --out runs/engage_v2/threshold.json
```

Evaluation (full metrics + confusion matrix):

```
python -m focusedai.eval \
  --ckpt runs/engage_v2/best.pt \
  --data_dir data_engagement_v2 \
  --split val \
  --neutral_index 2 \
  --threshold $(python - <<'PY'
import json
print(json.load(open("runs/engage_v2/threshold.json"))["best_threshold"])
PY
)
```

---

## **⚙️ Inference & Demo Details**

  

### **Inference policy (demo/app.py)**

  

The raw model gives logits. The demo applies:

1. **Softmax temperature** (--temp):
    
    - <1.0 sharpens (makes the top class more confident)
        
    - > 1.0 softens (more uncertain)
        
    
2. **Per-class logit bias** (--bias_f, --bias_d, --bias_n):
    
    These are added _before_ softmax. Example defaults:
    

```
--temp 0.60 \
--bias_f 0.50 \
--bias_d 0.10 \
--bias_n -0.40
```

2. That slightly favors focused and distracted and penalizes neutral, which helps combat “everything looks neutral” behavior on noisy webcam frames.
    
3. **Policy**:
    
    - argmax : pick the highest probability class (with the above adjustments)
        
    - thresholds : optional threshold-based decision with a neutral fallback
        
    
4. **Smoothing / hysteresis**:
    
    - EMA smoothing (--smooth) over probabilities
        
    - Simple hysteresis: requires a small confidence margin to switch classes, reducing jitter when probabilities are close
        
    

  

### **Webcam preprocessing**

  

The demo uses gr.Image(source="webcam") and:

- Optionally crops to the largest face region (Haar cascade)
    
- Or uses center crop / square pad to 224×224
    
- Normalizes the same way as training
    

---

## **🧪 Benchmarking**

  

Use scripts/benchmark.py to approximate throughput / latency:

```
export PYTHONPATH=$PWD

python scripts/benchmark.py \
  --ckpt runs/engage_v2/best.pt \
  --device auto \
  --runs 300
```

Output example:

```
{'mean_ms': 4.2, 'p95_ms': 4.6, 'runs': 300, 'device': 'mps'}
```

This was measured on an Apple M3 Pro using MPS. CPU is still fast enough for the live demo.

---

## **🧭 Design Choices & Lessons**

- **Small backbone**
    
    MobileNetV3-Small is a good tradeoff between speed and accuracy for webcam-sized inputs.
    
- **3 classes instead of many**
    
    Attention is inherently fuzzy; keeping classes at focused / neutral / distracted is more useful than over-segmenting into a large number of nearly overlapping labels.
    
- **Neutral drift in webcam data**
    
    A model trained on curated images tends to be over-cautious on shaky webcam frames, often defaulting to “neutral”. Temperature + bias + tie-breaking logic at inference time helps anchor the live behavior closer to how humans perceive attention.
    
- **No weights in repo**
    
    The final checkpoint (best.pt) is stored in a **Hugging Face model repo**, and the demo downloads it on startup. This keeps the GitHub repo light and avoids large-file issues.
    

---

## **🔧 Development / How to Extend**

  

Some ideas for improving or extending focusedAI:

- **Fine-tune on your own webcam**
    
    Capture a few hundred frames per class using scripts/collect.py, add them into the dataset, and run a short fine-tune (1–2 epochs) on top of engage_v2 to adapt to your environment.
    
- **Better distractions**
    
    Blend in samples from driving/phone datasets (e.g. StateFarm) into the distracted class, or add a separate “phone” label if you expand the head.
    
- **Export to ONNX / mobile**
    
    Use scripts/export_torchscript.py as a starting point, or export to ONNX and run in a mobile app.
    
- **Landmarks / gaze**
    
    Integrate a lightweight facial landmarks or gaze estimator to improve distinctions between neutral vs. distracted.
    

---

## **🧾 License**

  

This project is licensed under the **MIT License**. See LICENSE for details.

---

## **📌 Citation**

  

If you use _focusedAI_ in your work or portfolio:

```
@software{hiba2025focusedai,
  author       = {Mohamed E. Hiba},
  title        = {focusedAI: Real-time Attention State Classifier},
  year         = {2025},
  url          = {https://github.com/mohamedhiba/focusedAI}
}
```

