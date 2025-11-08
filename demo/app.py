# demo/app.py — robust webcam demo for focusedAI
# - optional Haar face-crop or no crop
# - center/square preproc (match training)
# - argmax vs thresholds policy
# - temperature + per-class logit bias (tilt away from neutral)
# - EMA smoothing + hysteresis
# - neutral tie-break + display gap (avoid equal % in UI)

import argparse
import numpy as np
import torch, cv2, gradio as gr
from PIL import Image, ImageOps
from torchvision import transforms
from focusedai.model import load_model
from focusedai.utils import device_auto

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])

    # Decision policy
    ap.add_argument("--policy", type=str, default="argmax", choices=["argmax","thresholds"])
    ap.add_argument("--tau_f", type=float, default=0.45)   # used if policy=thresholds
    ap.add_argument("--tau_d", type=float, default=0.45)   # used if policy=thresholds
    ap.add_argument("--smooth", type=float, default=0.0)   # EMA factor [0..1], 0 disables

    # Inference re-calibration
    ap.add_argument("--temp", type=float, default=0.75, help="softmax temperature (<1 sharpens)")
    ap.add_argument("--bias_f", type=float, default=0.20)
    ap.add_argument("--bias_d", type=float, default=0.10)
    ap.add_argument("--bias_n", type=float, default=-0.25)

    # Preproc & cropping
    ap.add_argument("--crop", type=str, default="none", choices=["haar","none"])
    ap.add_argument("--preproc", type=str, default="center", choices=["center","square"])
    ap.add_argument("--ensemble", action="store_true", help="avg(face, full) predictions")
    ap.add_argument("--gamma", type=float, default=1.0, help=">=0.7 brighten; <=1 darken")
    ap.add_argument("--min_face", type=int, default=60)
    ap.add_argument("--margin", type=float, default=0.30)

    # Tie-handling & display
    ap.add_argument("--tie_eps", type=float, default=0.05, help="neutral tie-break margin")
    ap.add_argument("--display_gap", type=float, default=0.04, help="boost winner prob for UI")
    return ap.parse_args()

# ---- Utils ----
_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def crop_face_or_center(rgb, min_face=60, margin=0.30):
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(min_face, min_face))
    if len(faces):
        x, y, fw, fh = max(faces, key=lambda r: r[2]*r[3])
        x1 = max(0, int(x - margin*fw)); y1 = max(0, int(y - margin*fh))
        x2 = min(w, int(x + fw + margin*fw)); y2 = min(h, int(y + fh + margin*fh))
        roi = rgb[y1:y2, x1:x2]
        if roi.size > 0:
            return roi
    return rgb

class SquarePad:
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h: return img
        if w > h:
            pad = (0, (w-h)//2, 0, (w-h)-(w-h)//2)
        else:
            pad = ((h-w)//2, 0, (h-w)-(h-w)//2, 0)
        return ImageOps.expand(img, border=pad, fill=0)

def make_tfms(preproc="center"):
    ops = [transforms.ToPILImage()]
    if preproc == "square":
        ops += [SquarePad(), transforms.Resize((224, 224))]
    else:
        ops += [transforms.Resize(256), transforms.CenterCrop(224)]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ]
    return transforms.Compose(ops)

def apply_gamma(rgb, gamma):
    if gamma == 1.0: return rgb
    x = rgb.astype(np.float32)/255.0
    x = np.clip(x, 1e-6, 1.0) ** gamma
    return np.clip(x*255.0, 0, 255).astype(np.uint8)

def softmax_np(z):
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / np.sum(ez)

# ---- Pipeline builder ----
def make_pipeline(model, device, labels, *,
                  policy, tau_f, tau_d, smooth,
                  temp, bias_f, bias_d, bias_n,
                  crop_mode, preproc, ensemble, gamma, min_face, margin,
                  tie_eps, display_gap):
    tfms = make_tfms(preproc)
    i_d = labels.index("distracted") if "distracted" in labels else 0
    i_f = labels.index("focused")    if "focused"    in labels else 1
    i_n = labels.index("neutral")    if "neutral"    in labels else 2

    ema = [None]
    last_label = [None]
    last_conf  = [0.0]
    SWITCH = 0.05
    bias_vec = None  # built on first call

    @torch.no_grad()
    def predict(img):
        nonlocal bias_vec
        if img is None:
            return None

        rgb = apply_gamma(img, gamma)

        # Build inputs
        roi = crop_face_or_center(rgb, min_face=min_face, margin=margin) if crop_mode=="haar" else rgb
        x_face = tfms(roi).unsqueeze(0).to(device)
        log_face = model(x_face)[0].detach().cpu().numpy()

        if ensemble:
            x_full = tfms(rgb).unsqueeze(0).to(device)
            log_full = model(x_full)[0].detach().cpu().numpy()
        else:
            log_full = None

        # Bias vector (per-class) and temperature
        if bias_vec is None:
            bias_vec = np.zeros_like(log_face, dtype=np.float32)
            bias_vec[i_f] += bias_f
            bias_vec[i_d] += bias_d
            bias_vec[i_n] += bias_n

        p1 = softmax_np((log_face + bias_vec) / max(temp, 1e-6))
        if ensemble:
            p2 = softmax_np((log_full + bias_vec) / max(temp, 1e-6))
            probs = 0.6*p1 + 0.4*p2
        else:
            probs = p1

        # EMA smoothing
        if smooth > 0.0:
            if ema[0] is None: ema[0] = probs
            else:              ema[0] = smooth*ema[0] + (1.0-smooth)*probs
            p = ema[0]
        else:
            p = probs

        # Decision
        if policy == "argmax":
            order = np.argsort(p)
            idx_top    = int(order[-1])
            idx_second = int(order[-2])
            # neutral tie-break
            if labels[idx_top] == "neutral" and (p[idx_top] - p[idx_second]) < tie_eps:
                idx_top = idx_second
            label = labels[idx_top]
            conf  = float(p[idx_top])
        else:
            pf, pd, pn = float(p[i_f]), float(p[i_d]), float(p[i_n])
            label, conf = "neutral", pn
            if pf >= tau_f and pf >= pd: label, conf = "focused", pf
            elif pd >= tau_d and pd > pf: label, conf = "distracted", pd
            # optional tie-break even in thresholds mode
            order = np.argsort(p); idx_top, idx_second = int(order[-1]), int(order[-2])
            if label == "neutral" and (p[idx_top] - p[idx_second]) < tie_eps:
                label = labels[idx_second]; conf = float(p[idx_second])

        # Hysteresis
        if last_label[0] is not None and label != last_label[0] and conf < last_conf[0] + SWITCH:
            label, conf = last_label[0], last_conf[0]
        last_label[0], last_conf[0] = label, conf

        # Display: widen the gap for readability
        q = p.copy()
        idx_label = labels.index(label)
        q[idx_label] = min(0.999, q[idx_label] + display_gap)
        q = q / q.sum()

        return {labels[i]: float(q[i]) for i in range(len(labels))}

    return predict

# ---- Main ----
def main():
    args = parse_args()
    device = device_auto() if args.device=="auto" else torch.device(args.device)
    model = load_model(args.ckpt, device=device)

    labels = ["distracted","focused","neutral"]
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        idx_to_class = ckpt.get("idx_to_class", None)
        if isinstance(idx_to_class, dict):
            labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    except Exception:
        pass

    fn = make_pipeline(
        model, device, labels,
        policy=args.policy, tau_f=args.tau_f, tau_d=args.tau_d, smooth=args.smooth,
        temp=args.temp, bias_f=args.bias_f, bias_d=args.bias_d, bias_n=args.bias_n,
        crop_mode=args.crop, preproc=args.preproc, ensemble=args.ensemble,
        gamma=args.gamma, min_face=args.min_face, margin=args.margin,
        tie_eps=args.tie_eps, display_gap=args.display_gap
    )

    with gr.Blocks() as demo:
        gr.Markdown("# focusedAI — Attention State Classifier")
        gr.Markdown("Center/square preproc · optional Haar crop/ensemble · argmax/thresholds · temp+bias · EMA+hysteresis")
        cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam",
                       type="numpy", webcam_options=gr.WebcamOptions(mirror=True))
        out = gr.Label(num_top_classes=3)
        cam.stream(fn, inputs=cam, outputs=out)
    demo.launch()

if __name__ == "__main__":
    main()
