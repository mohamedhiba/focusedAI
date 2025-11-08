import argparse, time
import torch
import cv2
import numpy as np
from torchvision import transforms
from .model import load_model
from .utils import device_auto

LABELS_DEFAULT = ["distracted", "neutral", "focused"]  # ensure idx_to_class aligns

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--threshold", type=float, default=None, help="If max prob < τ, output 'neutral'.")
    ap.add_argument("--smooth", type=float, default=0.7, help="EMA smoothing [0..1], 0 disables.")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    ap.add_argument("--labels", type=str, default=None, help="Comma-separated labels override.")
    ap.add_argument("--camera", type=int, default=0)
    return ap.parse_args()

def main():
    args = parse_args()
    device = device_auto() if args.device=="auto" else torch.device(args.device)

    model = load_model(args.ckpt, device=device)
    neutral_index = 1  # by convention; override if needed
    labels = LABELS_DEFAULT
    # try to read labels mapping from checkpoint
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        idx_to_class = ckpt.get("idx_to_class", None)
        if isinstance(idx_to_class, dict):
            labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    except Exception:
        pass
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]

    tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    ema = None
    last = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = tfms(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        if args.smooth > 0.0:
            if ema is None:
                ema = probs
            else:
                ema = args.smooth * ema + (1 - args.smooth) * probs
            probs_show = ema
        else:
            probs_show = probs

        pred_idx = int(np.argmax(probs_show))
        maxp = float(np.max(probs_show))
        if args.threshold is not None and maxp < args.threshold:
            pred_idx = neutral_index
            maxp = float(probs_show[neutral_index])

        label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)

        now = time.time()
        dt = now - last
        last = now
        fps = 1.0 / max(dt, 1e-6)

        # overlay
        cv2.putText(frame, f"{label} ({maxp:.2f})  {fps:.1f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"{label} ({maxp:.2f})  {fps:.1f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("focusedAI — press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
