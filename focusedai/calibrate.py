import argparse, os, numpy as np, json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
from .model import load_model
from .utils import device_auto

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["val","train"])
    ap.add_argument("--neutral_index", type=int, default=1, help="Class index to map low-confidence predictions to.")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    ap.add_argument("--out", type=str, default=None, help="Path to write best threshold json.")
    return ap.parse_args()

def main():
    args = parse_args()
    device = device_auto() if args.device == "auto" else torch.device(args.device)

    tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(os.path.join(args.data_dir, args.split), transform=tfms)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = load_model(args.ckpt, device=device)

    all_probs = []
    all_targets = []
    with torch.no_grad():
        for imgs, targets in dl:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
    import numpy as np
    P = np.concatenate(all_probs, 0)
    y = np.concatenate(all_targets, 0)

    best_tau, best_f1 = None, -1.0
    for tau in np.linspace(0.33, 0.95, 63):
        # If max prob < tau => neutral
        preds = P.argmax(axis=1)
        maxp = P.max(axis=1)
        preds = np.where(maxp < tau, args.neutral_index, preds)
        f1 = f1_score(y, preds, average="macro")
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)

    result = {"best_threshold": best_tau, "val_macro_f1_at_best_threshold": best_f1, "neutral_index": args.neutral_index}
    print(result)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
