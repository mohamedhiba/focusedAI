import argparse, os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .model import load_model
from .utils import evaluate_logits, device_auto

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["val","train"])
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--neutral_index", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    return ap.parse_args()

def main():
    args = parse_args()
    device = device_auto() if args.device == "auto" else torch.device(args.device)

    size = 224
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    ds = datasets.ImageFolder(os.path.join(args.data_dir, args.split), transform=val_tfms)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = load_model(args.ckpt, device=device)

    all_logits = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for imgs, targets in dl:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_targets.append(targets)

    logits = torch.cat(all_logits, 0)
    targets = torch.cat(all_targets, 0)

    macro_f1, cm, report = evaluate_logits(logits, targets, threshold=args.threshold, neutral_index=args.neutral_index)
    print("Macro-F1:", macro_f1)
    print("Confusion matrix:\n", cm)
    print(report)

if __name__ == "__main__":
    main()
