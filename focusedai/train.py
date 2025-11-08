import os, argparse, json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from .data import make_loaders
from .model import create_model
from .utils import evaluate_logits, device_auto

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out_dir", type=str, default="runs/mnv3")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = device_auto() if args.device=="auto" else torch.device(args.device)

    train_loader, val_loader, idx_to_class = make_loaders(args.data_dir, args.batch_size, args.num_workers)
    num_classes = len(idx_to_class)

    model = create_model(num_classes=num_classes, pretrained=True, freeze_backbone=args.freeze_backbone).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    counts = torch.bincount(torch.tensor(train_loader.dataset.targets))
    weights = (1.0 / counts.float()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    scaler = GradScaler(enabled=(device.type in ["cuda","mps"]))

    best_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        running_loss = 0.0
        for imgs, targets in pbar:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type in ["cuda","mps"])):
                logits = model(imgs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(imgs)
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        macro_f1, cm, report = evaluate_logits(all_logits, all_targets)

        print("Val macro-F1:", macro_f1)
        print("Confusion matrix:\n", cm)
        print(report)

        # Save best
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            ckpt = {
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "epoch": epoch,
                "macro_f1": best_f1,
                "idx_to_class": idx_to_class,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": macro_f1})

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
