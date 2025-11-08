import argparse, time, numpy as np, torch
from focusedai.model import load_model
from focusedai.utils import device_auto

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    ap.add_argument("--runs", type=int, default=300)
    return ap.parse_args()

def main():
    args = parse_args()
    device = device_auto() if args.device=="auto" else torch.device(args.device)
    model = load_model(args.ckpt, device=device)
    model.eval()
    x = torch.randn(1,3,224,224, device=device)

    # warmup
    for _ in range(20):
        with torch.no_grad():
            _ = model(x)

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        t1 = time.perf_counter()
        times.append((t1 - t0)*1000.0)  # ms

    p95 = np.percentile(times, 95)
    mean = np.mean(times)
    print({"mean_ms": float(mean), "p95_ms": float(p95), "runs": args.runs, "device": str(device)})

if __name__ == "__main__":
    main()
