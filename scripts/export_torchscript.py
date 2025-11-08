import argparse, torch
from focusedai.model import load_model

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.ckpt, device=device)
    model.eval()
    example = torch.randn(1,3,224,224, device=device)
    traced = torch.jit.trace(model, example)
    traced.save(args.out)
    print("Saved TorchScript to", args.out)

if __name__ == "__main__":
    main()
