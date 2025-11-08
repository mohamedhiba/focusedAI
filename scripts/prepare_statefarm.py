# scripts/prepare_statefarm.py
import os, argparse, random, shutil, glob
random.seed(42)

def copy_subset(src_dir, dst_dir, limit=None):
    os.makedirs(dst_dir, exist_ok=True)
    files = glob.glob(os.path.join(src_dir, "*.jpg"))
    if limit: files = random.sample(files, min(limit, len(files)))
    for f in files:
        shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/statefarm/imgs/train", help="root with c0..c9 folders")
    ap.add_argument("--out",  default="data", help="output ImageFolder root")
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--limit_per_class", type=int, default=1500, help="limit to keep it light; set None for all")
    ap.add_argument("--neutral_dir", type=str, default=None, help="optional directory of your webcam 'neutral' frames")
    args = ap.parse_args()

    classes = [f"c{i}" for i in range(10)]
    for c in classes:
        assert os.path.isdir(os.path.join(args.root, c)), f"missing {c}"

    # Map
    focused_src = os.path.join(args.root, "c0")
    distracted_srcs = [os.path.join(args.root, f"c{i}") for i in range(1,10)]

    # Create splits
    for split in ["train", "val"]:
        for k in ["focused","neutral","distracted"]:
            os.makedirs(os.path.join(args.out, split, k), exist_ok=True)

    # Focused
    f_all = glob.glob(os.path.join(focused_src, "*.jpg"))
    random.shuffle(f_all)
    if args.limit_per_class: f_all = f_all[:args.limit_per_class]
    n_val = int(len(f_all)*args.val_split)
    f_val, f_train = f_all[:n_val], f_all[n_val:]
    for src, split in [(f_train,"train"), (f_val,"val")]:
        for p in src:
            shutil.copy2(p, os.path.join(args.out, split, "focused", os.path.basename(p)))

    # Distracted (merge c1..c9)
    d_all = []
    for dsrc in distracted_srcs:
        files = glob.glob(os.path.join(dsrc, "*.jpg"))
        if args.limit_per_class: files = files[:args.limit_per_class]
        d_all += files
    random.shuffle(d_all)
    n_val = int(len(d_all)*args.val_split)
    d_val, d_train = d_all[:n_val], d_all[n_val:]
    for src, split in [(d_train,"train"), (d_val,"val")]:
        for p in src:
            shutil.copy2(p, os.path.join(args.out, split, "distracted", os.path.basename(p)))

    # Neutral
    if args.neutral_dir and os.path.isdir(args.neutral_dir):
        n_all = glob.glob(os.path.join(args.neutral_dir, "*.jpg"))
        random.shuffle(n_all)
        # keep size similar-ish to focused to avoid imbalance
        target = min(len(f_all), len(n_all))
        n_all = n_all[:target]
        n_val = int(len(n_all)*args.val_split)
        n_val_split, n_train_split = n_all[:n_val], n_all[n_val:]
        for src, split in [(n_train_split,"train"), (n_val_split,"val")]:
            for p in src:
                shutil.copy2(p, os.path.join(args.out, split, "neutral", os.path.basename(p)))
    else:
        print("No neutral_dir provided — the 'neutral' class will be small/empty. "
              "You can still rely on threshold gating to produce 'neutral' at runtime.")

    print("Prepared:", args.out)

if __name__ == "__main__":
    main()
