# scripts/collect.py
import os, cv2, time, argparse
CLASSES = ["distracted", "neutral", "focused"]

def ensure_dirs(root, split):
    for c in CLASSES:
        os.makedirs(os.path.join(root, split, c), exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data", help="root data dir")
    ap.add_argument("--split", type=str, default="train", choices=["train","val"])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--prefix", type=str, default="")
    args = ap.parse_args()

    ensure_dirs(args.out, "train")
    ensure_dirs(args.out, "val")
    split = args.split

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("\nControls:")
    print("  f / n / d  -> save frame as focused / neutral / distracted")
    print("  t / v      -> switch split to train / val (current: %s)" % split)
    print("  q          -> quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        cv2.putText(frame, f"SPLIT: {split}  (t/v to toggle)", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"SPLIT: {split}  (t/v to toggle)", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, "f=focused  n=neutral  d=distracted  q=quit",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("focusedAI collector", frame)
        k = cv2.waitKey(1) & 0xFF
        ts = int(time.time()*1000)

        if k in (ord('q'), 27):
            break
        elif k == ord('t'):
            split = "train"
        elif k == ord('v'):
            split = "val"
        elif k in (ord('f'), ord('n'), ord('d')):
            label = {ord('d'):"distracted", ord('n'):"neutral", ord('f'):"focused"}[k]
            out_dir = os.path.join(args.out, split, label)
            os.makedirs(out_dir, exist_ok=True)
            name = f"{args.prefix}{label}_{ts}.jpg"
            path = os.path.join(out_dir, name)
            cv2.imwrite(path, frame)
            print("saved:", path)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
