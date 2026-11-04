"""Run RF-DETR over the video and save raw person detections to CSV.

Slow step. retrack.py replays the CSV through any tracker in seconds.

    .venv-rf/bin/python src/detect_rf.py --size medium --out outputs/dets_rf_medium.csv
"""
import argparse
import csv
import time

import cv2
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall

MODELS = {"small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}
PERSON = 1  # COCO category id used by RF-DETR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/entrance.mp4")
    ap.add_argument("--size", default="medium", choices=MODELS)
    ap.add_argument("--weights", default=None, help="fine-tuned checkpoint")
    ap.add_argument("--person-id", type=int, default=PERSON)
    ap.add_argument("--out", default="outputs/dets_rf.csv")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kwargs = {"pretrain_weights": args.weights} if args.weights else {}
    model = MODELS[args.size](**kwargs)
    start = time.time()
    frame_idx = -1
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "t", "x1", "y1", "x2", "y2", "conf"])
        while cap.grab():
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
            if frame_idx % args.stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            dets = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=args.conf)
            dets = dets[dets.class_id == args.person_id]
            for box, conf in zip(dets.xyxy, dets.confidence):
                writer.writerow([frame_idx, round(frame_idx / fps, 3),
                                 *[round(float(v), 1) for v in box], round(float(conf), 3)])
            if frame_idx % (args.stride * 500) == 0:
                f.flush()
                rate = (frame_idx + 1) / (time.time() - start)
                print(f"{frame_idx}/{total}  {rate:.1f} video-fps  "
                      f"eta {(total - frame_idx) / rate / 60:.1f} min", flush=True)
    print(f"Done in {(time.time() - start) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
