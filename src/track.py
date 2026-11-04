"""Run person detection + tracking on the whole video and save raw tracks to CSV.

Slow step. Run once per model; counting (count.py) replays the CSV in seconds.

    python src/track.py --model yolo11l.pt --out outputs/tracks_yolo11l.csv
"""
import argparse
import csv
import time

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/entrance.mp4")
    ap.add_argument("--model", default="yolo11l.pt")
    ap.add_argument("--out", default="outputs/tracks.csv")
    ap.add_argument("--stride", type=int, default=2, help="process every Nth frame")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU; higher keeps overlapping people")
    ap.add_argument("--tracker", default="configs/botsort_entrance.yaml")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{args.video}: {total} frames at {fps:.2f} fps")

    model = YOLO(args.model)
    start = time.time()
    frame_idx = -1
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "t", "id", "x1", "y1", "x2", "y2", "conf"])
        while True:
            ok = cap.grab()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % args.stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            result = model.track(
                frame, persist=True, classes=[0], conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                tracker=args.tracker, device=args.device, verbose=False,
            )[0]
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                for box, tid, conf in zip(boxes, ids, confs):
                    writer.writerow([frame_idx, round(frame_idx / fps, 3), tid,
                                     *[round(float(v), 1) for v in box], round(float(conf), 3)])
            if frame_idx % (args.stride * 500) == 0:
                f.flush()
                rate = (frame_idx + 1) / (time.time() - start)
                print(f"{frame_idx}/{total}  {rate:.1f} video-fps  "
                      f"eta {(total - frame_idx) / rate / 60:.1f} min", flush=True)
    print(f"Done in {(time.time() - start) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
