"""Replay saved detections through a tracker and write tracks for count.py.

    .venv-rf/bin/python src/retrack.py --dets outputs/dets_rf_medium.csv \
        --tracker bytetrack --out outputs/tracks_rf_byte.csv
"""
import argparse

import numpy as np
import pandas as pd
import supervision as sv
from trackers import ByteTrackTracker, OCSORTTracker


def make_tracker(name, frame_rate, activation=0.5):
    if name == "bytetrack":
        return ByteTrackTracker(lost_track_buffer=int(frame_rate * 3), frame_rate=frame_rate,
                                track_activation_threshold=activation,
                                high_conf_det_threshold=activation)
    if name == "ocsort":
        return OCSORTTracker(lost_track_buffer=int(frame_rate * 3), frame_rate=frame_rate)
    raise ValueError(name)


def drop_static_boxes(df, min_seconds, frame_rate, cell=24):
    """Drop boxes sitting in the same place too long: mannequins, reflections."""
    key = [(df.x1 + df.x2) // (2 * cell), df.y2 // cell, (df.y2 - df.y1) // (2 * cell)]
    frames = df.groupby(key).frame.transform("nunique")
    return df[frames < min_seconds * frame_rate]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "ocsort"])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--static-seconds", type=float, default=30)
    ap.add_argument("--activation", type=float, default=0.5, help="min score to start a track")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.dets)
    before = len(df)
    df = drop_static_boxes(df, args.static_seconds, args.fps / args.stride)
    print(f"static filter: {before} -> {len(df)} detections")
    tracker = make_tracker(args.tracker, args.fps / args.stride, args.activation)
    groups = {frame: g for frame, g in df.groupby("frame")}
    rows = []
    for frame in range(0, int(df.frame.max()) + 1, args.stride):
        g = groups.get(frame)
        if g is None:
            dets = sv.Detections.empty()
        else:
            dets = sv.Detections(xyxy=g[["x1", "y1", "x2", "y2"]].values.astype(float),
                                 confidence=g.conf.values.astype(float),
                                 class_id=np.zeros(len(g), dtype=int))
        out = tracker.update(dets)
        if out.tracker_id is None:
            continue
        for box, tid, conf in zip(out.xyxy, out.tracker_id, out.confidence):
            if tid >= 0:
                rows.append([frame, round(frame / args.fps, 3), int(tid),
                             *[round(float(v), 1) for v in box], round(float(conf), 3)])
    cols = ["frame", "t", "id", "x1", "y1", "x2", "y2", "conf"]
    pd.DataFrame(rows, columns=cols).to_csv(args.out, index=False)
    print(f"{len(rows)} rows, {len({r[2] for r in rows})} ids -> {args.out}")


if __name__ == "__main__":
    main()
