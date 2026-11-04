"""Replay saved tracks and count IN/OUT crossings of the entrance zone.

The zone comes from the red box in the client's screenshot (1920x1080), scaled
to the 1280x720 stream. A person counts when the foot point goes all the way
through the zone: top to bottom is IN, bottom to top is OUT.

Cleaning steps before counting:
  1. drop static tracks (mannequins)
  2. drop duplicate tracks (two boxes on one person)
  3. stitch fragments of one person into a chain
Tracks that spend 15+ seconds inside the zone are ignored.

    python src/count.py --tracks outputs/tracks_yolo11l.csv
"""
import argparse

import numpy as np
import pandas as pd

ZONE_X1, ZONE_X2 = 241, 1223
ZONE_Y1, ZONE_Y2 = 245, 341
GT_IN, GT_OUT = 72, 74


def load(path, min_conf):
    df = pd.read_csv(path)
    df = df[df.conf >= min_conf].copy()
    df["cx"] = (df.x1 + df.x2) / 2
    return df


def drop_static(df, min_move=30):
    g = df.groupby("id")
    moving = (g.cx.max() - g.cx.min() > min_move) | (g.y2.max() - g.y2.min() > min_move)
    return df[df.id.isin(moving[moving].index)]


def iou(a, b):
    ix = np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0])
    iy = np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1])
    inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a) + area(b) - inter + 1e-6)


def drop_duplicates(df, min_iou=0.6, min_share=0.7):
    """Drop a track when most of its frames sit on top of a longer track."""
    cols = ["x1", "y1", "x2", "y2"]
    m = df.merge(df, on="frame", suffixes=("_a", "_b"))
    m = m[m.id_a != m.id_b]
    m["iou"] = iou(m[[c + "_a" for c in cols]].values, m[[c + "_b" for c in cols]].values)
    overlap = m[m.iou >= min_iou].groupby(["id_a", "id_b"]).size()
    length = df.groupby("id").size()
    drop = set()
    for (a, b), n in overlap.items():
        if a not in drop and b not in drop and length[a] <= length[b] and n / length[a] >= min_share:
            drop.add(a)
    return df[~df.id.isin(drop)]


def stitch(df, max_gap=3.0, max_dist=160):
    """Greedily link a track end to the nearest later track start."""
    ends = df.sort_values("t").groupby("id").agg(
        start=("t", "first"), end=("t", "last"), sx=("cx", "first"), sy=("y2", "first"),
        ex=("cx", "last"), ey=("y2", "last"))
    pairs = []
    for a, ra in ends.iterrows():
        for b, rb in ends.iterrows():
            gap = rb.start - ra.end
            if a != b and 0 <= gap <= max_gap:
                dist = np.hypot(rb.sx - ra.ex, rb.sy - ra.ey)
                if dist <= max_dist:
                    pairs.append((dist + 30 * gap, a, b))
    parent, used_a, used_b = {}, set(), set()
    for _, a, b in sorted(pairs):
        if a not in used_a and b not in used_b:
            parent[b] = a
            used_a.add(a)
            used_b.add(b)

    def root(i):
        while i in parent:
            i = parent[i]
        return i

    out = df.copy()
    out["chain"] = out.id.map(root)
    return out


def count_chain(track, top, bottom, max_zone_seconds):
    events = []
    last, entered = None, None
    for row in track.itertuples():
        if last is None and not ZONE_X1 <= row.cx <= ZONE_X2:
            continue
        current = "mall" if row.y2 < top else "store" if row.y2 > bottom else None
        if current is None:
            if entered is None:
                entered = row.t
            continue
        in_zone = 0 if entered is None else row.t - entered
        entered = None
        in_x = ZONE_X1 <= row.cx <= ZONE_X2
        if last is not None and current != last and in_x and in_zone < max_zone_seconds:
            events.append(("IN" if current == "store" else "OUT", row.t))
        last = current
    return events


def collapse(events, window=5):
    """Cancel opposite crossings close in time: someone hovering on the line."""
    kept = []
    for direction, t in events:
        if kept and kept[-1][0] != direction and t - kept[-1][1] <= window:
            kept.pop()
        else:
            kept.append((direction, t))
    return kept


def tally(df, key, args):
    rows = []
    for chain, track in df.sort_values("t").groupby(key):
        for direction, t in collapse(count_chain(track, args.top, args.bottom, args.dwell)):
            rows.append({"chain": chain, "direction": direction, "t": t})
    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)


def report(name, events):
    n_in, n_out = (events.direction == "IN").sum(), (events.direction == "OUT").sum()
    print(f"{name:<22} IN {n_in:>3}/{GT_IN}  OUT {n_out:>3}/{GT_OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="outputs/tracks_yolo11l.csv")
    ap.add_argument("--min-conf", type=float, default=0.25)
    ap.add_argument("--top", type=float, default=ZONE_Y1)
    ap.add_argument("--bottom", type=float, default=ZONE_Y2)
    ap.add_argument("--dwell", type=float, default=15)
    ap.add_argument("--events", default="outputs/events.csv")
    args = ap.parse_args()

    df = load(args.tracks, args.min_conf)
    report("raw", tally(df, "id", args))
    df = drop_static(df)
    report("- static", tally(df, "id", args))
    df = drop_duplicates(df)
    report("- duplicates", tally(df, "id", args))
    df = stitch(df)
    events = tally(df, "chain", args)
    report("+ stitched", events)
    events.to_csv(args.events, index=False)
    df.to_csv(args.events.replace(".csv", "_tracks.csv"), index=False)


if __name__ == "__main__":
    main()
