"""Score an events CSV against hand-reviewed crossings: one-to-one match in time.

    python src/evaluate.py --events outputs/events.csv
"""
import argparse

import pandas as pd


def score(events, labels, tolerance=2.5):
    out = {}
    for direction in ("IN", "OUT"):
        pred = events[events.direction == direction].t.tolist()
        truth = sorted(labels[labels.direction == direction].t)
        used, tp = set(), 0
        for t in sorted(pred):
            free = [(abs(g - t), i) for i, g in enumerate(truth) if i not in used and abs(g - t) <= tolerance]
            if free:
                used.add(min(free)[1])
                tp += 1
        out[direction] = {"pred": len(pred), "truth": len(truth), "tp": tp,
                          "precision": tp / max(len(pred), 1), "recall": tp / max(len(truth), 1),
                          "missed": [round(g) for i, g in enumerate(truth) if i not in used]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--labels", default="labels/dev_crossings_reviewed.csv")
    args = ap.parse_args()
    for direction, r in score(pd.read_csv(args.events), pd.read_csv(args.labels)).items():
        print(f"{direction:<3} pred {r['pred']:>3}  truth {r['truth']:>3}  tp {r['tp']:>3}  "
              f"precision {r['precision']:.1%}  recall {r['recall']:.1%}  missed {r['missed']}")


if __name__ == "__main__":
    main()
