#!/usr/bin/env python3
"""
Zero-shot cross-domain evaluation of all arms on the SMD eval set.

Loads the CITRA-3D-Real checkpoint of each experimental arm and evaluates it,
WITHOUT any fine-tuning, on the cleaned SMD set produced by smd_prepare.py.
Reports mAP50 / mAP50-95 as mean +/- std across seeds and writes a per-seed
JSON. Produces the numbers in the cross-domain results table of the paper.

IMPORTANT:
  - The arm -> checkpoint mapping (ARMS) matches the authors' run-directory
    layout. Adjust it if your layout differs.
  - For the sequential arms (A, B, A' frozen) the FINE-TUNE checkpoint is used,
    never the pre-train one.
  - mAP is integrated over confidence; keep conf low (default 1e-3) as in the
    in-domain evaluation, and use the same imgsz (640) used for training.

Usage:
    python smd_eval_crossdomain.py --runs /path/to/runs \
        --data ./data/smd_clean/data.yaml --out ./results/smd_crossdomain.json
"""
import argparse
import json
import os

import numpy as np
from ultralytics import YOLO

# arm -> list of checkpoint paths RELATIVE to --runs (fine-tune phase only).
# Add b2_long seed_2024 once that run finishes (then it becomes n=3).
ARMS = {
    "B1 (random init)": [
        "baselines/B1_random/seed_42/train/weights/best.pt",
        "baselines/B1_random/seed_123/train/weights/best.pt",
        "baselines/B1_random/seed_2024/train/weights/best.pt",
    ],
    "B2 (COCO)": [
        "baselines/B2_coco/seed_42/train/weights/best.pt",
        "baselines/B2_coco/seed_123/train/weights/best.pt",
        "baselines/B2_coco/seed_2024/train/weights/best.pt",
    ],
    "B2-long (volume control)": [
        "b2_long/seed_0042/weights/best.pt",
        "b2_long/seed_0123/weights/best.pt",
        # "b2_long/seed_2024/weights/best.pt",   # add when finished
    ],
    "A (InaTech curated)": [
        "braco_a/seed_0042/finetune/weights/best.pt",
        "braco_a/seed_0123/finetune/weights/best.pt",
        "braco_a/seed_2024/finetune/weights/best.pt",
    ],
    "B (InaTech random)": [
        "braco_b/seed_0042_finetune/weights/best.pt",
        "braco_b/seed_0123_finetune/weights/best.pt",
        "braco_b/seed_2024_finetune/weights/best.pt",
    ],
    "A' frozen": [
        "braco_frozen/seed_0042_finetune/weights/best.pt",
        "braco_frozen/seed_0123_finetune/weights/best.pt",
        "braco_frozen/seed_2024_finetune/weights/best.pt",
    ],
    "A' joint (proposed)": [
        "braco_balanced/seed_0042/weights/best.pt",
        "braco_balanced/seed_0123/weights/best.pt",
        "braco_balanced/seed_2024/weights/best.pt",
    ],
    "A' joint-rand (ablation)": [
        "braco_random_copypaste_v4/seed_0042/weights/best.pt",
        "braco_random_copypaste_v4/seed_0123/weights/best.pt",
        "braco_random_copypaste_v4/seed_2024/weights/best.pt",
    ],
}


def evaluate(runs, data, imgsz, conf, iou):
    results = {}
    for name, rels in ARMS.items():
        m50, m5095 = [], []
        for rel in rels:
            ck = os.path.join(runs, rel)
            if not os.path.exists(ck):
                print(f"   [missing] {name}: {rel}")
                continue
            try:
                r = YOLO(ck).val(data=data, split="test", imgsz=imgsz,
                                 conf=conf, iou=iou, verbose=False, plots=False)
                m50.append(float(r.box.map50))
                m5095.append(float(r.box.map))
            except Exception as e:
                print(f"   [error] {ck}: {e}")
        if m50:
            results[name] = {"mAP50": m50, "mAP50_95": m5095, "n": len(m50)}
    return results


def main():
    ap = argparse.ArgumentParser(description="Zero-shot cross-domain eval on SMD.")
    ap.add_argument("--runs", required=True, help="base dir containing the run folders")
    ap.add_argument("--data", default="./data/smd_clean/data.yaml")
    ap.add_argument("--out", default="./results/smd_crossdomain.json")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    args = ap.parse_args()

    results = evaluate(args.runs, args.data, args.imgsz, args.conf, args.iou)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)

    print(f"\n{'Arm':26} | {'mAP50 (mean+-std)':19} | {'mAP50-95 (mean+-std)':21} | seeds")
    print("-" * 90)
    for name, r in sorted(results.items(), key=lambda kv: -np.mean(kv[1]["mAP50"])):
        a, b = r["mAP50"], r["mAP50_95"]
        print(f"{name:26} | {np.mean(a):.4f} +- {np.std(a):.4f}   "
              f"| {np.mean(b):.4f} +- {np.std(b):.4f}    | {r['n']}")


if __name__ == "__main__":
    main()
