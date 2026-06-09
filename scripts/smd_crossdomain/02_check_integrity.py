#!/usr/bin/env python3
"""
SMD dataset integrity check (leakage + modality breakdown).

For any YOLO-format dataset with split subfolders, reports:
  - the visible on-shore vs on-board (_OB) vs NIR frame breakdown per split, and
  - video-level leakage: source videos whose frames appear in more than one split.

Rationale: Roboflow's default split for the SMD repackaging is per-frame, which
scatters every source video across train/valid/test and inflates any evaluation.
This script makes that leakage visible on the raw download, and verifies that a
re-split (e.g. produced by smd_prepare.py) is leakage-free.

A source video is identified by the filename prefix before '_frame'
(e.g. 'MVI_1646_VIS'); '_OB' marks on-board footage and 'NIR' marks near-infrared.

Usage:
    python smd_check_integrity.py --root ./data/Singapore-maritime-5   # raw download
    python smd_check_integrity.py --root ./data/smd_clean              # cleaned set
"""
import argparse
import glob
import os
from collections import defaultdict


def video_key(fn):
    return fn.split("_frame")[0]


def modality(v):
    if "NIR" in v:
        return "NIR"
    if "_OB" in v:
        return "onboard(OB)"
    return "onshore(VIS)"


def find_splits(root):
    return [d for d in ("train", "val", "valid", "test")
            if os.path.isdir(os.path.join(root, d, "images"))]


def main():
    ap = argparse.ArgumentParser(description="Check SMD split integrity (leakage + modality).")
    ap.add_argument("--root", required=True, help="dataset root containing <split>/images/")
    ap.add_argument("--show", type=int, default=15, help="max leaking videos to list")
    args = ap.parse_args()

    splits = find_splits(args.root)
    if not splits:
        raise SystemExit(f"No split with images/ found under {args.root}")

    vid_to_splits = defaultdict(set)
    counts = defaultdict(int)
    for sp in splits:
        for img in glob.glob(os.path.join(args.root, sp, "images", "*")):
            v = video_key(os.path.basename(img))
            vid_to_splits[v].add(sp)
            counts[(sp, modality(v))] += 1

    print("=== modality breakdown per split ===")
    mods = ["onshore(VIS)", "onboard(OB)", "NIR"]
    for sp in splits:
        line = " | ".join(f"{m}: {counts.get((sp, m), 0):5}" for m in mods)
        print(f"{sp:6} | {line}")

    leaks = {v: s for v, s in vid_to_splits.items() if len(s) > 1}
    print(f"\n=== leakage ===")
    print(f"videos total: {len(vid_to_splits)} | videos in >1 split: {len(leaks)}")
    for v in sorted(leaks)[:args.show]:
        print(f"  {v}: {sorted(vid_to_splits[v])}")
    if len(leaks) == 0:
        print("  (none -- split is video-disjoint)")


if __name__ == "__main__":
    main()
