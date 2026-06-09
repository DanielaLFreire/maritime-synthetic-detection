#!/usr/bin/env python3
"""
SMD cross-domain evaluation set -- download and preparation.

Builds the zero-shot cross-domain evaluation set used in the paper
(Section "Zero-shot cross-domain generalisation") from the Singapore Maritime
Dataset (SMD), as repackaged on Roboflow Universe
(maritime-cumkb/singapore-maritime, version 5).

Pipeline:
  1. Download the dataset from Roboflow (YOLO format).
  2. Keep ONLY the visible on-shore footage (drop on-board '_OB' and NIR).
  3. Remap the 9 original classes to a single 'vessel' class
     (Boat, Ferry, Kayak, Sail boat, Speed boat, Vessel-ship);
     discard Buoy, Flying bird-plane and Other.
  4. Re-split BY VIDEO to preclude train/val/test leakage. Roboflow's default
     split is per-frame and scatters every source video across all splits.
  5. Temporally subsample frames per video to reduce near-duplicates.
  6. Write a single-class data.yaml with an absolute 'path:' root.

By default everything is routed to the 'test' split (ratio 0/0/1): the
configuration used for the zero-shot probe (no training is done on SMD).

Original dataset: Prasad et al., "Video processing from electro-optical
sensors for object detection and tracking in a maritime environment: A
survey", IEEE T-ITS 18(8):1993-2016, 2017. doi:10.1109/TITS.2016.2634580

The Roboflow API key MUST be provided via the ROBOFLOW_API_KEY environment
variable (never hard-code it):
    export ROBOFLOW_API_KEY="xxxx"

Usage:
    python smd_prepare.py --out ./data/smd_clean
    python smd_prepare.py --raw ./data/Singapore-maritime-5 --skip-download \
        --stride 4 --ratio 0 0 1 --seed 42
"""
import argparse
import glob
import os
import random
import re
import shutil
from collections import defaultdict

# --- Reproducibility-critical constants (documented; change with care) ------
ROBOFLOW_WORKSPACE = "maritime-cumkb"
ROBOFLOW_PROJECT = "singapore-maritime"
ROBOFLOW_VERSION = 5
EXPORT_FORMAT = "yolov11"

# Original SMD/Roboflow class indices kept and merged into 'vessel':
#   0 Boat, 2 Ferry, 4 Kayak, 6 Sail boat, 7 Speed boat, 8 Vessel-ship
# Discarded (not vessels): 1 Buoy, 3 Flying bird-plane, 5 Other
VESSEL_IDS = {0, 2, 4, 6, 7, 8}

# Roboflow's default split folder names.
RAW_SPLITS = ("train", "valid", "test")


def video_key(fn):
    """Video identifier = filename prefix before '_frame' (e.g. 'MVI_1646_VIS')."""
    return fn.split("_frame")[0]


def frame_num(fn):
    m = re.search(r"_frame(\d+)", fn)
    return int(m.group(1)) if m else 0


def keep_video(v):
    """Keep only visible on-shore videos: drop on-board (_OB) and NIR."""
    return ("NIR" not in v) and ("_OB" not in v)


def remap_label(path):
    """Return YOLO lines remapped to single class 0 (vessel); [] if none/missing."""
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        p = line.split()
        if p and int(p[0]) in VESSEL_IDS:
            out.append("0 " + " ".join(p[1:]))
    return out


def download(raw_dir):
    if os.path.exists(os.path.join(raw_dir, "data.yaml")):
        print(f"[download] already present in {raw_dir}, skipping.")
        return
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY in the environment before downloading.")
    from roboflow import Roboflow

    rf = Roboflow(api_key=key)
    (rf.workspace(ROBOFLOW_WORKSPACE)
       .project(ROBOFLOW_PROJECT)
       .version(ROBOFLOW_VERSION)
       .download(EXPORT_FORMAT, location=raw_dir))


def write_yaml(base):
    with open(os.path.join(base, "data.yaml"), "w") as f:
        f.write(
            f"path: {os.path.abspath(base)}\n"
            "train: train/images\n"
            "val: val/images\n"
            "test: test/images\n"
            "nc: 1\n"
            "names: ['vessel']\n"
        )


def build(raw_dir, out_dir, stride, ratio, seed):
    shutil.rmtree(out_dir, ignore_errors=True)

    frames = defaultdict(list)
    for sp in RAW_SPLITS:
        for img in glob.glob(os.path.join(raw_dir, sp, "images", "*")):
            fn = os.path.basename(img)
            v = video_key(fn)
            if not keep_video(v):
                continue
            lbl = os.path.join(raw_dir, sp, "labels", os.path.splitext(fn)[0] + ".txt")
            frames[v].append((img, lbl, fn))

    # Split BY VIDEO (group-aware): every frame of a video goes to one split.
    vids = sorted(frames)
    random.Random(seed).shuffle(vids)
    n = len(vids)
    n_tr, n_va = int(n * ratio[0]), int(n * ratio[1])
    split_of = {
        v: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
        for i, v in enumerate(vids)
    }

    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(out_dir, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, sp, "labels"), exist_ok=True)

    stats = defaultdict(lambda: {"imgs": 0, "vids": set(), "boxes": 0})
    for v, items in frames.items():
        sp = split_of[v]
        # temporal subsampling: keep 1 of every `stride` frames per video
        for img, lbl, fn in sorted(items, key=lambda t: frame_num(t[2]))[::stride]:
            labels = remap_label(lbl)
            if not labels:  # frame has no vessel after remap -> drop
                continue
            shutil.copy(img, os.path.join(out_dir, sp, "images", fn))
            stem = os.path.splitext(fn)[0]
            with open(os.path.join(out_dir, sp, "labels", stem + ".txt"), "w") as f:
                f.write("\n".join(labels) + "\n")
            stats[sp]["imgs"] += 1
            stats[sp]["vids"].add(v)
            stats[sp]["boxes"] += len(labels)

    write_yaml(out_dir)
    print(f"=== SMD clean (stride={stride}, ratio={ratio}, seed={seed}) ===")
    for sp in ("train", "val", "test"):
        s = stats[sp]
        print(f"{sp:5} | imgs: {s['imgs']:5} | videos: {len(s['vids']):3} | boxes: {s['boxes']:6}")


def main():
    ap = argparse.ArgumentParser(description="Download and prepare the SMD cross-domain eval set.")
    ap.add_argument("--raw", default="./data/Singapore-maritime-5", help="Roboflow download dir")
    ap.add_argument("--out", default="./data/smd_clean", help="output clean dataset dir")
    ap.add_argument("--stride", type=int, default=4, help="keep 1 of every N frames per video")
    ap.add_argument("--ratio", type=float, nargs=3, default=[0, 0, 1],
                    metavar=("TR", "VA", "TE"), help="train/val/test split fractions, by video")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-download", action="store_true",
                    help="assume the Roboflow data already exists in --raw")
    args = ap.parse_args()

    if not args.skip_download:
        download(args.raw)
    build(args.raw, args.out, args.stride, tuple(args.ratio), args.seed)


if __name__ == "__main__":
    main()
