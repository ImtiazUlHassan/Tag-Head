"""
TAG-Head evaluation script using Decord — an alternative to evaluate.py
for users who have difficulty installing NVIDIA DALI.

Decord is codec-agnostic and easier to install:
    pip install decord

Note: results may differ slightly from the DALI-based evaluate.py (~1-2%)
due to minor differences in the video resize implementation between DALI
and PyTorch's F.interpolate. The DALI script reproduces the exact paper results.

Example usage:
    # HAA500
    python evaluate_decord.py --dataset haa500 \
        --csv       data/haa500test.csv \
        --video_dir /path/to/haa500/videos/ \
        --weights   weights/haa500_best.pth

    # FineGym99
    python evaluate_decord.py --dataset gym99 \
        --csv       data/gym99val_filtered.csv \
        --video_dir /path/to/gym99/videos/ \
        --weights   weights/gym99_best.pth

    # FineGym288
    python evaluate_decord.py --dataset gym288 \
        --csv       data/FineGym288_val.csv \
        --video_dir /path/to/gym288/videos/ \
        --weights   weights/gym288_best.pth
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

from data_utils import get_val_transforms, set_seed
from model import DATASET_CONFIGS, build_model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TAG-Head (Decord backend)")
    p.add_argument("--dataset",   required=True, choices=list(DATASET_CONFIGS))
    p.add_argument("--csv",       required=True, help="Split CSV to evaluate on")
    p.add_argument("--video_dir", required=True, help="Root video directory")
    p.add_argument("--weights",   required=True, help="Path to .pth checkpoint")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def load_video(video_path: str, num_frames: int) -> torch.Tensor:
    """
    Read a video with Decord, uniformly sample num_frames frames,
    resize shorter side to 128, and return [C, T, H, W] float in [0, 1].
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    T  = len(vr)

    indices = np.linspace(0, T - 1, num=num_frames, dtype=int)
    frames  = vr.get_batch(indices).asnumpy()   # [T, H, W, C] uint8

    frames = torch.from_numpy(frames).float() / 255.0   # [T, H, W, C]
    frames = frames.permute(3, 0, 1, 2)                 # [C, T, H, W]

    # Resize shorter side to 128 — matches DALI resize_shorter=128
    C, T, H, W = frames.shape
    scale  = 128.0 / min(H, W)
    new_H  = max(int(H * scale), 128)
    new_W  = max(int(W * scale), 128)
    frames = F.interpolate(
        frames.reshape(C * T, 1, H, W),
        size=(new_H, new_W),
        mode="bilinear",
        align_corners=False,
    ).reshape(C, T, new_H, new_W)

    return frames


def evaluate(model, csv_path, video_dir, num_frames, num_classes):
    transform = get_val_transforms()
    df = pd.read_csv(csv_path)

    class_correct = defaultdict(int)
    class_total   = defaultdict(int)
    overall_correct, overall_total = 0, 0
    skipped = 0

    model.eval()
    with torch.no_grad():
        for _, row in df.iterrows():
            video_path = os.path.join(video_dir, str(row["Class"]), row["FileName"])
            label_id   = int(row["ClassEncoded"])

            if not os.path.exists(video_path):
                skipped += 1
                continue

            try:
                frames = load_video(video_path, num_frames)
            except Exception as e:
                skipped += 1
                continue

            frames = transform(frames)          # NormalizeVideo + CenterCropVideo(112)
            frames = frames.unsqueeze(0).cuda() # [1, C, T, H, W]

            pred = model(frames).argmax(dim=1).item()

            class_correct[label_id] += int(pred == label_id)
            class_total[label_id]   += 1
            overall_correct         += int(pred == label_id)
            overall_total           += 1

    if skipped:
        print(f"Warning: skipped {skipped} videos (missing or unreadable)")

    top1 = 100.0 * overall_correct / overall_total

    per_class_acc = [
        class_correct[c] / class_total[c]
        for c in range(num_classes)
        if class_total[c] > 0
    ]
    mca = 100.0 * sum(per_class_acc) / len(per_class_acc)

    return top1, mca


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg         = DATASET_CONFIGS[args.dataset]
    num_frames  = cfg["num_frames"]
    num_classes = cfg["num_classes"]

    model = build_model(args.dataset).cuda()

    state = torch.load(args.weights, map_location="cuda")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    print(f"Loaded weights from {args.weights}")

    top1, mca = evaluate(model, args.csv, args.video_dir, num_frames, num_classes)
    print(f"\nDataset : {args.dataset.upper()}")
    print(f"Top-1   : {top1:.1f}%")
    print(f"MCA     : {mca:.1f}%")


if __name__ == "__main__":
    main()
