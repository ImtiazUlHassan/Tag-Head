"""
TAG-Head evaluation script — computes Top-1 accuracy and Mean Class Accuracy (MCA).

Example usage:
    # Evaluate HAA500 on the test split
    python evaluate.py --dataset haa500 \
        --csv      /path/to/haa500test.csv \
        --video_dir /path/to/haa500/videos/ \
        --weights  weights/haa500_best.pth

    # Evaluate Gym99 on the validation split
    python evaluate.py --dataset gym99 \
        --csv      /path/to/gym99val_filtered.csv \
        --video_dir /path/to/gym99/videos/ \
        --weights  weights/gym99_best.pth

    # Evaluate Gym288 on the validation split
    python evaluate.py --dataset gym288 \
        --csv      /path/to/FineGym288_val.csv \
        --video_dir /path/to/gym288/videos/ \
        --weights  weights/gym288_last.pth
"""

import argparse
from collections import defaultdict

import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from data_utils import (
    get_val_transforms, load_video_metadata,
    set_seed, uniform_sample, video_pipe,
)
from model import DATASET_CONFIGS, build_model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TAG-Head")
    p.add_argument("--dataset",   required=True, choices=list(DATASET_CONFIGS))
    p.add_argument("--csv",       required=True, help="Split CSV to evaluate on")
    p.add_argument("--video_dir", required=True, help="Root video directory")
    p.add_argument("--weights",   required=True, help="Path to .pth checkpoint")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def evaluate(model, csv_path, video_dir, num_frames, num_classes):
    transform = get_val_transforms()

    file_paths, labels, total_frames, boundaries = load_video_metadata(
        csv_path, video_dir, shuffle=False
    )

    loader = DALIGenericIterator(
        [video_pipe(batch_size=1, num_threads=4, device_id=0,
                    filenames=file_paths, labels=labels)],
        ["videos", "labels", "info"],
        reader_name="Reader",
    )

    # Per-class accumulators for MCA
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)
    overall_correct, overall_total = 0, 0

    model.eval()
    chunks    = []
    video_idx = 0

    with torch.no_grad():
        for i, data in enumerate(loader):
            chunk    = data[0]["videos"][0]
            label_id = int(data[0]["labels"])
            chunks.append(chunk)

            if (i + 1) == boundaries[video_idx]:
                frames = uniform_sample(chunks, total_frames[video_idx], num_frames)
                frames = transform(frames.permute(3, 0, 1, 2))  # [C, T, H, W]
                frames = frames.unsqueeze(0).cuda()             # [1, C, T, H, W]

                logits = model(frames)
                pred   = logits.argmax(dim=1).item()

                class_correct[label_id] += int(pred == label_id)
                class_total[label_id]   += 1
                overall_correct         += int(pred == label_id)
                overall_total           += 1

                chunks    = []
                video_idx += 1

    top1 = 100.0 * overall_correct / overall_total

    # MCA: mean of per-class accuracy
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

    cfg        = DATASET_CONFIGS[args.dataset]
    num_frames = cfg["num_frames"]
    num_classes = cfg["num_classes"]

    # Build model and load weights
    model = build_model(args.dataset).cuda()

    state = torch.load(args.weights, map_location="cuda")
    # Support both bare state-dicts and checkpoint dicts
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
