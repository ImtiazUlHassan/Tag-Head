"""
TAG-Head training script.

Example usage:
    python train.py --dataset haa500 \
        --csv_train /path/to/haa500train.csv \
        --csv_val   /path/to/haa500val.csv \
        --video_dir /path/to/haa500/videos/ \
        --output_dir ./runs/haa500

    python train.py --dataset gym99 \
        --csv_train /path/to/gym99train_filtered.csv \
        --csv_val   /path/to/gym99val_filtered.csv \
        --video_dir /path/to/gym99/videos/ \
        --output_dir ./runs/gym99

    python train.py --dataset gym288 \
        --csv_train /path/to/FineGym288_train.csv \
        --csv_val   /path/to/FineGym288_val.csv \
        --video_dir /path/to/gym288/videos/ \
        --output_dir ./runs/gym288
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_utils import (
    get_train_transforms, get_val_transforms,
    load_video_metadata, set_seed, uniform_sample, video_pipe,
)
from model import DATASET_CONFIGS, build_model

# ---------------------------------------------------------------------------
# Dataset-level training defaults
# ---------------------------------------------------------------------------

TRAIN_DEFAULTS = {
    "haa500": dict(num_frames=32, epochs=80,  batch_size=8, lr=1e-5),
    "gym99":  dict(num_frames=64, epochs=60,  batch_size=8, lr=1e-5),
    "gym288": dict(num_frames=64, epochs=60,  batch_size=8, lr=1e-5),
}


def parse_args():
    p = argparse.ArgumentParser(description="Train TAG-Head")

    # Required
    p.add_argument("--dataset",   required=True, choices=list(DATASET_CONFIGS),
                   help="Target dataset")
    p.add_argument("--csv_train", required=True, help="Path to training split CSV")
    p.add_argument("--csv_val",   required=True, help="Path to validation split CSV")
    p.add_argument("--video_dir", required=True,
                   help="Root directory containing class-named subdirectories with videos")
    p.add_argument("--output_dir", required=True, help="Where to save checkpoints")

    # Training hyper-parameters (default from dataset presets)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--seed",       type=int,   default=42)

    # Optional WandB logging
    p.add_argument("--wandb",         action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", default="TAG-Head",
                   help="W&B project name")
    p.add_argument("--wandb_run",     default=None,
                   help="W&B run name (default: auto)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loader helpers
# ---------------------------------------------------------------------------

def make_loader(csv_path, video_dir, shuffle):
    file_paths, labels, total_frames, boundaries = load_video_metadata(
        csv_path, video_dir, shuffle=shuffle
    )
    loader = DALIGenericIterator(
        [video_pipe(batch_size=1, num_threads=4, device_id=0,
                    filenames=file_paths, labels=labels)],
        ["videos", "labels", "info"],
        reader_name="Reader",
    )
    return loader, file_paths, labels, total_frames, boundaries


def collect_batch(loader, boundaries, total_frames, num_frames,
                  transform, batch_size):
    """
    Iterate over DALI output and collect one batch of (video, label) pairs.
    Returns a generator that yields (videos_tensor, labels_tensor) batches.
    """
    chunks, label_batch, video_batch = [], [], []
    video_idx = 0

    for i, data in enumerate(loader):
        chunk      = data[0]["videos"][0]
        label_id   = int(data[0]["labels"])
        chunks.append(chunk)

        if (i + 1) == boundaries[video_idx]:
            frames = uniform_sample(chunks, total_frames[video_idx], num_frames)
            frames = transform(frames.permute(3, 0, 1, 2))   # [C, T, H, W]
            video_batch.append(frames)
            label_batch.append(label_id)
            chunks = []
            video_idx += 1

            if len(video_batch) == batch_size:
                yield (
                    torch.stack(video_batch).cuda(),
                    torch.tensor(label_batch, dtype=torch.long).cuda(),
                )
                video_batch, label_batch = [], []

    # Leftover partial batch
    if video_batch:
        yield (
            torch.stack(video_batch).cuda(),
            torch.tensor(label_batch, dtype=torch.long).cuda(),
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(model, loader, total_frames, boundaries, num_frames,
              transform, batch_size, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for videos, labels in collect_batch(
            loader, boundaries, total_frames, num_frames,
            transform, batch_size
        ):
            outputs = model(videos)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    avg_loss = total_loss / total if total else 0.0
    accuracy = 100.0 * correct / total if total else 0.0
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # Fill in dataset defaults for any unspecified hyper-parameters
    defaults = TRAIN_DEFAULTS[args.dataset]
    num_frames = defaults["num_frames"]
    epochs     = args.epochs     or defaults["epochs"]
    batch_size = args.batch_size or defaults["batch_size"]
    lr         = args.lr         or defaults["lr"]

    os.makedirs(args.output_dir, exist_ok=True)

    # Optional WandB
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"TAG-Head-{args.dataset}",
            config=dict(dataset=args.dataset, epochs=epochs,
                        batch_size=batch_size, lr=lr),
        )

    # Model
    model     = build_model(args.dataset).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    transform_train = get_train_transforms()
    transform_val   = get_val_transforms()

    # Resume from checkpoint if present
    ckpt_path  = os.path.join(args.output_dir, "last_checkpoint.pth")
    start_epoch, best_val_acc = 0, 0.0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cuda")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_acc  = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best val acc {best_val_acc:.2f}%")

    for epoch in range(start_epoch, epochs):
        # Re-shuffle training data each epoch
        train_loader, _, _, train_total_frames, train_boundaries = make_loader(
            args.csv_train, args.video_dir, shuffle=True
        )

        train_loss, train_acc = run_epoch(
            model, train_loader, train_total_frames, train_boundaries,
            num_frames, transform_train, batch_size, criterion, optimizer
        )

        val_loader, _, _, val_total_frames, val_boundaries = make_loader(
            args.csv_val, args.video_dir, shuffle=False
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, val_total_frames, val_boundaries,
            num_frames, transform_val, batch_size, criterion
        )

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}%"
        )

        if args.wandb:
            import wandb
            wandb.log(dict(
                epoch=epoch + 1,
                train_loss=train_loss, train_acc=train_acc,
                val_loss=val_loss,   val_acc=val_acc,
            ))

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_model.pth"))
            print(f"  -> New best: {best_val_acc:.2f}%")

        # Save rolling checkpoint
        torch.save(
            dict(
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                best_val_acc=best_val_acc,
            ),
            ckpt_path,
        )

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
