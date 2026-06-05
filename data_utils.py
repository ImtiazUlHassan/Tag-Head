import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
from pytorchvideo.transforms import RandomResizedCrop, Normalize
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo

# ImageNet-style mean/std used throughout
MEAN = [0.45, 0.45, 0.45]
STD  = [0.225, 0.225, 0.225]

# DALI reads this many raw frames per chunk; videos longer than this are
# split into multiple chunks that are then reassembled before sampling.
_DALI_SEQUENCE_LENGTH = 40


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_video_metadata(csv_path: str, video_root: str, shuffle: bool = True, seed: int = 42):
    """
    Read a split CSV and return parallel lists consumed by the DALI pipeline.

    CSV columns expected: FileName, Class, TotalFrames, ClassEncoded

    Returns:
        file_paths  : list of absolute video paths
        labels      : list of integer class indices
        total_frames: list of frame counts per video
        boundaries  : cumulative chunk boundaries (used to reassemble chunks)
    """
    df = pd.read_csv(csv_path)
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    file_paths = df.apply(
        lambda r: os.path.join(video_root, str(r["Class"]), r["FileName"]), axis=1
    ).tolist()
    labels       = df["ClassEncoded"].tolist()
    total_frames = df["TotalFrames"].tolist()

    chunks_per_video = [
        int(np.ceil(tf / _DALI_SEQUENCE_LENGTH)) for tf in total_frames
    ]
    boundaries = np.cumsum(chunks_per_video)

    return file_paths, labels, total_frames, boundaries


# ---------------------------------------------------------------------------
# DALI video pipeline
# ---------------------------------------------------------------------------

@pipeline_def
def video_pipe(filenames, labels):
    """DALI pipeline: decode + resize video chunks from disk."""
    videos, labels = fn.readers.video_resize(
        device="gpu",
        filenames=filenames,
        labels=labels,
        sequence_length=_DALI_SEQUENCE_LENGTH,
        shard_id=0,
        num_shards=1,
        random_shuffle=False,
        pad_sequences=True,
        enable_frame_num=False,
        name="Reader",
        resize_shorter=128,
    )
    info = fn.get_property(videos, key="source_info")
    info = fn.pad(info)
    return videos / 255.0, labels, info


def decode_filename(info_field) -> str:
    return "".join(chr(i) for i in info_field)


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def uniform_sample(chunks, total_frames: int, num_samples: int) -> torch.Tensor:
    """
    Concatenate DALI chunks, trim to `total_frames`, then uniformly sample
    `num_samples` frames.

    Args:
        chunks      : list of tensors from consecutive DALI iterations
        total_frames: number of valid frames in this video
        num_samples : target frame count (32 for HAA500, 64 for FineGym)

    Returns:
        Tensor of shape [num_samples, H, W, C]
    """
    full = torch.cat(chunks, dim=0)[:total_frames]
    indices = torch.linspace(0, total_frames - 1, steps=num_samples).long()
    return full[indices]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        NormalizeVideo(MEAN, STD),
        RandomResizedCrop(
            target_height=112,
            target_width=112,
            scale=(0.5, 1.0),
            aspect_ratio=(1.0, 1.0),
            interpolation="bilinear",
        ),
    ])


def get_val_transforms():
    return transforms.Compose([
        NormalizeVideo(MEAN, STD),
        CenterCropVideo(112),
    ])
