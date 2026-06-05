# Pretrained Weights

Download the pretrained TAG-Head checkpoints and place them in this folder.

| File | Dataset | Top-1 | MCA |
|---|---|---|---|
| `haa500_best.pth` | HAA500 | 86.1% | — |
| `gym99_best.pth` | FineGym99 | 95.6% | 93.8% |
| `gym288_last.pth` | FineGym288 | 92.2% | 68.6% |

> Download link: _[to be added — upload to Google Drive or HuggingFace and paste link here]_

All weights were produced by training with the R(2+1)D-34 backbone
pretrained on IG-65M (automatically downloaded via `torch.hub` on first run).
