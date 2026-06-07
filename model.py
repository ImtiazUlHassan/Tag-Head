import torch
import torch.nn as nn
from torch_geometric.nn import APPNP, global_mean_pool
from torch_geometric.data import Data, Batch


def _build_backbone():
    """R(2+1)D-34 pretrained on IG-65M, with avgpool and fc removed."""
    model = torch.hub.load(
        "moabitcoin/ig65m-pytorch",
        "r2plus1d_34_32_ig65m",
        num_classes=359,
        pretrained=True,
    )
    return nn.Sequential(*list(model.children())[:-2])


class TAGHead(nn.Module):
    """
    Time-Aligned Graph Head for Plug-and-Play Fine-grained Action Recognition.

    Wraps a frozen/fine-tuned R(2+1)D-34 backbone with:
      1. Learnable 3D spatio-temporal positional encoding (ST-PE)
      2. Multi-layer Transformer encoder for global context
      3. Spatio-temporal graph with intra-frame fully-connected (IF-FC)
         and time-aligned temporal (TAT) edges
      4. Parameter-free APPNP propagation for feature refinement
      5. Global mean pooling + linear classifier

    Args:
        num_classes (int): Number of action categories.
        num_temporal_tokens (int): Temporal tokens from backbone
            (4 for 32-frame input, 8 for 64-frame input).
        d_model (int): Token feature dimension (fixed at 512 by the backbone).
        nhead (int): Number of Transformer attention heads.
        num_transformer_layers (int): Depth of Transformer encoder.
        dim_feedforward (int): FFN width inside Transformer.
        dropout (float): Dropout rate.
        appnp_k (int): Number of APPNP propagation steps.
        appnp_alpha (float): APPNP teleport probability.
    """

    def __init__(
        self,
        num_classes: int,
        num_temporal_tokens: int,
        d_model: int = 512,
        nhead: int = 8,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.5,
        appnp_k: int = 2,
        appnp_alpha: float = 0.1,
    ):
        super().__init__()

        self.csnmodel = _build_backbone()

        T, H, W = num_temporal_tokens, 7, 7
        N = T * H * W  # total graph nodes

        # ST-PE: learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, N, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # Pre-compute graph edges (no learnable params)
        self.edge_indexes = _build_edge_index(T, H, W)

        # APPNP (parameter-free propagation)
        self.appnp = APPNP(K=appnp_k, alpha=appnp_alpha)

        # Classifier
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor of shape [B, 3, T_in, 112, 112].
        Returns:
            logits: [B, num_classes]
        """
        # Backbone -> [B, 512, T, 7, 7]
        features = self.csnmodel(x)
        B = features.shape[0]

        # Flatten to token sequence -> [B, N, 512]
        tokens = features.flatten(start_dim=2).permute(0, 2, 1)

        # Add positional encoding
        tokens = tokens + self.pos_embedding

        # Transformer encoder -> [B, N, 512]
        tokens = self.transformer(tokens)

        # Build graph batch
        edge_idx = self.edge_indexes.to(features.device)
        graphs = [Data(x=tokens[i], edge_index=edge_idx) for i in range(B)]
        batch = Batch.from_data_list(graphs)

        # APPNP feature refinement
        h = self.appnp(batch.x, batch.edge_index)

        # Global spatio-temporal mean pooling -> [B, 512]
        h = global_mean_pool(h, batch.batch)

        return self.mlp_head(h)


def _build_edge_index(T: int, H: int, W: int) -> torch.Tensor:
    """
    Builds the spatio-temporal edge set:
      - IF-FC: every spatial node fully connected to every other
               spatial node within the same frame.
      - TAT:   each spatial node connected to the node at the same
               spatial position in the adjacent frame (bidirectional).
    """
    P = H * W
    edges = []

    # IF-FC edges
    for t in range(T):
        base = t * P
        for i in range(P):
            for j in range(P):
                if i != j:
                    edges.append([base + i, base + j])

    # TAT edges
    for t in range(T - 1):
        for p in range(P):
            u = t * P + p
            v = (t + 1) * P + p
            edges.append([u, v])
            edges.append([v, u])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ---------------------------------------------------------------------------
# Dataset presets (architecture hyper-parameters that match saved checkpoints)
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "haa500": dict(
        num_classes=500,
        num_frames=32,
        num_temporal_tokens=4,
        dim_feedforward=512,
    ),
    "gym99": dict(
        num_classes=99,
        num_frames=64,
        num_temporal_tokens=8,
        dim_feedforward=1024,
    ),
    "gym288": dict(
        num_classes=288,
        num_frames=64,
        num_temporal_tokens=8,
        dim_feedforward=2048,
    ),
}


def build_model(dataset: str, **kwargs) -> TAGHead:
    """Instantiate TAGHead with the preset config for a given dataset."""
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(DATASET_CONFIGS)}")
    cfg = {**DATASET_CONFIGS[dataset], **kwargs}
    cfg.pop("num_frames", None)  # num_frames is a data-loading param, not a model param
    return TAGHead(**cfg)
