"""Channel-Independent Patch Transformer (Exp 16).

Each channel is patched and processed independently through a shared transformer.
The output head is channel-independent: Linear(N→T) temporal + Linear(d_model→1) channel.
Total params ~72K regardless of D. This is the PatchTST CI design, right-sized.
"""

import torch
import torch.nn as nn


class CITransformer(nn.Module):
    """Channel-Independent Patch Transformer.

    Key design: every channel goes through the same transformer independently.
    Shared weights across channels = implicit regularization (like DLinear).
    """

    def __init__(
        self,
        input_dim: int,
        forecast_length: int,
        lookback_length: int,
        patch_size: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.forecast_length = forecast_length
        self.lookback_length = lookback_length
        self.patch_size = patch_size
        self.n_patches = lookback_length // patch_size
        self.d_model = d_model

        # Patch embedding: P → d_model (channel-independent, shared)
        self.patch_embed = nn.Linear(patch_size, d_model)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Shared transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CI-Head: temporal projection (shared across channels)
        # Instead of flatten→giant linear, project N→T per d_model dimension
        self.head_norm = nn.LayerNorm(d_model)
        self.temporal_proj = nn.Linear(self.n_patches, forecast_length)
        self.channel_proj = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, lookback: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lookback: [B, H, D]
        Returns:
            forecast: [B, T, D]
        """
        B, H, D = lookback.shape

        # Channel-independent patching: treat each channel as a separate sample
        # [B, H, D] → [B, D, H] → [B*D, H]
        x = lookback.transpose(1, 2).reshape(B * D, H)

        # Create patches: [B*D, H] → [B*D, N, P]
        x = x[:, :self.n_patches * self.patch_size]
        x = x.reshape(B * D, self.n_patches, self.patch_size)

        # Patch embedding: [B*D, N, P] → [B*D, N, d_model]
        x = self.patch_embed(x)
        x = x + self.pos_embed

        # Shared transformer: [B*D, N, d_model]
        x = self.transformer(x)
        x = self.head_norm(x)

        # CI-Head: temporal projection
        # [B*D, N, d_model] → transpose → [B*D, d_model, N]
        x = x.transpose(1, 2)
        # Linear(N → T): [B*D, d_model, T]
        x = self.temporal_proj(x)
        # [B*D, T, d_model]
        x = x.transpose(1, 2)
        # Linear(d_model → 1): [B*D, T, 1]
        x = self.channel_proj(x)

        # Reshape back: [B*D, T, 1] → [B, D, T] → [B, T, D]
        x = x.squeeze(-1).reshape(B, D, self.forecast_length)
        x = x.transpose(1, 2)

        return x


def create_model(config: dict) -> CITransformer:
    data_config = config.get("data", {})
    input_dim = 1 if data_config.get("univariate", False) else 7
    lookback_length = data_config.get("lookback_length", 336)
    forecast_length = data_config.get("forecast_length", 168)

    patch_size = min(16, lookback_length // 8)
    patch_size = max(4, patch_size)

    return CITransformer(
        input_dim=input_dim,
        forecast_length=forecast_length,
        lookback_length=lookback_length,
        patch_size=patch_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.3,
    )
